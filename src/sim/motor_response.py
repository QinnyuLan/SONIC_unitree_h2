"""Opt-in effective motor-response candidates, not identified hardware parameters.

Groups contain actual actuator indices, after any closed-chain torque mapping.
Each selected motor applies gain, a zero-order-hold first-order response, then
an effort cap. Unselected motors pass through; an empty configuration is an
exact copy-only no-op. The native simulator effort clipping remains separate.
"""

import math
from numbers import Integral, Real

import numpy as np

H2_MOTOR_GROUP_TRANSMISSIONS = {
    "left_ankle": "left_ankle_loop",
    "right_ankle": "right_ankle_loop",
    "left_knee": "left_knee_loop",
    "right_knee": "right_knee_loop",
}


def _vector(value, name, size=None):
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real vector") from exc
    if result.ndim != 1 or result.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a finite real vector")
    if size is not None and result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _scalar(value, name, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return result


class MotorResponse:
    """Bounded candidate dynamics on explicitly selected motor groups only.

    ``time_constant_ms`` is a lag, not a pure transport delay. Filter state is
    the pre-limit effort; saturation does not feed back into that state.
    ``reset(initial)`` sets this state directly in effort units, without gain.
    Diagnostics describe the most recent successful call. Invalid input leaves
    both state and diagnostics unchanged. Input arrays are never modified.
    """

    _FIELDS = {"gain", "time_constant_ms", "limit_scale"}

    def __init__(self, nominal_limits: np.ndarray, groups: dict[str, list[int]],
                 config: dict[str, dict]):
        limits = _vector(nominal_limits, "nominal_limits")
        if not limits.size or np.any(limits <= 0):
            raise ValueError("nominal_limits must be nonempty and positive")
        with np.errstate(over="ignore", invalid="ignore"):
            self.nominal_limits = limits.astype(np.float64, copy=True)
        if not np.all(np.isfinite(self.nominal_limits)):
            raise ValueError("nominal_limits exceed float64 range")
        self.size = limits.size
        if not isinstance(groups, dict) or not isinstance(config, dict):
            raise ValueError("groups and config must be dictionaries")
        used = set()
        validated_groups = {}
        for name, indices in groups.items():
            if not isinstance(name, str) or not name:
                raise ValueError("group names must be nonempty strings")
            if (not isinstance(indices, (list, tuple, np.ndarray))
                    or isinstance(indices, np.ndarray) and indices.ndim != 1
                    or not len(indices)):
                raise ValueError(f"group {name} must contain motor indices")
            checked = []
            for index in indices:
                if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral):
                    raise ValueError(f"group {name} indices must be integers")
                index = int(index)
                if not 0 <= index < self.size:
                    raise ValueError(f"group {name} motor index {index} is out of bounds")
                if index in used:
                    raise ValueError(f"motor index {index} occurs in duplicate or overlapping groups")
                used.add(index)
                checked.append(index)
            validated_groups[name] = np.asarray(checked, dtype=np.intp)
        if set(config) - set(validated_groups):
            raise ValueError("config contains an unknown motor group")

        self.selected = np.zeros(self.size, dtype=bool)
        self.gain = np.ones(self.size)
        self.time_constant_s = np.zeros(self.size)
        self.limits = self.nominal_limits.copy()
        for name, parameters in config.items():
            if not isinstance(parameters, dict) or set(parameters) - self._FIELDS:
                raise ValueError(f"group {name} contains invalid response parameters")
            gain = _scalar(parameters.get("gain", 1), "gain", .4, 1.2)
            tau = _scalar(parameters.get("time_constant_ms", 0), "time_constant_ms", 0, 50)
            scale = _scalar(parameters.get("limit_scale", 1), "limit_scale", .25, 1)
            indices = validated_groups[name]
            self.selected[indices] = True
            self.gain[indices] = gain
            self.time_constant_s[indices] = tau * .001
            self.limits[indices] *= scale
        self.enabled = bool(config)
        self._motor_parameters = tuple(
            (int(index), float(self.gain[index]), float(self.time_constant_s[index]),
             float(self.limits[index])) for index in np.flatnonzero(self.selected)
        )
        self._cached_dt = None
        self._cached_coefficients = ()
        self.reset()

    def reset(self, initial: np.ndarray | None = None):
        with np.errstate(over="ignore", invalid="ignore"):
            state = (np.zeros(self.size) if initial is None
                     else _vector(initial, "initial", self.size).astype(np.float64, copy=True))
        if not np.all(np.isfinite(state)):
            raise ValueError("initial exceeds float64 range")
        applied = state.copy()
        applied[self.selected] = np.clip(state[self.selected],
                                         -self.limits[self.selected], self.limits[self.selected])
        self._state = state
        self.last_prelimit = state.copy()
        self.last_applied = applied
        self.clipped = applied != state

    def apply(self, requested: np.ndarray, dt: float) -> np.ndarray:
        request = _vector(requested, "requested", self.size)
        if type(dt) is not float or dt != self._cached_dt:
            dt = _scalar(dt, "dt", 0, np.inf)
            if dt == 0:
                raise ValueError("dt must be positive")
        if not self.enabled:
            result = request.copy()
            self._state = result.copy()
            self.last_prelimit = result.copy()
            self.last_applied = result.copy()
            self.clipped = np.zeros(self.size, dtype=bool)
            return result

        # Most candidates select one or two motors. Cache constant-step coefficients
        # and avoid allocating full-size boolean-index temporaries at every 1 kHz tick.
        coefficients = self._cached_coefficients
        if dt != self._cached_dt:
            coefficients = []
            for index, gain, tau, limit in self._motor_parameters:
                alpha = float(-np.expm1(-dt / tau)) if tau else 1.0
                coefficients.append((index, gain, alpha, 1 - alpha, limit))
            coefficients = tuple(coefficients)
        if request.dtype.itemsize > 8:
            with np.errstate(over="ignore", invalid="ignore"):
                prelimit = request.astype(np.float64, copy=True)
            if not np.isfinite(prelimit).all():
                raise ValueError("requested exceeds finite float64 range")
        else:
            prelimit = request.astype(np.float64, copy=True)
        for index, gain, alpha, retained, _ in coefficients:
            target = float(prelimit[index]) * gain
            effort = retained * float(self._state[index]) + alpha * target
            if not math.isfinite(effort):
                raise ValueError("motor response calculation exceeds finite float64 range")
            prelimit[index] = effort
        applied = prelimit.copy()
        for index, _, _, _, limit in coefficients:
            effort = float(prelimit[index])
            if effort > limit:
                applied[index] = limit
            elif effort < -limit:
                applied[index] = -limit
        # Commit only after every selected motor has passed validation.
        self._cached_dt = dt
        self._cached_coefficients = coefficients
        self._state = prelimit.copy()
        self.last_prelimit = prelimit
        self.last_applied = applied.copy()
        self.clipped = applied != prelimit
        return applied
