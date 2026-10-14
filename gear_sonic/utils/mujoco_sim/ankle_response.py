"""Opt-in ankle friction hypotheses; all forces evolve from simulated state.

The dynamic model uses the LuGre bristle-state equation with sigma_1=0:
  dz/dt = v - abs(v)*sigma_0*z / F(v)
  F(v) = Fc + (Fs-Fc)*exp(-(v/vs)^2)
The bristle state is integrated exactly for frozen velocity during a step.
This is an effective, unloaded-response model, not a measured hardware model.
"""

from dataclasses import dataclass
import json
import math

import numpy as np

MODEL_TEXT_KEY = "h2_ankle_pitch_response_v1"


def model_response_config(model, config):
    """An explicit empty override disables embedded parameters for A/B tests."""
    if "ANKLE_PITCH_RESPONSE" in config:
        return config["ANKLE_PITCH_RESPONSE"]
    import mujoco
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, MODEL_TEXT_KEY)
    if index < 0:
        return {}
    start = int(model.text_adr[index])
    size = int(model.text_size[index])
    return json.loads(model.text_data[start:start+size].rstrip(b"\0").decode("ascii"))


@dataclass(frozen=True)
class BristleParameters:
    static_nm: float
    dynamic_nm: float
    velocity_rad_s: float
    stiffness_nm_rad: float
    position_slope_nm_rad2: float = 0.0
    center_rad: float = -.363

    def __post_init__(self):
        values = tuple(self.__dict__.values())
        if not all(math.isfinite(x) for x in values):
            raise ValueError("friction parameters must be finite")
        if not 0 < self.dynamic_nm <= self.static_nm <= 10:
            raise ValueError("require 0 < dynamic_nm <= static_nm <= 10")
        if not .001 <= self.velocity_rad_s <= 5:
            raise ValueError("velocity_rad_s outside [.001, 5]")
        if not 20 <= self.stiffness_nm_rad <= 10000:
            raise ValueError("stiffness_nm_rad outside [20, 10000]")
        if not 0 <= self.position_slope_nm_rad2 <= 30:
            raise ValueError("position friction slope outside [0, 30]")


class BristleFriction:
    def __init__(self, parameters):
        self.parameters = parameters
        self.z = 0.0

    def reset(self, torque=0.0):
        p = self.parameters
        self.z = float(np.clip(torque, -p.static_nm, p.static_nm))/p.stiffness_nm_rad

    def force(self, q, velocity, dt):
        p = self.parameters
        limit = p.dynamic_nm + (p.static_nm-p.dynamic_nm)*math.exp(-(velocity/p.velocity_rad_s)**2)
        limit += p.position_slope_nm_rad2*(q-p.center_rad)**2
        rate = abs(velocity)*p.stiffness_nm_rad/limit
        if rate > 1e-12:
            target = math.copysign(limit/p.stiffness_nm_rad, velocity)
            self.z += (target-self.z)*(-math.expm1(-rate*dt))
        return p.stiffness_nm_rad*self.z


class AnkleResponse:
    """Only configured output-pitch coordinates receive the extra force."""

    def __init__(self, model, config):
        if not isinstance(config, dict) or set(config)-{"left", "right"}:
            raise ValueError("ANKLE_PITCH_RESPONSE expects left/right dictionaries")
        self.axes = []
        for side, parameters in config.items():
            p = BristleParameters(**parameters)
            joint = model.joint(side+"_ankle_pitch_joint")
            self.axes.append((int(joint.qposadr[0]), int(joint.dofadr[0]), BristleFriction(p)))
        self.last_time = None
        self.last_force = np.zeros(model.nv)

    def reset(self):
        for _, _, friction in self.axes:
            friction.reset()
        self.last_time = None
        self.last_force.fill(0)

    def apply(self, model, data):
        # Calls at the same physics time must not advance or accumulate twice.
        if self.last_time is not None and data.time < self.last_time:
            self.reset()
        if self.last_time == data.time:
            return
        self.last_force.fill(0)
        for qpos, dof, friction in self.axes:
            self.last_force[dof] = -friction.force(data.qpos[qpos], data.qvel[dof], model.opt.timestep)
        # These pitch DOFs are owned by this opt-in model; do not overwrite other forces.
        for _, dof, _ in self.axes:
            data.qfrc_applied[dof] = self.last_force[dof]
        self.last_time = float(data.time)
