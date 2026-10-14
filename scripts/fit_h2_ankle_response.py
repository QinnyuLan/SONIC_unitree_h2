#!/usr/bin/env python3
"""Fit dynamic friction on a one-DOF MuJoCo surrogate, validate full H2 separately."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from gear_sonic.utils.mujoco_sim.ankle_response import BristleFriction, BristleParameters


def parameters(x):
    fs, ratio, vs, stiffness, slope = x
    return BristleParameters(fs, fs*ratio, vs, stiffness, slope)


def load_trials(source, keys, cycles=3):
    result = []
    metadata = json.loads((source/"metadata.json").read_text())
    index = metadata["active_index"]
    if index not in (5, 11) or metadata.get("status") != "completed":
        raise ValueError("expected a completed left/right pitch experiment")
    for amplitude, frequency in keys:
        path = source / "friction_hard" / f"{amplitude:g}deg_{frequency:g}Hz.npz"
        z = np.load(path)
        c, real = z["commands"], z["real"]
        begin = c[c[:, 1] == 1, 0][0]
        end = min(c[-1, 0], begin+cycles/frequency)
        t = np.arange(0, end, .001)
        ci = np.searchsorted(c[:, 0], t, side="right")-1
        u = c[ci, 3+index]
        kp = c[ci, 3+62+index]
        kd = c[ci, 3+93+index]
        mask = (t >= begin) & (t <= real[-1, 0])
        q = np.interp(t, real[:, 0], real[:, 1])
        dq = np.interp(t, real[:, 0], real[:, 2])
        dq = savgol_filter(dq, 31, 2)
        result.append(dict(amplitude=amplitude, frequency=frequency, t=t, u=u, kp=kp, kd=kd,
                           q=q, dq=dq, mask=mask, begin=begin))
    return result


class Surrogate:
    def __init__(self):
        # Equivalent pitch inertia/damping/dry loss computed from the full
        # official equality-constraint tangent at the ready configuration.
        self.model = mujoco.MjModel.from_xml_string('''<mujoco>
          <option timestep="0.001" integrator="implicitfast" gravity="0 0 0"/>
          <worldbody><body><joint name="pitch" damping="0.2248775585" frictionloss="0.6284347511"/>
          <inertial pos="0 0 0" mass="1" diaginertia="0.03878127194 0.03878127194 0.03878127194"/>
          </body></worldbody></mujoco>''')
        self.data = mujoco.MjData(self.model)

    def run(self, trial, x):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        friction = BristleFriction(parameters(x))
        d.qpos[0] = -.363
        mujoco.mj_forward(m, d)
        output = np.empty((len(trial["t"]), 2))
        for i, (u, kp, kd) in enumerate(zip(trial["u"], trial["kp"], trial["kd"])):
            q, v = float(d.qpos[0]), float(d.qvel[0])
            output[i] = q, v
            d.qfrc_applied[0] = kp*(u-q)-kd*v-friction.force(q, v, .001)
            mujoco.mj_step(m, d)
        return output


def residual(predicted, trial):
    mask = trial["mask"]
    t, u = trial["t"][mask], trial["u"][mask]
    q, v = predicted[mask].T
    real_q, real_v = trial["q"][mask], trial["dq"][mask]
    # Mean offsets depend on unobserved friction preload / fixture geometry.
    # Compare AC trajectories without fitted phase shifts and report DC separately.
    qerr = ((q-q.mean())-(real_q-real_q.mean()))/np.deg2rad(max(trial["amplitude"], 3))
    verr = (v-real_v)/(2*np.pi*trial["frequency"]*np.deg2rad(max(trial["amplitude"], 3)))
    # Equal-sized vectors give trials equal weight irrespective of their period.
    selected = np.linspace(0, len(t)-1, 200).astype(int)
    return np.r_[qerr[selected], .2*verr[selected]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-evaluations", type=int, default=24)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    keys = [(3, .25), (6, .5), (6, 1), (10, .25), (25, .25), (1, 2)]
    trials = load_trials(args.source, keys)
    sim = Surrogate()
    right = json.loads((args.source/"metadata.json").read_text())["active_index"] == 11
    x0 = [1.6 if right else .6, .6, .2, 700., 3.]
    calls = 0
    best = float("inf")
    started = time.monotonic()

    def objective(x):
        nonlocal calls, best
        calls += 1
        r = np.concatenate([residual(sim.run(t, x), t) for t in trials])
        score = float(np.mean(r*r))
        if score < best:
            best = score
            (args.output/"best.json").write_text(json.dumps(dict(parameters=parameters(x).__dict__,
                surrogate_score=score, evaluations=calls, training_trials=keys), indent=2)+"\n")
        if calls % 6 == 1:
            print(f"eval {calls}, score {score:.6f}, best {best:.6f}, elapsed {time.monotonic()-started:.1f}s, parameters={x}", flush=True)
        return r

    result = least_squares(objective, x0, bounds=([.1, .1, .003, 50, 0], [5, 1, 1, 5000, 20]),
                           x_scale="jac", diff_step=.025, max_nfev=args.max_evaluations, ftol=.002, xtol=.002)
    (args.output/"fit.json").write_text(json.dumps(dict(parameters=parameters(result.x).__dict__,
        success=bool(result.success), message=result.message, evaluations=calls, training_trials=keys,
        warning="SURROGATE FIT ONLY: must be validated in full official H2 before deployment"), indent=2)+"\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
