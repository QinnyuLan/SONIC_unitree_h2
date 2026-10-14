#!/usr/bin/env python3
"""Refine bristle parameters inside the actual full closed-chain H2 fixture."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

from analyze_h2_ankle_friction_hypothesis import FixtureEnv, make_config, simulate
from fit_h2_ankle_response import load_trials, parameters
from gear_sonic.utils.mujoco_sim.ankle_response import AnkleResponse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--initial", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-evaluations", type=int, default=6)
    p.add_argument("--full-shape", action="store_true", help="also refine Stribeck speed and angle dependence")
    p.add_argument("--initial-speed", type=float)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    fit = json.loads(a.initial.read_text())
    params = fit["parameters"]
    metadata = json.loads((a.source/"metadata.json").read_text())
    side, active = metadata["args"]["side"], metadata["active_index"]
    keys = [(3, .25), (6, 1), (10, .25)]
    trials = load_trials(a.source, keys, cycles=2)
    for trial, (amplitude, frequency) in zip(trials, keys):
        data = np.load(a.source/"friction_hard"/f"{amplitude:g}deg_{frequency:g}Hz.npz")
        trial["full"] = dict(commands=data["commands"], states=data["real"],
                              spec=dict(amplitude_deg=amplitude, frequency_hz=frequency))
    config = make_config(73, a.source/"metadata.json")
    env = FixtureEnv(config, onscreen=False)
    initial = env.mj_data.qpos.copy()
    calls, best, start = 0, float("inf"), time.monotonic()
    x0 = [params["static_nm"], params["dynamic_nm"]/params["static_nm"], params["stiffness_nm_rad"]]
    bounds = ([.1, .1, 50], [5, 1, 5000])
    if a.full_shape:
        x0 = [params["static_nm"], params["dynamic_nm"]/params["static_nm"],
              a.initial_speed or params["velocity_rad_s"], params["stiffness_nm_rad"],
              params["position_slope_nm_rad2"]]
        bounds = ([.1, .05, .005, 50, 0], [5, 1, .8, 5000, 25])

    def objective(x):
        nonlocal calls, best
        calls += 1
        full = x if a.full_shape else [x[0], x[1], params["velocity_rad_s"], x[2], params["position_slope_nm_rad2"]]
        candidate = parameters(full).__dict__
        vectors = []
        for trial in trials:
            env.ankle_response = AnkleResponse(env.mj_model, {side: candidate})
            prediction = simulate(env, initial, trial["full"], active, cycles=2)
            mask = trial["mask"]
            t = trial["t"][mask]
            actual = trial["q"][mask]
            q = np.interp(t, prediction[:, 0], prediction[:, 1])
            actual_v = trial["dq"][mask]
            v = np.interp(t, prediction[:, 0], prediction[:, 2])
            scale = max(np.ptp(actual)/2, np.deg2rad(.75 if a.full_shape else .35))
            qerr = ((q-q.mean())-(actual-actual.mean()))/scale
            verr = (v-actual_v)/(2*np.pi*trial["frequency"]*scale)
            selected = np.linspace(0, len(t)-1, 200).astype(int)
            vectors.append(np.r_[qerr[selected], .12*verr[selected]])
        r = np.concatenate(vectors)
        score = float(np.mean(r*r))
        if score < best:
            best = score
            (a.output/"best.json").write_text(json.dumps(dict(parameters=candidate, score=score,
                training_trials=fit["training_trials"], refined_trials=keys, evaluations=calls), indent=2)+"\n")
        print(f"eval {calls}: score={score:.6f}, best={best:.6f}, params={x}, elapsed={time.monotonic()-start:.1f}s", flush=True)
        return r

    result = least_squares(objective, x0, bounds=bounds,
                           diff_step=.05, x_scale="jac", max_nfev=a.max_evaluations, ftol=.004, xtol=.004)
    # Persist the best evaluated candidate, including finite-difference points.
    best_record = json.loads((a.output/"best.json").read_text())
    best_record.update(success=bool(result.success), message=result.message, total_evaluations=calls)
    (a.output/"fit.json").write_text(json.dumps(best_record, indent=2)+"\n")


if __name__ == "__main__":
    main()
