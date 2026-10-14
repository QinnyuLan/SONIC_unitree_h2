#!/usr/bin/env python3
"""Validate fitted friction in the full official H2 loop fixture, without DDS."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from analyze_h2_ankle_friction_hypothesis import FixtureEnv, make_config, simulate, metrics, ROOT, CONFIG


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--fit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trials", default="all")
    p.add_argument("--model", type=Path, help="validate the delivered MJCF with both embedded ankle models")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    fit = json.loads(a.fit.read_text())
    meta = json.loads((a.source/"metadata.json").read_text())
    side, active = meta["args"]["side"], meta["active_index"]
    config = make_config(73, a.source/"metadata.json")
    if a.model:
        from view_h2_ankle_sysid import ViewFixture
        from gear_sonic.utils.mujoco_sim.ankle_response import model_response_config
        config["ROBOT_SCENE"] = str(a.model.resolve())
        env = ViewFixture(config, onscreen=False)
        embedded = model_response_config(env.mj_model, config)
        if embedded[side] != fit["parameters"]:
            raise ValueError("delivered model parameters differ from the selected fit")
    else:
        config["ANKLE_PITCH_RESPONSE"] = {side: fit["parameters"]}
        env = FixtureEnv(config, onscreen=False)
    initial = env.mj_data.qpos.copy()
    keys = [(s["amplitude_deg"], s["frequency_hz"]) for s in meta["stages"]]
    if a.trials != "all":
        keys = [tuple(map(float, x.split(":"))) for x in a.trials.split(",")]
    report = dict(status="running", source=str(a.source.resolve()), side=side,
                  parameters=fit["parameters"], training_trials=fit["training_trials"],
                  model_sha256=hashlib.sha256((ROOT/config["ROBOT_SCENE"]).read_bytes()).hexdigest(),
                  delivered_model=bool(a.model),
                  input="original published commands; no fitted time shifts; no measured output injected",
                  trials=[])
    start = time.monotonic()
    for amplitude, frequency in keys:
        stem = f"{amplitude:g}deg_{frequency:g}Hz"
        source = np.load(a.source/"friction_hard"/(stem+".npz"))
        spec = dict(amplitude_deg=amplitude, frequency_hz=frequency)
        trial = dict(spec=spec, commands=source["commands"], states=source["real"])
        prediction = simulate(env, initial, trial, active)
        predicted = metrics(trial, prediction, active)
        real = metrics(trial, trial["states"], active)
        mask = prediction[:, 0] >= trial["commands"][trial["commands"][:, 1] == 1, 0][0]
        times = prediction[mask, 0]
        real_q = np.interp(times, trial["states"][:, 0], trial["states"][:, 1])
        q = prediction[mask, 1]
        q_ac_error = (q-q.mean())-(real_q-real_q.mean())
        row = dict(**spec, held_out=[amplitude, frequency] not in fit["training_trials"],
                   real=real, predicted=predicted, q_rmse_deg=float(np.rad2deg(np.sqrt(np.mean((q-real_q)**2)))),
                   q_ac_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(q_ac_error**2)))))
        report["trials"].append(row)
        np.savez_compressed(a.output/(stem+".npz"), commands=trial["commands"], real=trial["states"], predicted=prediction)
        (a.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(f"{side} {stem}: gain {real['gain']:.3f}/{predicted['gain']:.3f}, phase {real['phase_deg']:.1f}/{predicted['phase_deg']:.1f}, AC RMSE {row['q_ac_rmse_deg']:.3f} deg; {time.monotonic()-start:.1f}s", flush=True)
    report["status"] = "completed"
    (a.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
