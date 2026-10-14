#!/usr/bin/env python3
"""Offline test of extra pitch friction in the official H2 closed-chain model.

Uses actual LowCmd timelines, not encoder references. Fits one friction increment
on one trial; other trials are held out. No DDS or real robot connection.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from analyze_h2_ankle_sysid import fit_sine, trial_specs
from sim_h2_ankle_sysid import CONFIG, ROOT, FixtureEnv, make_config, apply_pitch_response
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import H2_JOINT_NAMES_DDS


def read_columns(path, columns, text=()):
    with path.open() as stream:
        reader = csv.reader(stream)
        header = next(reader)
        indices = [header.index(c) for c in columns]
        text_indices = [header.index(c) for c in text]
        rows, labels = [], []
        for row in reader:
            rows.append([float(row[i]) for i in indices])
            if text_indices:
                labels.append([row[i] for i in text_indices])
    data = np.asarray(rows)
    if not np.isfinite(data).all():
        raise ValueError(f"non-finite data in {path}")
    return data, np.asarray(labels)


def load_trials(source):
    meta = json.loads((source / "metadata.json").read_text())
    if meta.get("status") != "completed":
        raise ValueError("expected completed hardware trial")
    index = meta["active_index"]
    columns = ["send_begin_monotonic_ns", "measure", "frequency_hz"]
    columns += [f"{prefix}_{j}" for prefix in ("q_des", "dq_des", "kp", "kd", "tau_ff") for j in range(31)]
    commands, labels = read_columns(source / "commands.csv", columns, ["phase"])
    states, _ = read_columns(source / "states.csv", ["receive_monotonic_ns", f"q_{index}", f"dq_{index}"])
    origin = commands[0, 0]
    commands[:, 0] = (commands[:, 0]-origin)*1e-9
    states[:, 0] = (states[:, 0]-origin)*1e-9
    if np.any(np.diff(commands[:, 0]) <= 0) or np.any(np.diff(states[:, 0]) <= 0):
        raise ValueError("non-monotonic samples")
    trials = []
    for spec in trial_specs(meta):
        mask = (labels[:, 0] == spec["excitation_phase"]) & (commands[:, 2] == spec["frequency_hz"])
        c = commands[mask].copy()
        measured = c[:, 1] == 1
        start, end = c[0, 0], c[measured, 0][-1]
        c = c[c[:, 0] <= end]
        s = states[(states[:, 0] >= start) & (states[:, 0] <= end)].copy()
        c[:, 0] -= start
        s[:, 0] -= start
        trials.append(dict(spec=spec, commands=c, states=s))
    return meta, trials


def simulate(env, initial, trial, active, cycles=None):
    """Same PD/closed-chain torque mapping as DefaultEnv, without logging/render cost."""
    m, d = env.mj_model, env.mj_data
    mujoco.mj_resetData(m, d)
    if env.ankle_response is not None:
        env.ankle_response.reset()
    d.qpos[:] = initial
    mujoco.mj_forward(m, d)
    env._closed_loop_jacobian_cache_time = None
    order = np.array([H2_JOINT_NAMES_DDS.index(n) for n in env.body_joint_names])
    c = trial["commands"]
    targets = c[:, 3:].reshape(-1, 5, 31)[:, :, order]
    active_local = int(np.flatnonzero(order == active)[0])
    qidx, vidx = env.body_qpos_index, env.body_dof_index
    measured_begin = c[c[:, 1] == 1, 0][0]
    end = c[-1, 0] if cycles is None else min(c[-1, 0], measured_begin + cycles/trial["spec"]["frequency_hz"])
    times = np.arange(0, end, env.sim_dt)
    ci = np.searchsorted(c[:, 0], times, side="right")-1
    result = np.empty((len(times), 3))
    for k, command_index in enumerate(ci):
        qdes, dqdes, kp, kd, ff = targets[command_index]
        result[k] = times[k], d.qpos[qidx[active_local]], d.qvel[vidx[active_local]]
        tau = ff + kp*(qdes-d.qpos[qidx]) + kd*(dqdes-d.qvel[vidx])
        env.torques.fill(0)
        env.torques[env.body_actuator_index] = env._map_body_torques_to_actuators(tau)
        torque = np.clip(env.torques, -env.torque_limit, env.torque_limit)
        d.ctrl[:] = np.concatenate((np.zeros(6), torque)) if env.config["FREE_BASE"] else torque
        if env.ankle_response is not None:
            env.ankle_response.apply(m, d)
        mujoco.mj_step(m, d)
    if not np.isfinite(result).all():
        raise ValueError("non-finite simulation")
    return result


def metrics(trial, states, active):
    c = trial["commands"]
    begin = c[c[:, 1] == 1, 0][0]
    selected = states[:, 0] >= begin
    t, q, dq = states[selected].T
    ci = np.searchsorted(c[:, 0], t, side="right")-1
    u = c[ci, 3+active]
    f = trial["spec"]["frequency_hz"]
    uf, _ = fit_sine(t-begin, u, f)
    yf, residual = fit_sine(t-begin, q, f)
    h = yf/uf
    return dict(gain=float(abs(h)), phase_deg=float(np.rad2deg(np.angle(h))),
                peak_to_peak_deg=float(np.rad2deg(np.ptp(q))),
                mean_deg=float(np.rad2deg(q.mean())),
                peak_speed_deg_s=float(np.rad2deg(np.abs(dq).max())),
                low_speed_fraction=float(np.mean(np.abs(dq) < np.deg2rad(.2))),
                residual_deg=float(np.rad2deg(residual)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fit-trial", default="6:0.25", help="amplitude_deg:frequency_hz")
    p.add_argument("--grid", default="0,0.5,1,1.5,2,2.5", help="added output-joint frictionloss, Nm")
    p.add_argument("--trials", default="all", help="all, or comma-separated amplitude:frequency pairs")
    p.add_argument("--hard-friction", action="store_true", help="reduce pitch friction constraint creep, opt-in")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    meta, trials = load_trials(a.source)
    side, active = meta["args"]["side"], meta["active_index"]
    key = lambda t: (t["spec"]["amplitude_deg"], t["spec"]["frequency_hz"])
    fit_key = tuple(map(float, a.fit_trial.split(":")))
    fit_trial = next(t for t in trials if key(t) == fit_key)
    grid = [float(v) for v in a.grid.split(",")]
    if not grid or not all(np.isfinite(v) and 0 <= v <= 10 for v in grid):
        p.error("grid values must be in [0,10] Nm")
    env = FixtureEnv(make_config(73, a.source / "metadata.json"), onscreen=False)
    initial = env.mj_data.qpos.copy()
    passive = env.mj_model.dof_frictionloss.copy()
    baseline_solref = env.mj_model.dof_solref.copy()
    baseline_solimp = env.mj_model.dof_solimp.copy()
    fit_real = metrics(fit_trial, fit_trial["states"], active)
    candidates = []
    start = time.monotonic()
    for value in grid:
        env.mj_model.dof_frictionloss[:] = passive
        env.mj_model.dof_solref[:] = baseline_solref
        env.mj_model.dof_solimp[:] = baseline_solimp
        apply_pitch_response(env, {side: dict(frictionloss_add_nm=value, hard_friction=a.hard_friction)})
        prediction = simulate(env, initial, fit_trial, active, cycles=3)
        m = metrics(fit_trial, prediction, active)
        score = abs(m["gain"]-fit_real["gain"])
        candidates.append(dict(increment_nm=value, gain_error=score, **m))
        print(f"{side} fit {fit_key}: added friction={value:g}, gain={m['gain']:.4f}, real={fit_real['gain']:.4f}; {time.monotonic()-start:.1f}s", flush=True)
    best = min(candidates, key=lambda c: c["gain_error"])["increment_nm"]
    report = dict(status="running", source=str(a.source.resolve()), config=str(CONFIG),
                  asset_sha256=hashlib.sha256((ROOT/env.config["ROBOT_SCENE"]).read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  input="actual published q/dq/kp/kd/tau_ff; causal ZOH; no encoder reference",
                  fixture="official closed chain, pelvis fixed, no contacts, other joints PD-held",
                  caveat="Effective friction hypothesis only; unloaded, not identified physical friction or loaded gait validation.",
                  fit_trial=fit_key, fit_objective="fundamental gain error, first 3 steady cycles in simulation",
                  candidates=candidates, chosen_increment_nm=best, hard_friction=a.hard_friction, side=side, trials=[])
    if a.trials != "all":
        selected = {tuple(map(float, s.split(":"))) for s in a.trials.split(",")}
        trials = [t for t in trials if key(t) in selected]
    env.mj_model.dof_frictionloss[:] = passive
    env.mj_model.dof_solref[:] = baseline_solref
    env.mj_model.dof_solimp[:] = baseline_solimp
    report["overrides"] = apply_pitch_response(env, {side: dict(frictionloss_add_nm=best, hard_friction=a.hard_friction)})
    for i, trial in enumerate(trials):
        prediction = simulate(env, initial, trial, active)
        actual = metrics(trial, trial["states"], active)
        fitted = metrics(trial, prediction, active)
        name = f"{key(trial)[0]:g}deg_{key(trial)[1]:g}Hz"
        np.savez_compressed(a.output / (name+".npz"), commands=trial["commands"],
                            real=trial["states"], predicted=prediction)
        report["trials"].append(dict(**trial["spec"], held_out=key(trial) != fit_key, real=actual, predicted=fitted))
        (a.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(f"{side} {i+1}/{len(trials)} {name}: gain real={actual['gain']:.4f}, sim={fitted['gain']:.4f}; {time.monotonic()-start:.1f}s", flush=True)
    report["status"] = "completed"
    (a.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"Saved {a.output}; NO DDS connection", flush=True)


if __name__ == "__main__":
    main()
