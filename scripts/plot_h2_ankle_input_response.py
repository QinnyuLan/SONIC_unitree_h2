#!/usr/bin/env python3
"""Plot published joint targets versus measured response; never reads encoder input."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "gear_sonic_deploy/logs"
LEFT = LOGS / "ankle_dynamic_left_20260906_210301_532438"
RIGHT = LOGS / "ankle_dynamic_right_20260906_211438_540301"
COLORS = dict(target="#666666", real="#b43931", original="#2376a5", candidate="#18866c")


def columns(path, names):
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        indices = [header.index(n) for n in names]
        return np.array([[float(row[i]) for i in indices] for row in reader])


def deploy(output):
    # DDS 0:29 are actual body joints. CSV local indices 15/16 are synthetic head values.
    local = list(range(15)) + list(range(17, 31))
    records = []
    for path in sorted(LOGS.glob("h2_step20000*")):
        q = columns(path/"q.csv", ["time_monotonic_ms"]+[f"q_{i}" for i in local])
        dq = columns(path/"dq.csv", ["time_monotonic_ms"]+[f"dq_{i}" for i in local])
        tau = columns(path/"motor_torque.csv", ["time_monotonic_ms"]+[f"tau_{i}" for i in range(29)])
        play = columns(path/"motion_playing.csv", ["playing_0"])[:, 0] > 0
        with (path/"motion_name.csv").open() as f:
            names = np.array([r["motion_name"] for r in csv.DictReader(f)])
        c = columns(path/"lowcmd_command.csv", ["time_monotonic_ms", "program_state", "play"] +
                    [f"{prefix}_{i}" for prefix in ("q_target", "dq_target", "kp", "kd", "tau_ff") for i in range(29)])
        assert len(q) == len(dq) == len(tau) == len(play) == len(names)
        np.testing.assert_allclose(q[:, 0], dq[:, 0], atol=.001, rtol=0)
        ci = np.searchsorted(c[:, 0], q[:, 0], side="right")-1
        valid = ci >= 0
        ci = np.maximum(ci, 0)
        command = c[ci, 3:].reshape(-1, 5, 29)
        target, desired_velocity, kp, kd, ff = command.transpose(1, 0, 2)
        predicted_effort = kp*(target-q[:, 1:])+kd*(desired_velocity-dq[:, 1:])+ff
        squat = valid & play & (names == "squat_001__A359") & (c[ci, 1] == 2) & (c[ci, 2] == 1)
        # Identify extending knees from feedback, not from encoder reference.
        knee_speed = np.convolve(dq[:, [4, 10]].mean(axis=1), np.ones(5)/5, mode="same")
        extending = squat & (knee_speed < np.deg2rad(-5))
        for window, mask in (("all_squat_playback", squat), ("measured_knee_extension", extending)):
            for j in range(29):
                e = np.rad2deg(target[mask, j]-q[mask, j+1])
                records.append(dict(log=path.name, window=window, joint_dds=j, q_csv_local=local[j],
                    samples=int(mask.sum()), kp=float(np.median(kp[mask, j])), kd=float(np.median(kd[mask, j])),
                    target_minus_q_rmse_deg=float(np.sqrt(np.mean(e**2))), mean_error_deg=float(e.mean()),
                    p95_abs_error_deg=float(np.percentile(np.abs(e), 95)),
                    target_min_deg=float(np.rad2deg(target[mask, j].min())),
                    target_max_deg=float(np.rad2deg(target[mask, j].max())),
                    actual_min_deg=float(np.rad2deg(q[mask, j+1].min())),
                    actual_max_deg=float(np.rad2deg(q[mask, j+1].max())),
                    max_command_age_ms=float((q[:, 0]-c[ci, 0])[mask].max())))
        if path.name != "h2_step20000_ankle_kp1.2_kd1.0_20260906_181056":
            continue
        indices = np.flatnonzero(squat)
        first = np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1)[0]
        t = (q[first, 0]-q[first[0], 0])/1000
        fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True, layout="constrained")
        for row, (label, joints) in enumerate((("Hip pitch", (0, 6)), ("Knee", (3, 9)), ("Ankle pitch", (5, 11)))):
            for col, j in enumerate(joints):
                ax = axes[row, col]
                ax.plot(t, np.rad2deg(target[first, j]), "--", color=COLORS["target"], label="Published LowCmd q_target")
                ax.plot(t, np.rad2deg(q[first, j+1]), color=COLORS["real"], label="Measured q")
                ax.set(title=f"{'Left' if col == 0 else 'Right'} {label}", ylabel="Angle (deg)", xlim=(0, t[-1]))
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("Time since first squat playback (s)")
        fig.suptitle("Loaded policy: actual motor inputs vs output | ankle Kp=48, Kd=2\nPD error includes load-support effort; it is not pure position lag", fontsize=12)
        fig.savefig(output/"loaded_lowcmd_vs_q.png", dpi=160)
        plt.close(fig)
        rows = []
        for k, idx in enumerate(first):
            row = dict(time_s=t[k])
            for side, j in (("left", 5), ("right", 11)):
                row.update({side+"_"+name: float(value) for name, value in dict(
                    target_deg=np.rad2deg(target[idx, j]), q_deg=np.rad2deg(q[idx, j+1]),
                    dq_deg_s=np.rad2deg(dq[idx, j+1]), pd_requested_nm=predicted_effort[idx, j],
                    tau_est_nm=tau[idx, j+1]).items()})
            rows.append(row)
        write_csv(output/"loaded_ankle_lowcmd_trace.csv", rows)
    write_csv(output/"all_body_joints_lowcmd_response.csv", records)


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def friction(output, folder):
    trials = [(3, .25), (6, 1), (10, .6)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    rows = []
    for row, (side, source, active) in enumerate((("Left", LEFT, 5), ("Right", RIGHT, 11))):
        report = json.loads((source/folder/"report.json").read_text())
        original = json.loads((source/"sim_replay/analysis.json").read_text())
        orig_map = {(r["command_amplitude_deg"], r["frequency_hz"]): r for r in original["frequencies"]}
        for entry in report["trials"]:
            key = entry["amplitude_deg"], entry["frequency_hz"]
            result = dict(side=side, amplitude_deg=key[0], frequency_hz=key[1], held_out=entry["held_out"])
            for name in ("gain", "phase_deg", "peak_to_peak_deg"):
                orig_key = "output_peak_to_peak_deg" if name == "peak_to_peak_deg" else name
                result.update({"real_"+name: entry["real"][name], "candidate_"+name: entry["predicted"][name],
                               "original_"+name: orig_map[key][orig_key]})
            rows.append(result)
        for col, (amplitude, frequency) in enumerate(trials):
            data = np.load(source/folder/f"{amplitude:g}deg_{frequency:g}Hz.npz")
            c, s, pred = data["commands"], data["real"], data["predicted"]
            start = c[c[:, 1] == 1, 0][0]+1/frequency
            end = start+2/frequency
            ax = axes[row, col]
            cmask = (c[:, 0] >= start) & (c[:, 0] <= end)
            ax.plot(c[cmask, 0]-start, np.rad2deg(c[cmask, 3+active]), "--", color=COLORS["target"], label="Published q_target")
            for array, name in ((s, "real"), (pred, "candidate")):
                mask = (array[:, 0] >= start) & (array[:, 0] <= end)
                ax.plot(array[mask, 0]-start, np.rad2deg(array[mask, 1]), color=COLORS[name], label=name.capitalize(), linewidth=1.4)
            ax.set(title=f"{side}: +/-{amplitude:g} deg, {frequency:g} Hz", xlabel="Time (s)", ylabel="Pitch angle (deg)", xlim=(0, end-start))
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Actual LowCmd replay in official H2 closed chain | Kp=40, Kd=2\nFriction hypothesis; one fitted trial per side, these three trials held out", fontsize=12)
    fig.savefig(output/"pitch_friction_heldout_waveforms.png", dpi=170)
    plt.close(fig)
    write_csv(output/"pitch_friction_validation.csv", rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--friction-folder", help="folder name relative to both hardware log directories")
    p.add_argument("--skip-deploy", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    if not a.skip_deploy:
        deploy(a.output)
    if a.friction_folder:
        friction(a.output, a.friction_folder)
    print(a.output)


if __name__ == "__main__":
    main()
