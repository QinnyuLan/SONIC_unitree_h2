#!/usr/bin/env python3
"""Summarize full-model ankle validation, with unshifted angle/velocity plots."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def key(row):
    return row.get("amplitude_deg", row.get("command_amplitude_deg")), row["frequency_hz"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left", type=Path, required=True, help="full validation directory")
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rows, summary, reports = [], {}, {}
    for side, directory in (("left", a.left), ("right", a.right)):
        report = json.loads((directory/"report.json").read_text())
        if report["status"] != "completed" or len(report["trials"]) != 18:
            raise ValueError("expected all 18 completed trials per side")
        source = Path(report["source"])
        baseline = {key(r): r for r in json.loads((source/"sim_replay/analysis.json").read_text())["frequencies"]}
        old = {key(r): r for r in json.loads((source/"friction_hard/report.json").read_text())["trials"]}
        reports[side] = directory, source, report
        for r in report["trials"]:
            real, sim = r["real"], r["predicted"]
            previous, base = old[key(r)]["predicted"], baseline[key(r)]
            rows.append(dict(side=side, amplitude_deg=key(r)[0], frequency_hz=key(r)[1],
                fit_trial=not r["held_out"], real_gain=real["gain"], sim_gain=sim["gain"],
                old_friction_gain=previous["gain"], baseline_gain=base["gain"],
                real_phase_deg=real["phase_deg"], sim_phase_deg=sim["phase_deg"],
                old_friction_phase_deg=previous["phase_deg"], baseline_phase_deg=base["phase_deg"],
                q_rmse_deg=r["q_rmse_deg"], q_ac_rmse_deg=r["q_ac_rmse_deg"],
                mean_error_deg=sim["mean_deg"]-real["mean_deg"],
                real_peak_speed_deg_s=real["peak_speed_deg_s"], sim_peak_speed_deg_s=sim["peak_speed_deg_s"],
                real_low_speed_fraction=real["low_speed_fraction"], sim_low_speed_fraction=sim["low_speed_fraction"]))
        subset = [r for r in rows if r["side"] == side]
        summary[side] = dict(parameters=report["parameters"], validation=str(directory.resolve()))
        for name, chosen in (("all", subset), ("not_directly_fitted", [r for r in subset if not r["fit_trial"]])):
            gain_error = lambda field: float(np.mean([abs(r[field]-r["real_gain"]) for r in chosen]))
            strong = [r for r in chosen if r["real_gain"] > .2]
            summary[side][name] = dict(trials=len(chosen), gain_mae=gain_error("sim_gain"),
                baseline_gain_mae=gain_error("baseline_gain"), old_friction_gain_mae=gain_error("old_friction_gain"),
                strong_response_phase_mae_deg=float(np.mean([abs(r["sim_phase_deg"]-r["real_phase_deg"]) for r in strong])),
                strong_response_old_phase_mae_deg=float(np.mean([abs(r["old_friction_phase_deg"]-r["real_phase_deg"]) for r in strong])),
                median_q_ac_rmse_deg=float(np.median([r["q_ac_rmse_deg"] for r in chosen])),
                max_q_ac_rmse_deg=float(max(r["q_ac_rmse_deg"] for r in chosen)),
                max_abs_mean_error_deg=float(max(abs(r["mean_error_deg"]) for r in chosen)))
    with (a.output/"validation.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (a.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")

    # No time shifts, DC alignment or output injection: show raw LowCmd and q.
    selected = [(3, .25), (6, 1), (10, .6)]
    colors = dict(command="#858585", real="#111111", previous="#d77a21", fitted="#197d9b")
    for amplitude, frequency in selected:
        fig, axes = plt.subplots(2, 2, figsize=(12, 6.6), sharex=True, layout="constrained")
        for col, side in enumerate(("left", "right")):
            directory, source, report = reports[side]
            stem = f"{amplitude:g}deg_{frequency:g}Hz.npz"
            cache = np.load(directory/stem)
            previous = np.load(source/"friction_hard"/stem)["predicted"]
            commands, real, fitted = cache["commands"], cache["real"], cache["predicted"]
            begin = commands[commands[:, 1] == 1, 0][0]+1/frequency
            end = begin+2/frequency
            active = 5 if side == "left" else 11
            c = commands[(commands[:, 0] >= begin) & (commands[:, 0] < end)]
            axes[0, col].plot(c[:, 0]-begin, np.rad2deg(c[:, 3+active]), color=colors["command"],
                              linewidth=1, linestyle="--", label="Published q_des")
            for name, data, color in (("Measured", real, colors["real"]), ("Previous dry friction", previous, colors["previous"]),
                                     ("New dynamic friction", fitted, colors["fitted"])):
                d = data[(data[:, 0] >= begin) & (data[:, 0] < end)][::2]
                axes[0, col].plot(d[:, 0]-begin, np.rad2deg(d[:, 1]), color=color, linewidth=1.4, label=name)
                axes[1, col].plot(d[:, 0]-begin, np.rad2deg(d[:, 2]), color=color, linewidth=1.1)
            axes[0, col].set_title(side.capitalize()+" ankle pitch")
            axes[0, col].set_ylabel("Angle [deg]")
            axes[1, col].set_ylabel("Velocity [deg/s]")
            axes[1, col].set_xlabel("Time within measurement window [s]")
            for ax in axes[:, col]:
                ax.grid(alpha=.18)
                ax.spines[["top", "right"]].set_visible(False)
        label = "directly fitted condition" if [amplitude, frequency] in reports["left"][2]["training_trials"] else "not directly fitted"
        fig.suptitle(f"+/-{amplitude:g} deg at {frequency:g} Hz | {label} | Kp=40, Kd=2")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
        fig.savefig(a.output/f"response_{amplitude:g}deg_{frequency:g}Hz.png", dpi=160)
        plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
