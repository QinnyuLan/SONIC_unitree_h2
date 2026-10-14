#!/usr/bin/env python3
"""Coverage and tracking error for deploy-ready H2 ankle range tests; no DDS."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def analyze(path):
    meta = json.loads((path / "metadata.json").read_text())
    if meta.get("schema") != "h2_ankle_range_v1":
        raise ValueError("expected a range-test log, not the old sine-test log")
    with (path / "commands.csv").open() as f:
        commands = [(int(r["send_begin_monotonic_ns"]), r["phase"],
                     float(r["q_des_5"]), float(r["q_des_11"])) for r in csv.DictReader(f)]
    with (path / "states.csv").open() as f:
        states = [(int(r["receive_monotonic_ns"]), *[float(r[f"{key}_{i}"])
                   for key in ("q", "dq", "tau_est") for i in (5, 11)]) for r in csv.DictReader(f)]
    if not commands or not states:
        raise ValueError("no usable commands/states")
    origin = min(commands[0][0], states[0][0])
    ct = np.array([(r[0]-origin)/1e9 for r in commands])
    st = np.array([(r[0]-origin)/1e9 for r in states])
    values = np.asarray([r[1:] for r in states])
    if np.any(np.diff(ct)<=0) or np.any(np.diff(st)<=0) or not np.isfinite(values).all():
        raise ValueError("non-monotonic time or non-finite state")
    ci = np.searchsorted(ct, st, side="right")-1
    ci = np.maximum(0, ci)
    reports = []
    completed = {(s["side"], s["level"]) for s in meta["completed_stages"]}
    for stage in meta["stages"]:
        prefix = f"range_{stage['side']}_{stage['level']}_"
        mask = np.array([commands[i][1].startswith(prefix) for i in ci]) & (st>=ct[0]) & (st<=ct[-1])
        if not mask.any():
            continue
        side = int(stage["side"] == "right")
        indices = ci[mask]
        target = np.array([commands[i][2+side] for i in indices])
        q = values[mask, side]
        err = np.rad2deg(q-target)
        report = dict(side=stage["side"], level=stage["level"],
            completed=(stage["side"], stage["level"]) in completed,
            requested_lower_deg=float(np.rad2deg(stage["lower"])), requested_upper_deg=float(np.rad2deg(stage["upper"])),
            published_lower_deg=float(np.rad2deg(target.min())), published_upper_deg=float(np.rad2deg(target.max())),
            actual_lower_deg=float(np.rad2deg(q.min())), actual_upper_deg=float(np.rad2deg(q.max())),
            tracking_rmse_deg=float(np.sqrt(np.mean(err**2))), max_abs_error_deg=float(np.max(np.abs(err))),
            max_abs_speed_deg_s=float(np.max(np.abs(np.rad2deg(values[mask, 2+side])))),
            max_abs_tau_est=float(np.max(np.abs(values[mask, 4+side]))))
        for direction in ("down", "up"):
            hm = np.array([commands[i][1] == prefix+direction+"_hold" for i in indices])
            report[direction+"_hold_mean_error_deg"] = float(err[hm].mean()) if hm.any() else None
        reports.append(report)
    if not reports:
        raise ValueError("no range scan started; inspect preparation/stop events")
    result = dict(status=meta["status"], source_status=meta.get("source_status"), stages=reports,
                  note="Requested range is not demonstrated coverage; compare actual extrema and errors. Not a frequency-response fit.")
    with (path / "range_analysis.json").open("w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    with (path / "range_response.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("logs", nargs="+", type=Path)
    for path in p.parse_args().logs:
        result = analyze(path)
        print(f"{path}: {result['status']}")
        for r in result["stages"]:
            print(f"{r['side']} #{r['level']} complete={r['completed']} "
                  f"target [{r['published_lower_deg']:.2f},{r['published_upper_deg']:.2f}] deg "
                  f"actual [{r['actual_lower_deg']:.2f},{r['actual_upper_deg']:.2f}] deg "
                  f"RMSE={r['tracking_rmse_deg']:.3f} deg max_error={r['max_abs_error_deg']:.3f} deg")
