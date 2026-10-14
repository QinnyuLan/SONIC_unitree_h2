#!/usr/bin/env python3
"""Fit local ankle frequency responses from raw H2 sysid logs (no DDS access)."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def fit_sine(t, y, frequency):
    design = np.column_stack((np.sin(2*np.pi*frequency*t), np.cos(2*np.pi*frequency*t),
                              np.ones_like(t), t-t.mean()))
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != 4:
        raise ValueError("insufficient independent samples for sine fitting")
    residual = y-design@coefficients
    return complex(coefficients[0], coefficients[1]), float(np.sqrt(np.mean(residual**2)))


def dynamic_quality(t, reference, output, frequency, response):
    # Jointly fit harmonics to separate repeatable distortion from the fundamental.
    columns = [fn(2*np.pi*k*frequency*t) for k in (1, 2, 3) for fn in (np.sin, np.cos)]
    design = np.column_stack((*columns, np.ones_like(t), t-t.mean()))
    coefficients, _, rank, _ = np.linalg.lstsq(design, output, rcond=None)
    amplitudes = np.hypot(coefficients[:6:2], coefficients[1:6:2])
    thd = float(np.linalg.norm(amplitudes[1:])/amplitudes[0]) if rank == 8 and amplitudes[0]>1e-8 else None
    cycle_responses = []
    for cycle in range(math.ceil((t[-1]-t[0])*frequency)):
        selected = (t>=cycle/frequency) & (t<(cycle+1)/frequency)
        times = t[selected]
        if len(times)<20 or (times[-1]-times[0])*frequency<.9:
            continue
        u, _ = fit_sine(times, reference[selected], frequency)
        y, _ = fit_sine(times, output[selected], frequency)
        if abs(u)>1e-6:
            cycle_responses.append(y/u)
    hs = np.asarray(cycle_responses)
    gain_cv = phase_std = None
    if len(hs)>=3 and abs(response)>1e-8 and np.abs(hs).mean()>1e-8:
        gain_cv = float(np.std(np.abs(hs), ddof=1)/np.abs(hs).mean())
        phase_std = float(np.rad2deg(np.std(np.angle(hs/response), ddof=1)))
    return dict(output_harmonic_ratio_2_3=thd, cycle_gain_cv=gain_cv,
                cycle_phase_std_deg=phase_std, fitted_cycles=len(hs))


def trial_specs(meta):
    if meta["schema"] == "h2_ankle_dynamic_v2":
        specs = [dict(amplitude_deg=s["amplitude_deg"], frequency_hz=s["frequency_hz"],
                      excitation_phase=s["excitation_phase"]) for s in meta["stages"]]
        if len({s["excitation_phase"] for s in specs}) != len(specs):
            raise ValueError("multi-amplitude trials must have distinct command phases")
        return specs
    return [dict(amplitude_deg=meta["args"]["amplitude_deg"], frequency_hz=f, excitation_phase="excitation")
            for f in meta["args"]["frequencies"]]


def amplitude_comparisons(rows):
    result = []
    for frequency in sorted({r["frequency_hz"] for r in rows}):
        group = sorted((r for r in rows if r["frequency_hz"] == frequency), key=lambda r: r["command_amplitude_deg"])
        if len(group)<2:
            continue
        result.append(dict(frequency_hz=frequency,
            amplitudes_deg=[r["command_amplitude_deg"] for r in group],
            gains=[r["gain"] for r in group], phases_deg=[r["phase_deg"] for r in group],
            output_harmonic_ratios_2_3=[r["output_harmonic_ratio_2_3"] for r in group],
            quality_flags=[r["quality_flags"] for r in group]))
    return result


def analyze(path):
    meta = json.loads((path / "metadata.json").read_text())
    if meta.get("schema") not in {"h2_ankle_sysid_v1", "h2_ankle_dynamic_v1", "h2_ankle_dynamic_v2"}:
        raise ValueError(f"unsupported log schema: {path}")
    index = meta["active_index"]
    # Read just the needed columns; joint state is NEVER assumed to be policy ordered.
    with (path / "commands.csv").open() as f:
        commands = [(int(r["send_begin_monotonic_ns"]), r["phase"], float(r["frequency_hz"]),
                     bool(int(r["measure"])), float(r[f"q_des_{index}"]), float(r["segment_time_s"]),
                     float(r[f"kp_{index}"]), float(r[f"kd_{index}"]))
                    for r in csv.DictReader(f)]
    with (path / "states.csv").open() as f:
        states = [(int(r["receive_monotonic_ns"]), float(r[f"q_{index}"]),
                   float(r[f"q_{index-1}"]), float(r[f"tau_est_{index}"]), int(r["tick"]), float(r[f"dq_{index}"]))
                  for r in csv.DictReader(f)]
    if len(commands) < 20 or len(states) < 20:
        raise ValueError("not enough command/state data; monitor-only logs cannot identify a response")
    origin = min(commands[0][0], states[0][0])
    ct = np.array([(r[0]-origin)/1e9 for r in commands])
    st = np.array([(r[0]-origin)/1e9 for r in states])
    if np.any(np.diff(ct) <= 0) or np.any(np.diff(st) <= 0):
        raise ValueError("timestamps are not strictly increasing")
    q = np.array([r[1] for r in states])
    roll = np.array([r[2] for r in states])
    tau = np.array([r[3] for r in states])
    dq = np.array([r[5] for r in states])
    if not all(np.isfinite(v).all() for v in (q, roll, tau, dq)):
        raise ValueError("non-finite state data")
    rows, unmeasured = [], []
    for spec in trial_specs(meta):
        frequency = spec["frequency_hz"]
        selected = np.array([r[1] == spec["excitation_phase"] and r[2] == frequency and r[3] for r in commands])
        indices = np.flatnonzero(selected)
        if len(indices) < 20:
            unmeasured.append(dict(**spec, reason="no constant-amplitude command window"))
            continue
        begin, end = ct[indices[0]], ct[indices[-1]]
        duration = end-begin
        complete_window = duration >= (meta["args"]["cycles"]-0.1)/frequency
        # Causal zero-order hold of the ACTUAL published target at each receive time.
        mask = (st >= begin) & (st <= end)
        times = st[mask]
        if len(times) < 20 or duration * frequency < 2.5:
            unmeasured.append(dict(**spec, reason="fewer than 2.5 measurement cycles or insufficient states"))
            continue
        ci = np.searchsorted(ct, times, side="right")-1
        reference = np.array([commands[i][4] for i in ci])
        t = times-begin
        input_fit, input_residual = fit_sine(t, reference, frequency)
        output_fit, output_residual = fit_sine(t, q[mask], frequency)
        roll_fit, _ = fit_sine(t, roll[mask], frequency)
        if abs(input_fit) < 1e-6:
            raise ValueError("input amplitude is too small to identify a response")
        response = output_fit/input_fit
        quality = dynamic_quality(t, reference, q[mask], frequency, response)
        flags = []
        kp_values = {commands[i][6] for i in indices}
        kd_values = {commands[i][7] for i in indices}
        if len(kp_values) != 1 or len(kd_values) != 1:
            flags.append("gains_changed_during_measurement")
        if meta["status"] != "completed" or meta.get("source_status", "completed") != "completed":
            flags.append("run_or_replay_source_did_not_complete")
        if not complete_window:
            flags.append("incomplete_frequency_window")
        if abs(output_fit) < 3*output_residual:
            flags.append("low_output_sine_to_residual_ratio")
        if abs(response) < 0.1:
            flags.append("very_small_response_check_stiction_and_resolution")
        if np.max(np.diff(times)) > 0.02:
            flags.append("state_receive_gap_over_20ms")
        if quality["output_harmonic_ratio_2_3"] is not None and quality["output_harmonic_ratio_2_3"]>.2:
            flags.append("nonlinear_harmonic_distortion")
        if quality["cycle_gain_cv"] is not None and quality["cycle_gain_cv"]>.2:
            flags.append("cycle_gain_not_repeatable")
        if quality["cycle_phase_std_deg"] is not None and quality["cycle_phase_std_deg"]>15:
            flags.append("cycle_phase_not_repeatable")
        rows.append(dict(command_amplitude_deg=spec["amplitude_deg"], frequency_hz=frequency,
            excitation_phase=spec["excitation_phase"], kp=commands[indices[0]][6], kd=commands[indices[0]][7],
            gain=abs(response),
            gain_db=20*math.log10(abs(response)) if abs(response)>0 else None,
            phase_deg=math.degrees(np.angle(response)), input_amplitude_deg=math.degrees(abs(input_fit)),
            output_amplitude_deg=math.degrees(abs(output_fit)),
            output_min_deg=math.degrees(float(q[mask].min())), output_max_deg=math.degrees(float(q[mask].max())),
            output_peak_to_peak_deg=math.degrees(float(np.ptp(q[mask]))),
            peak_speed_deg_s=math.degrees(float(np.abs(dq[mask]).max())),
            low_speed_fraction_under_0p2_deg_s=float(np.mean(np.abs(dq[mask]) < math.radians(.2))),
            peak_tracking_error_deg=math.degrees(float(np.abs(q[mask]-reference).max())),
            roll_coupling_gain=abs(roll_fit/input_fit),
            tracking_rmse_deg=math.degrees(float(np.sqrt(np.mean((q[mask]-reference)**2)))),
            mean_tracking_error_deg=math.degrees(float(np.mean(q[mask]-reference))),
            output_fit_residual_deg=math.degrees(output_residual),
            input_fit_residual_deg=math.degrees(input_residual),
            peak_abs_tau_est=float(np.max(np.abs(tau[mask]))), samples=len(times),
            **quality, quality_flags=flags))
    if not rows:
        raise ValueError("no frequency has at least 2.5 constant-amplitude cycles")
    result = dict(log=str(path.resolve()), status=meta["status"], side=meta["args"]["side"],
        kp_scale=meta["args"]["kp_scale"], kd_scale=meta["args"]["kd_scale"],
        median_state_rate_hz=float(1/np.median(np.diff(st))),
        max_state_receive_gap_ms=float(1000*np.max(np.diff(st))),
        median_command_rate_hz=float(1/np.median(np.diff(ct))),
        timebase=("synthetic simulation time aligned to original commands; no network delay" if meta.get("offline_replay")
                  else "workstation send/receive monotonic clock; includes transport and feedback delay"),
        caution="Local closed-loop FRF only. Phase is not pure motor delay; tau_est is not calibrated output torque.",
        center_command_deg=math.degrees(meta["base_q"][index]),
        excitation_amplitude_deg=meta["args"].get("amplitude_deg"),
        offline_replay=meta.get("offline_replay", False), source_log=meta.get("source_log"),
        quality_note="Harmonic/cycle thresholds are diagnostic heuristics. Do not fit linear damping/bandwidth to weak or nonlinear responses.",
        frequencies=rows, unmeasured_trials=unmeasured)
    if meta["args"]["target"] == "sim":
        elapsed_ticks = (states[-1][4]-states[0][4]) & 0xffffffff
        result["sim_time_to_wall_time_ratio"] = elapsed_ticks / 1000 / (st[-1]-st[0])
        if abs(result["sim_time_to_wall_time_ratio"]-1) > .05:
            result["caution"] += " Simulation was not real-time: do not compare its FRF with hardware; use offline replay."
            for row in rows:
                row["quality_flags"].append("simulation_not_realtime")
    result["amplitude_comparison"] = amplitude_comparisons(rows)
    with (path / "analysis.json").open("w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    with (path / "frequency_response.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def compare_replay(real, sim, output):
    if not sim["offline_replay"] or sim["source_log"] != real["log"] or real["offline_replay"]:
        raise ValueError("comparison requires the original log followed by its own offline replay")
    for key in ("side", "kp_scale", "kd_scale", "center_command_deg"):
        if real[key] != sim[key]:
            raise ValueError(f"comparison mismatch: {key}")
    key = lambda row: (row["command_amplitude_deg"], row["frequency_hz"])
    simulated = {key(row): row for row in sim["frequencies"]}
    rows = []
    for actual in real["frequencies"]:
        if key(actual) not in simulated:
            continue
        predicted = simulated[key(actual)]
        if (actual["kp"], actual["kd"]) != (predicted["kp"], predicted["kd"]):
            raise ValueError("recorded per-trial gains differ")
        rows.append(dict(amplitude_deg=actual["command_amplitude_deg"], frequency_hz=actual["frequency_hz"],
            kp=actual["kp"], kd=actual["kd"], real_gain=actual["gain"], sim_gain=predicted["gain"],
            real_to_sim_gain_ratio=actual["gain"]/predicted["gain"] if predicted["gain"]>1e-8 else None,
            real_phase_deg=actual["phase_deg"], sim_phase_deg=predicted["phase_deg"],
            extra_real_lag_deg=math.degrees(np.angle(np.exp(1j*math.radians(predicted["phase_deg"]-actual["phase_deg"])))),
            real_peak_to_peak_deg=actual["output_peak_to_peak_deg"], sim_peak_to_peak_deg=predicted["output_peak_to_peak_deg"],
            real_harmonic_ratio_2_3=actual["output_harmonic_ratio_2_3"],
            sim_harmonic_ratio_2_3=predicted["output_harmonic_ratio_2_3"],
            real_quality_flags=actual["quality_flags"], sim_quality_flags=predicted["quality_flags"]))
    if not rows:
        raise ValueError("no matched amplitude/frequency trials")
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("logs", nargs="+", type=Path, help="one real/sim log, or several logs to compare")
    p.add_argument("--compare-output", type=Path, help="paired original/replay CSV (requires exactly two logs)")
    a = p.parse_args()
    if a.compare_output and len(a.logs) != 2:
        p.error("--compare-output requires original log and its offline replay")
    results = []
    for path in a.logs:
        result = analyze(path)
        results.append(result)
        print(f"\n{path}: {result['side']} Kp scale={result['kp_scale']} Kd scale={result['kd_scale']}")
        print(f"State median {result['median_state_rate_hz']:.1f} Hz; status={result['status']}")
        print(result["caution"])
        print("Amp(deg) Hz       gain      phase(deg)   output p-p(deg)  tracking RMSE(deg)   flags")
        for row in result["frequencies"]:
            print(f"{row['command_amplitude_deg']:<8g} {row['frequency_hz']:<8g} {row['gain']:<9.4f} {row['phase_deg']:<12.2f} "
                  f"{row['output_peak_to_peak_deg']:<16.4f} {row['tracking_rmse_deg']:<20.4f} "
                  f"{','.join(row['quality_flags'])}")
        for row in result["amplitude_comparison"]:
            print(f"Shared {row['frequency_hz']:g} Hz: amplitudes={row['amplitudes_deg']}, "
                  f"gains={[round(v, 4) for v in row['gains']]}")
        if result["unmeasured_trials"]:
            print(f"Unmeasured/incomplete trials: {len(result['unmeasured_trials'])}; see analysis.json")
    if a.compare_output:
        compared = compare_replay(*results, a.compare_output)
        print(f"Paired real/replay comparison: {len(compared)} trials -> {a.compare_output}")


if __name__ == "__main__":
    main()
