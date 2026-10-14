#!/usr/bin/env python3
"""Read-only effort/timing audit of existing H2 squat and unloaded logs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AXES = [('left_knee', 3), ('left_ankle_roll', 4), ('left_ankle_pitch', 5),
        ('right_knee', 9), ('right_ankle_roll', 10), ('right_ankle_pitch', 11)]


def read_numeric(path, names, time_name, scale=1, start=-np.inf, end=np.inf,
                 extra_names=()):
    rows, extra = [], {n: [] for n in extra_names}
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        indices = [header.index(n) for n in names]
        ti = header.index(time_name)
        ei = {n: header.index(n) for n in extra_names}
        for row in reader:
            time = float(row[ti]) * scale
            if time > end:
                break
            if time < start:
                continue
            rows.append([float(row[i]) for i in indices])
            for n, i in ei.items():
                extra[n].append(row[i])
    if not rows:
        raise ValueError(f'No samples selected from {path}')
    result = np.asarray(rows, dtype=np.float64)
    return {n: result[:, i] for i, n in enumerate(names)}, {
        n: np.asarray(v) for n, v in extra.items()}


def scalar_stats(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(count=0)
    return dict(count=len(a), mean=float(np.mean(a)), std=float(np.std(a)),
                min=float(np.min(a)), max=float(np.max(a)),
                p05=float(np.percentile(a, 5)), p50=float(np.median(a)),
                p95=float(np.percentile(a, 95)), p99=float(np.percentile(a, 99)))


def causal_indices(command_time, state_time):
    if np.any(np.diff(command_time) < 0) or np.any(np.diff(state_time) < 0):
        raise ValueError('Non-monotonic arrival times; causal matching is undefined')
    return np.searchsorted(command_time, state_time, side='right') - 1


def linear_relation(x, y):
    good = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[good], np.asarray(y)[good]
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return dict(samples=len(x), correlation=None, gain=None, intercept=None,
                    residual_rms=None, identity_rms=None)
    xc, yc = x - x.mean(), y - y.mean()
    gain = float(xc @ yc / (xc @ xc))
    intercept = float(y.mean() - gain*x.mean())
    return dict(samples=len(x), correlation=float(np.corrcoef(x, y)[0, 1]),
                gain=gain, intercept=intercept,
                residual_rms=float(np.sqrt(np.mean((y-gain*x-intercept)**2))),
                identity_rms=float(np.sqrt(np.mean((y-x)**2))))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def span(values, digits=3):
    values = list(values)
    return f'{min(values):.{digits}f}-{max(values):.{digits}f}'


def lag_scan(t, pd, tau, mask, max_lag_ms):
    rows = []
    # Shift the reconstructed complete effort, including measured q/dq. This is
    # an arrival-clock sensitivity scan, not an estimate of acquisition delay.
    for lag in range(max_lag_ms + 1):
        query = t - lag
        good = mask & (query >= t[0])
        past = np.interp(query[good], t, pd)
        rows.append(dict(lag_ms=lag, **linear_relation(past, tau[good])))
    return rows


def channel_summary(t, q, dq, tau, pd, target, kp, kd, mask):
    output = dict(samples=int(mask.sum()), kp_min=float(kp[mask].min()),
                  kp_max=float(kp[mask].max()), kd_min=float(kd[mask].min()),
                  kd_max=float(kd[mask].max()))
    for name, value in [('q_deg', np.rad2deg(q)), ('dq_deg_s', np.rad2deg(dq)),
                        ('target_deg', np.rad2deg(target)), ('tau_est', tau),
                        ('pd_requested', pd), ('target_minus_q_deg', np.rad2deg(target-q))]:
        selected = value[mask]
        output.update({name+'_'+n: v for n, v in scalar_stats(selected).items() if n != 'count'})
        output[name+'_rms'] = float(np.sqrt(np.mean(selected**2)))
        output[name+'_abs_p95'] = float(np.percentile(np.abs(selected), 95))
    selected = tau[mask]
    output['tau_unique_values'] = len(np.unique(selected))
    output['tau_zero_fraction'] = float(np.mean(selected == 0))
    output['tau_identical_adjacent_fraction'] = float(np.mean(np.diff(selected) == 0))
    adjacent = mask[:-1] & mask[1:] & (np.diff(t) < 5)
    for name, value in [('q', q), ('dq', dq), ('tau_est', tau)]:
        output[name+'_identical_adjacent_fraction'] = float(np.mean(np.diff(value)[adjacent] == 0))
    substantial = mask & (np.abs(pd) > 5)
    output['same_sign_fraction_abs_pd_over_5'] = (float(np.mean(
        np.sign(tau[substantial]) == np.sign(pd[substantial]))) if np.any(substantial) else None)
    output['tau_near_observed_abs_max_fraction'] = float(np.mean(
        np.abs(selected) >= .99 * np.abs(selected).max()))
    finite = np.isfinite(q) & np.isfinite(dq) & np.isfinite(tau) & np.isfinite(pd)
    output['nonfinite_samples'] = int(np.sum(mask & ~finite))
    return output


def publisher_summary(t, handles, windows):
    result = []
    for handle in np.unique(handles):
        mask = handles == handle
        result.append(dict(handle=handle, samples=int(mask.sum()), first_ms=float(t[mask][0]),
                           last_ms=float(t[mask][-1]),
                           active_samples=[int(np.sum(mask & (t >= a) & (t < b))) for a, b in windows]))
    return result


def audit_squat(run, output, max_lag_ms):
    from analyze_h2_squat_comparison import load_deploy
    deploy = load_deploy(run)
    windows = [(float(deploy['t'][g[0]]), float(deploy['t'][g[-1]] + 20)) for g in deploy['groups']]
    if len(windows) != 3:
        raise ValueError(f'Expected three completed playbacks, got {len(windows)}')
    meta = json.loads((run/'dds/metadata.json').read_text())
    start, end = windows[0][0] - 100, windows[-1][1] + 100
    sn = ['time_monotonic_ms', 'time_realtime_ms', 'dds_source_timestamp_ns', 'tick',
          'mode_pr', 'mode_machine', 'crc_ok']
    sn += [f'{p}_{j}' for p in ('q', 'dq', 'tau_est', 'motor_error', 'temperature_0', 'temperature_1') for _, j in AXES]
    cn = ['time_monotonic_ms', 'time_realtime_ms', 'dds_source_timestamp_ns', 'mode_pr', 'mode_machine', 'crc_ok']
    cn += [f'{p}_{j}' for p in ('mode', 'q_target', 'dq_target', 'kp', 'kd', 'tau_ff') for _, j in AXES]
    state, se = read_numeric(run/'dds/lowstate.csv', sn, 'time_monotonic_ms', start=start, end=end,
                             extra_names=['publisher_handle'])
    # Read all command handles to distinguish startup discovery from active data.
    command, ce = read_numeric(run/'dds/lowcmd.csv', cn, 'time_monotonic_ms', extra_names=['publisher_handle'])
    t, ct = state['time_monotonic_ms'], command['time_monotonic_ms']
    ci = causal_indices(ct, t)
    if np.any(ci < 0):
        raise ValueError('Missing preceding command for selected states')
    repeat = np.zeros(len(t), dtype=int)
    for r, (a, b) in enumerate(windows, 1):
        repeat[(t >= a) & (t < b)] = r
    active = repeat > 0
    valid = (state['crc_ok'] == 1) & (command['crc_ok'][ci] == 1)
    valid &= (state['mode_pr'] == 0) & (command['mode_pr'][ci] == 0)
    if not np.all(valid[active]):
        raise ValueError('Invalid CRC or non-PR data in active playback; refusing effort reconstruction')
    quality = dict(source=str(run), windows=windows, metadata=meta,
        active_rows=int(active.sum()), invalid_active_rows=int(np.sum(active & ~valid)),
            command_handles=publisher_summary(ct, ce['publisher_handle'], windows),
        state_handles=publisher_summary(t, se['publisher_handle'], windows),
        state_mode_pr=np.unique(state['mode_pr'][active]).tolist(),
        command_mode_pr=np.unique(command['mode_pr'][ci][active]).tolist(),
        timing=[], semantics=dict(
            effort='kp*(q_target-q)+kd*(dq_target-dq)+tau_ff, causal LowCmd in recorder arrival clock',
            tau_est='Drive estimate; coordinate conversion and calibration not established by mode_pr=0',
            lag='Descriptive 0..30 ms past-effort sensitivity; NOT a calibrated transport/motor delay',
            publisher='Per-message handle counts are exact strings; discovery count lacks timestamped event history'))
    summaries, lags, paired = [], [], dict(time_monotonic_ms=t, repeat=repeat, valid=valid,
                                         command_age_ms=t-ct[ci], joint_dds_indices=np.array([j for _, j in AXES]))
    for r, (a, b) in enumerate(windows, 1):
        mask = repeat == r
        tick = state['tick'][mask]
        dt = np.diff(t[mask])
        dc = ct[(ct >= a) & (ct < b)]
        src = state['dds_source_timestamp_ns'][mask] / 1e6
        source_elapsed = src-src[0]
        tick_elapsed = tick-tick[0]
        quality['timing'].append(dict(repeat=r, state_arrival_interval_ms=scalar_stats(dt),
            tick_delta=scalar_stats(np.diff(tick)), tick_zero_count=int(np.sum(np.diff(tick) == 0)),
            tick_negative_count=int(np.sum(np.diff(tick) < 0)),
            command_arrival_interval_ms=scalar_stats(np.diff(dc)),
            command_age_at_state_ms=scalar_stats(t[mask]-ct[ci][mask]),
            elapsed_arrival_minus_tick_ms=scalar_stats((t[mask]-t[mask][0])-tick_elapsed),
            elapsed_source_minus_tick_ms=scalar_stats(source_elapsed-tick_elapsed),
            source_stamp_zero_fraction=float(np.mean(src == 0)),
            source_minus_host_realtime_ms=scalar_stats(src-state['time_realtime_ms'][mask]),
            command_publisher_handles=np.unique(ce['publisher_handle'][ci][mask]).tolist(),
            motor_errors=int(sum(np.count_nonzero(state[f'motor_error_{j}'][mask]) for _, j in AXES))))
    for name, j in AXES:
        q, dq, tau = (state[f'{p}_{j}'] for p in ('q', 'dq', 'tau_est'))
        target, vt, kp, kd, ff = (command[f'{p}_{j}'][ci] for p in ('q_target', 'dq_target', 'kp', 'kd', 'tau_ff'))
        pd = kp*(target-q)+kd*(vt-dq)+ff
        for n, values in [('q', q), ('dq', dq), ('tau_est', tau), ('pd', pd), ('target', target), ('kp', kp), ('kd', kd)]:
            paired[f'{n}_{j}'] = values
        for r in (0, 1, 2, 3):
            mask = active if r == 0 else repeat == r
            summaries.append(dict(joint=name, dds=j, repeat=r, **channel_summary(t, q, dq, tau, pd, target, kp, kd, mask)))
            for row in lag_scan(t, pd, tau, mask, max_lag_ms):
                lags.append(dict(joint=name, dds=j, repeat=r, **row))
    np.savez_compressed(output/'squat_paired.npz', **paired)
    write_csv(output/'squat_channels.csv', summaries)
    write_csv(output/'squat_lag_sensitivity.csv', lags)
    # A two-input description can flag coordinate coupling, but is not a motor map.
    coupling = []
    for side, roll, pitch in [('left', 4, 5), ('right', 10, 11)]:
        for target in (roll, pitch):
            x = np.column_stack((paired[f'pd_{roll}'], paired[f'pd_{pitch}'], np.ones(len(t))))
            train = (repeat == 1) | (repeat == 2)
            test = repeat == 3
            y = paired[f'tau_est_{target}']
            coef = np.linalg.lstsq(x[train], y[train], rcond=None)[0]
            one = np.column_stack((paired[f'pd_{target}'], np.ones(len(t))))
            coef1 = np.linalg.lstsq(one[train], y[train], rcond=None)[0]
            coupling.append(dict(side=side, tau_est_dds=target, roll_coefficient=float(coef[0]),
                pitch_coefficient=float(coef[1]), intercept=float(coef[2]),
                heldout_rms_two_input=float(np.sqrt(np.mean((x[test]@coef-y[test])**2))),
                heldout_rms_same_axis=float(np.sqrt(np.mean((one[test]@coef1-y[test])**2))),
                fit='playbacks 1+2; evaluate 3; no lag; descriptive only'))
    write_csv(output/'ankle_cross_axis_description.csv', coupling)
    return quality, summaries, lags, coupling


def audit_unloaded(path, output, max_lag_ms):
    from analyze_h2_ankle_sysid import trial_specs
    meta = json.loads((path/'metadata.json').read_text())
    j = meta['active_index']
    names = ['send_end_monotonic_ns', 'frequency_hz', 'measure']
    names += [f'{p}_{j}' for p in ('q_des', 'dq_des', 'kp', 'kd', 'tau_ff')]
    command, extra = read_numeric(path/'commands.csv', names, 'send_end_monotonic_ns', extra_names=['phase'])
    ct = command['send_end_monotonic_ns']/1e6
    sn = ['receive_monotonic_ns', 'tick', 'mode_pr', 'crc_ok', f'q_{j}', f'dq_{j}', f'tau_est_{j}']
    state, _ = read_numeric(path/'states.csv', sn, 'receive_monotonic_ns', scale=1e-6,
                            start=ct[0], end=ct[-1])
    t = state['receive_monotonic_ns']/1e6
    ci = causal_indices(ct, t)
    q, dq, tau = (state[f'{p}_{j}'] for p in ('q', 'dq', 'tau_est'))
    target, vt, kp, kd, ff = (command[f'{p}_{j}'][ci] for p in ('q_des', 'dq_des', 'kp', 'kd', 'tau_ff'))
    pd = kp*(target-q)+kd*(vt-dq)+ff
    rows, lag_rows = [], []
    for spec in trial_specs(meta):
        mask = ((extra['phase'][ci] == spec['excitation_phase']) & (command['measure'][ci] == 1)
                & (command['frequency_hz'][ci] == spec['frequency_hz']))
        if not np.any(mask):
            continue
        values = dict(run=path.name, dds=j, amplitude_deg=spec['amplitude_deg'], frequency_hz=spec['frequency_hz'])
        if np.any((state['mode_pr'][mask] != 0) | (state['crc_ok'][mask] != 1)):
            raise ValueError(f'Invalid unloaded PR/CRC data in {path}: {spec}')
        rows.append(dict(**values, **channel_summary(t, q, dq, tau, pd, target, kp, kd, mask)))
        for row in lag_scan(t, pd, tau, mask, max_lag_ms):
            lag_rows.append(dict(**values, **row))
    write_csv(output/f'unloaded_{j}_channels.csv', rows)
    write_csv(output/f'unloaded_{j}_lag_sensitivity.csv', lag_rows)
    return dict(source=str(path), mode_pr=np.unique(state['mode_pr']).tolist(),
                invalid_crc=int(np.sum(state['crc_ok'] != 1)), trials=len(rows),
                command_clock='send_end_monotonic_ns; local publish completion, not actuator receipt',
                command_age_at_state_ms=scalar_stats(t-ct[ci])), rows, lag_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'gear_sonic_deploy/logs/squat_compare/real01')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-lag-ms', type=int, default=30)
    parser.add_argument('--unloaded', type=Path, action='append', default=[])
    args = parser.parse_args(argv)
    if not 0 <= args.max_lag_ms <= 100:
        parser.error('--max-lag-ms must be within 0..100')
    args.output.mkdir(parents=True, exist_ok=False)
    quality, rows, lags, coupling = audit_squat(args.run, args.output, args.max_lag_ms)
    unloaded = []
    unloaded_rows = []
    unloaded_lags = []
    for path in args.unloaded:
        audit, channel_rows, lag_rows = audit_unloaded(path, args.output, args.max_lag_ms)
        unloaded.append(audit)
        unloaded_rows.extend(channel_rows)
        unloaded_lags.extend(lag_rows)
    quality['unloaded'] = unloaded
    (args.output/'audit.json').write_text(json.dumps(quality, indent=2, allow_nan=False)+'\n')
    report = ['# H2 actuator effort observability audit', '',
        'This is an offline diagnostic, not an identified actuator model. No gains, model files, or robot processes are changed.', '',
        '## Data and timing', '',
        f"Active squat LowState samples: {quality['active_rows']}; valid PR/CRC samples required.",
        'The recorder discovery maximum is not a timestamped publisher history. Per-message publisher handles are reported in audit.json; no competing writer is inferred from discovery alone.',
        'LowState arrival, DDS source stamp, and hardware tick are different clocks. Source timestamps have an unverified remote clock offset. Scanned lags shift reconstructed PD backward on the local arrival clock; a best lag is NOT measured motor delay.', '',
        '| Command publisher | All recorded samples | Active samples in playbacks 1 / 2 / 3 |',
        '| --- | ---: | --- |']
    for handle in quality['command_handles']:
        report.append(f"| {handle['handle']} | {handle['samples']} | {' / '.join(map(str, handle['active_samples']))} |")
    per_window_handles = [len(item['command_publisher_handles']) for item in quality['timing']]
    report += ['', f'Active command source counts by playback: {per_window_handles}. Evaluate active counts independently from the recorder discovery maximum.', '',
        f"Per-playback LowCmd intervals average {span(item['command_arrival_interval_ms']['mean'] for item in quality['timing'])} ms; causal command age at LowState arrival has p95 {span(item['command_age_at_state_ms']['p95'] for item in quality['timing'])} ms. LowState arrival intervals average {span(item['state_arrival_interval_ms']['mean'] for item in quality['timing'])} ms with maxima {span(item['state_arrival_interval_ms']['max'] for item in quality['timing'])} ms.",
        'Integer tick repeats occur, but time-varying tick granularity is not proof of repeated sensor data. Source and host clocks must not be directly subtracted as transport delay without synchronization evidence.', '',
        '## Squat effort comparison', '',
        '| Joint | PD abs p95 | tau_est abs p95 | Lag-0 correlation | Lag-0 fitted slope | Best scanned lag | Best correlation |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, j in AXES:
        s = next(r for r in rows if r['dds'] == j and r['repeat'] == 0)
        ls = [r for r in lags if r['dds'] == j and r['repeat'] == 0]
        best = max(ls, key=lambda r: r['correlation'] if r['correlation'] is not None else -np.inf)
        first = ls[0]
        report.append(f"| {name} | {s['pd_requested_abs_p95']:.3f} | {s['tau_est_abs_p95']:.3f} | {first['correlation']:.3f} | {first['gain']:.3f} | {best['lag_ms']} ms | {best['correlation']:.3f} |")
    report += ['', 'Torque-like values retain logged units; Nm calibration and PR output-effort semantics are not established for tau_est.', '',
        'A high tau_est/PD correlation does not prove delivered output torque is correct: tau_est may share controller calculations or hidden filtering rather than being an independent calibrated sensor. Do not fit a request-to-output gain loss using this relationship without validating that independence.', '',
        'Pitch/roll two-input regression fits repetitions 1+2 and evaluates repetition 3. Its coefficients describe correlated signals, not a known transmission map. Held-out pitch residual RMS before/after adding roll PD:', '']
    for j in (5, 11):
        item = next(row for row in coupling if row['tau_est_dds'] == j)
        s = next(row for row in rows if row['dds'] == j and row['repeat'] == 0)
        report += [f"- {item['side']}: {item['heldout_rms_same_axis']:.3f} / {item['heldout_rms_two_input']:.3f} logged torque units.",
            f"  Identical-adjacent fractions: tau_est={s['tau_est_identical_adjacent_fraction']:.3f}, q={s['q_identical_adjacent_fraction']:.3f}, dq={s['dq_identical_adjacent_fraction']:.3f}. Different update/hold/quantization behavior must not be mistaken for physical torque saturation."]
    report += ['',
        '## Unloaded effort comparison', '',
        '| DDS ankle pitch | Conditions | Lag-0 slope range | Lag-0 correlation range | Best scanned lag range |',
        '| --- | ---: | ---: | ---: | ---: |']
    for j in sorted({row['dds'] for row in unloaded_rows}):
        selected = [row for row in unloaded_lags if row['dds'] == j]
        zeros = [row for row in selected if row['lag_ms'] == 0]
        best = []
        for condition in [row for row in unloaded_rows if row['dds'] == j]:
            ls = [row for row in selected if row['amplitude_deg'] == condition['amplitude_deg']
                  and row['frequency_hz'] == condition['frequency_hz']]
            best.append(max(ls, key=lambda row: row['correlation'])['lag_ms'])
        report.append(f"| {j} | {len(zeros)} | {min(r['gain'] for r in zeros):.3f}-{max(r['gain'] for r in zeros):.3f} | {min(r['correlation'] for r in zeros):.3f}-{max(r['correlation'] for r in zeros):.3f} | {min(best)}-{max(best)} ms |")
    report += ['', 'Unloaded lag-zero fitted offset ranges in logged torque units:']
    for j in sorted({row['dds'] for row in unloaded_rows}):
        offsets = [row['intercept'] for row in unloaded_lags if row['dds'] == j and row['lag_ms'] == 0]
        report.append(f'- DDS {j}: {span(offsets)}.')
    report += ['', 'Offsets matter relative to the small unloaded effort. Do not force a single through-zero torque gain across operating regimes.', '',
        '## Interpretation limits', '',
        '- A fitted tau_est/PD slope is a descriptive signal relationship, not an actuator efficiency or gain calibration.',
        '- Ankle pitch/roll channels may involve coupled coordinate conversion. mode_pr=0 establishes the requested command mode, not the torque-estimator implementation.',
        '- Reconstructed PD uses measured q/dq and therefore carries feedback, estimator filtering, and clock uncertainty. It is not the actual inner-loop effort before hardware limits.',
        '- Constant-value and near-maximum fractions are saturation hints only. No saturation threshold is identified from an observed maximum.',
        '- Three squat repetitions do not provide independent excitation of knees, ankles, contact loading, and body motion. A coupled regression can reveal correlation but cannot uniquely identify the transmission.',
        '- The unloaded sweeps use actual logged publication timestamps; their lower load cannot establish loaded effort limits.',
        '- Original references, physical models, gains and logs remain untouched.']
    (args.output/'README.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(output=str(args.output), active_rows=quality['active_rows'],
                          command_handles=quality['command_handles'], unloaded=unloaded), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
