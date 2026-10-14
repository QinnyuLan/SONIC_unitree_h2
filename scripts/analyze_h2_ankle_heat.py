#!/usr/bin/env python3
"""Analyze H2 deploy CSV logs; read-only, no DDS, no robot connection."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

JOINTS = {4: 'left_roll', 10: 'right_roll', 5: 'left_pitch', 11: 'right_pitch'}
LABELS = {4: '左 roll', 10: '右 roll', 5: '左 pitch', 11: '右 pitch'}


def read_table(path, columns):
    with path.open() as stream:
        rows = csv.DictReader(stream)
        required = ['time_monotonic_ms', *columns]
        missing = set(required) - set(rows.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        a = np.asarray([[float(row[k]) for k in required] for row in rows], dtype=float)
    if a.ndim != 2 or len(a) < 2 or not np.isfinite(a[:, 0]).all():
        raise ValueError(f'{path}: empty/invalid timestamps')
    if np.any(np.diff(a[:, 0]) < 0):
        raise ValueError(f'{path}: timestamps are not monotonic')
    return a[:, 0], a[:, 1:]


def previous_sample(times, source_times, source_values, max_age_ms):
    indices = np.searchsorted(source_times, times, side='right') - 1
    safe = np.clip(indices, 0, len(source_times) - 1)
    age = times - source_times[safe]
    valid = (indices >= 0) & (age <= max_age_ms) & np.isfinite(source_values[safe]).all(axis=1)
    result = source_values[safe].copy()
    result[~valid] = np.nan
    return result, valid, age


def rms(values):
    return float(np.sqrt(np.mean(values ** 2)))


def summarize(q, dq, target, dqtarget, kp, kd, ff, torque, mask):
    if not mask.any():
        return {'samples': 0}
    err = target[mask] - q[mask]
    p = kp[mask] * err
    d = kd[mask] * (dqtarget[mask] - dq[mask])
    return dict(samples=int(mask.sum()), kp_median=float(np.median(kp[mask])),
        kd_median=float(np.median(kd[mask])), error_rms_deg=rms(np.rad2deg(err)),
        error_bias_deg=float(np.mean(np.rad2deg(err))),
        error_peak_deg=float(np.max(np.abs(np.rad2deg(err)))),
        velocity_rms_rad_s=rms(dq[mask]), tau_est_rms=rms(torque[mask]),
        tau_est_abs_peak=float(np.max(np.abs(torque[mask]))),
        tau_est_mean=float(np.mean(torque[mask])),
        p_term_rms=rms(p), d_term_rms=rms(d), pd_request_rms=rms(p + d + ff[mask]))


def analyze(directory):
    meta = json.loads((directory / 'metadata.json').read_text())
    if meta.get('robot_config', {}).get('robot_profile') != 'h2_native':
        raise ValueError('Expected h2_native deployment log with verified joint ordering')
    ids = list(JOINTS)
    times, q = read_table(directory / 'q.csv', [f'q_{i}' for i in ids])
    times_s = (times - times[0]) / 1000
    all_valid = np.isfinite(q).all(axis=1)
    arrays = {}
    quality = {}
    for name, file, columns in [
        ('dq', 'dq.csv', [f'dq_{i}' for i in ids]),
        ('torque', 'motor_torque.csv', [f'tau_{i}' for i in ids]),
        ('temperature', 'motor_temperature.csv', [f'temp_{2*i+c}' for i in ids for c in (0, 1)]),
        ('errors', 'motor_error.csv', [f'err_{i}' for i in ids]),
    ]:
        st, values = read_table(directory / file, columns)
        arrays[name], valid, _ = previous_sample(times, st, values, 1.0)
        all_valid &= valid
        quality[name + '_unmatched_rows'] = int((~valid).sum())
    groups = ['q_target', 'dq_target', 'kp', 'kd', 'tau_ff']
    ct, commands = read_table(directory / 'lowcmd_command.csv',
                             ['program_state', 'start', 'play'] + [f'{g}_{i}' for g in groups for i in ids])
    command, valid, age = previous_sample(times, ct, commands, 30.0)
    quality.update(state_rows=len(times), command_rows=len(ct),
        state_gap_max_ms=float(np.max(np.diff(times))), command_unmatched_rows=int((~valid).sum()),
        max_matched_command_age_ms=float(np.max(age[valid])) if valid.any() else None,
        alignment='last published command at or before state log time, max age 30 ms; arrival/log clocks, not hardware acquisition times')
    all_valid &= valid
    active = all_valid & (command[:, 0] == 2) & (command[:, 1] == 1)
    if active.sum() < 2:
        raise ValueError('Not enough matched active-control samples to analyze')
    playing = active & (command[:, 2] == 1)
    holding = active & ~playing
    report = dict(source=str(directory.resolve()), duration_s=float(times_s[-1]),
                  robot_config=meta['robot_config'], quality=quality, joints={})
    aligned_columns = ['time_s', 'active', 'playing']
    aligned = [times_s, active.astype(int), playing.astype(int)]
    for col, joint in enumerate(ids):
        target, dqtarget, kp, kd, ff = [command[:, 3 + g*4 + col] for g in range(5)]
        dq, torque = arrays['dq'][:, col], arrays['torque'][:, col]
        temperatures = arrays['temperature'][:, 2*col:2*col+2]
        row = {'dds_id': joint}
        for phase, mask in [('active', active), ('playing', playing), ('holding', holding)]:
            row[phase] = summarize(q[:, col], dq, target, dqtarget, kp, kd, ff, torque, mask)
        row['temperature_channels'] = []
        for channel in range(2):
            temp = temperatures[:, channel]
            mask = np.isfinite(temp)
            t, y = times_s[mask], temp[mask]
            row['temperature_channels'].append(dict(channel=channel, start=float(y[0]), end=float(y[-1]),
                peak=float(y.max()), rise=float(y[-1]-y[0]), slope_C_per_min=float(np.polyfit(t-t[0], y, 1)[0]*60)))
        error = arrays['errors'][:, col]
        row['motor_status_values'] = sorted(set(int(v) for v in error[np.isfinite(error)]))
        report['joints'][JOINTS[joint]] = row
        pd = kp*(target-q[:, col]) + kd*(dqtarget-dq) + ff
        for name, values in [('q_rad', q[:, col]), ('q_target_rad', target), ('dq_rad_s', dq),
                             ('kp', kp), ('kd', kd), ('tau_ff', ff), ('tau_est', torque),
                             ('pd_request', pd), ('temp_0', temperatures[:, 0]),
                             ('temp_1', temperatures[:, 1]), ('motor_status', error)]:
            aligned_columns.append(f'{JOINTS[joint]}_{name}');aligned.append(values)
    out = directory / 'ankle_heat_analysis';out.mkdir(exist_ok=True)
    np.savetxt(out/'aligned_ankles.csv', np.column_stack(aligned), delimiter=',',
               header=','.join(aligned_columns), comments='', fmt='%.7g')
    (out/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    lines = ['# H2 脚踝温升与跟踪分析', '', f'日志：`{directory.resolve()}`',
             f'时长：{times_s[-1]:.1f} s；控制样本 {active.sum()}，播放样本 {playing.sum()}，保持样本 {holding.sum()}。', '',
             '| 关节 | 温度0 起→末/峰值 °C | 温度1 起→末/峰值 °C | 误差 RMS ° | 估计力矩 RMS | PD请求 RMS | Kp/Kd |',
             '|---|---|---|---:|---:|---:|---|']
    for joint, name in JOINTS.items():
        r=report['joints'][name];a=r['active'];t=r['temperature_channels']
        temp=lambda x:f"{x['start']:.0f}→{x['end']:.0f}/{x['peak']:.0f}"
        lines.append(f"| {LABELS[joint]} | {temp(t[0])} | {temp(t[1])} | {a['error_rms_deg']:.2f} | {a['tau_est_rms']:.3f} | {a['pd_request_rms']:.3f} | {a['kp_median']:.1f}/{a['kd_median']:.1f} |")
    l,r=report['joints']['left_roll'],report['joints']['right_roll']
    lines += ['', f"右/左 roll：估计力矩 RMS 比 {r['active']['tau_est_rms']/max(l['active']['tau_est_rms'],1e-9):.2f}，"
              f"PD 请求 RMS 比 {r['active']['pd_request_rms']/max(l['active']['pd_request_rms'],1e-9):.2f}，"
              f"速度 RMS 比 {r['active']['velocity_rms_rad_s']/max(l['active']['velocity_rms_rad_s'],1e-9):.2f}。", '',
              '温度0/1保留 SDK 原始通道；本地代码标为绕组/驱动，但应以该固件定义为准。',
              'tau_est 是驱动估计量，不是校准的输出关节力矩；PD请求是 Kp*(q目标-q)+Kd*(dq目标-dq)+前馈，未考虑执行器饱和与闭链转换。',
              '本日志没有电流、电压、绕组电阻和真实热损耗，不能由力矩推导实际发热功率，不能仅凭温差判定电机或散热故障。',
              '控制日志约50 Hz，不能排除更高频率的振动；温度可能受此前动作和初始温度影响。',
              f"匹配质量：{json.dumps(quality,ensure_ascii=False)}", '',
              '分播放/保持状态的指标见 summary.json，目标/实际角度、速度、温度、力矩与增益见 aligned_ankles.csv。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(5, 2, figsize=(15, 15), sharex=True, constrained_layout=True)
    for c, pair in enumerate([(4,10),(5,11)]):
        for joint,color in zip(pair,['#2468b4','#d65d31']):
            k=ids.index(joint);label=JOINTS[joint]
            target=command[:,3+k];kp=command[:,11+k];kd=command[:,15+k]
            pd=kp*(target-q[:,k])+kd*(command[:,7+k]-arrays['dq'][:,k])+command[:,19+k]
            axs[0,c].plot(times_s,arrays['temperature'][:,2*k],color=color,label=label+' temp0')
            axs[0,c].plot(times_s,arrays['temperature'][:,2*k+1],color=color,ls='--',alpha=.7,label=label+' temp1')
            axs[1,c].plot(times_s,np.rad2deg(q[:,k]),color=color,label=label+' measured')
            axs[1,c].plot(times_s,np.rad2deg(target),color=color,ls='--',alpha=.65,label=label+' target')
            axs[2,c].plot(times_s,np.rad2deg(target-q[:,k]),color=color,label=label)
            axs[3,c].plot(times_s,arrays['torque'][:,k],color=color,label=label+' estimate')
            axs[3,c].plot(times_s,pd,color=color,ls='--',alpha=.5,label=label+' PD request')
            axs[4,c].plot(times_s,arrays['dq'][:,k],color=color,label=label)
        for row,ylabel in enumerate(['Temperature (C)','Angle (deg)','Target - actual (deg)','Torque estimate / request (Nm)','Velocity (rad/s)']):
            axs[row,c].set_ylabel(ylabel);axs[row,c].grid(alpha=.2);axs[row,c].legend(fontsize=7,loc='upper right')
        axs[0,c].set_title('Ankle roll' if c==0 else 'Ankle pitch');axs[-1,c].set_xlabel('Time (s)')
    fig.suptitle('H2 ankle comparison | '+directory.name)
    fig.savefig(out/'ankle_comparison.png',dpi=150);plt.close(fig)
    print('\n'.join(lines[:12]));print(f'报告：{out.resolve()}')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log_dir',type=Path)
    args=parser.parse_args()
    analyze(args.log_dir.resolve())


if __name__=='__main__':
    main()
