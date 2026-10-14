#!/usr/bin/env python3
"""Offline full closed-chain validation of a motor candidate on ankle sweeps."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'external_dependencies/unitree_sdk2_python'))

import numpy as np

from analyze_h2_ankle_friction_hypothesis import load_trials, metrics, simulate
from analyze_h2_ankle_sysid import trial_specs
from sim_h2_ankle_sysid import FixtureEnv, make_config
from gear_sonic.utils.mujoco_sim.motor_response import (
    H2_MOTOR_GROUP_TRANSMISSIONS, MotorResponse,
)

LOGS = ROOT/'gear_sonic_deploy/logs'
SOURCES = {
    'left': LOGS/'ankle_dynamic_left_20260906_210301_532438',
    'right': LOGS/'ankle_dynamic_right_20260906_211438_540301',
}
IDENTIFICATION = ROOT/'gear_sonic/data/assets/robot_description/h2_ankle_sysid/identification.json'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_fingerprint(model):
    fields = ('eq_data', 'eq_solref', 'eq_solimp', 'dof_frictionloss', 'dof_damping',
              'dof_armature', 'dof_solref', 'dof_solimp', 'body_mass', 'body_inertia',
              'actuator_ctrlrange', 'geom_contype', 'geom_conaffinity')
    return {field: hashlib.sha256(np.asarray(getattr(model, field)).tobytes()).hexdigest()
            for field in fields}


def get_trials(source, use_cache=True):
    metadata = json.loads((source/'metadata.json').read_text())
    if metadata.get('status') != 'completed':
        raise ValueError('Expected completed hardware sweeps')
    specifications = trial_specs(metadata)
    files = [source/'friction_hard'/f"{s['amplitude_deg']:g}deg_{s['frequency_hz']:g}Hz.npz"
             for s in specifications]
    if not use_cache or not all(path.is_file() for path in files):
        meta, trials = load_trials(source)
        return meta, trials, dict(source='actual command/state CSV via load_trials')
    trials = []
    for spec, path in zip(specifications, files):
        with np.load(path, allow_pickle=False) as a:
            command, real = a['commands'].copy(), a['real'].copy()
        if command.ndim != 2 or command.shape[1] != 158 or real.ndim != 2 or real.shape[1] != 3:
            raise ValueError(f'Unexpected cached raw-input shapes: {path}')
        if (not np.isfinite(command).all() or not np.isfinite(real).all()
                or np.any(np.diff(command[:, 0]) <= 0) or np.any(np.diff(real[:, 0]) <= 0)
                or not np.all(command[:, 2] == spec['frequency_hz'])
                or not np.any(command[:, 1] == 1) or command[0, 0] != 0):
            raise ValueError(f'Invalid cached command/state data: {path}')
        trials.append(dict(spec=spec, commands=command, states=real))
    return metadata, trials, dict(source='raw commands/real arrays cached by original replay analysis; predicted array ignored',
                                  files={str(p): sha256(p) for p in files})


def compare(trial, real, prediction, active):
    begin = trial['commands'][trial['commands'][:, 1] == 1, 0][0]
    end = prediction[-1, 0]
    measured = real[(real[:, 0] >= begin) & (real[:, 0] <= end)]
    predicted = prediction[(prediction[:, 0] >= measured[0, 0]) & (prediction[:, 0] <= measured[-1, 0])]
    measured = measured[(measured[:, 0] >= predicted[0, 0]) & (measured[:, 0] <= predicted[-1, 0])]
    if len(predicted) < 30 or len(measured) < 30:
        raise ValueError('Insufficient overlapping steady samples')
    actual_metrics = metrics(trial, measured, active)
    predicted_metrics = metrics(trial, predicted, active)
    times = predicted[:, 0]
    real_q = np.interp(times, real[:, 0], real[:, 1])
    real_dq = np.interp(times, real[:, 0], real[:, 2])
    q = predicted[:, 1]
    dq = predicted[:, 2]
    ac_error = (q-q.mean())-(real_q-real_q.mean())
    phase = (predicted_metrics['phase_deg']-actual_metrics['phase_deg']+180) % 360-180
    phase_reliable = (actual_metrics['gain'] >= .1 and
                      actual_metrics['peak_to_peak_deg'] > 6*actual_metrics['residual_deg'])
    return dict(real=actual_metrics, predicted=predicted_metrics,
                interval_begin_s=float(times[0]), interval_end_s=float(times[-1]),
                cycles_measured=float((times[-1]-times[0])*trial['spec']['frequency_hz']),
                ac_q_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(ac_error**2)))),
                raw_q_rmse_deg=float(np.rad2deg(np.sqrt(np.mean((q-real_q)**2)))),
                dq_rmse_deg_s=float(np.rad2deg(np.sqrt(np.mean((dq-real_dq)**2)))),
                mean_error_deg=float(np.rad2deg(q.mean()-real_q.mean())),
                gain_absolute_error=abs(predicted_metrics['gain']-actual_metrics['gain']),
                phase_reliable=phase_reliable,
                phase_absolute_error_deg=abs(phase) if phase_reliable else None)


def run_prediction(env, initial, trial, active, parameters, groups, cycles):
    original_mapping = env._map_body_torques_to_actuators
    response = MotorResponse(env.body_torque_limit, groups, parameters)
    response.reset()
    selected = np.flatnonzero(response.selected)
    diagnostics = dict(filtered_motor_positions=selected.tolist(), selected_clipped_steps=0,
                       calls=0, max_abs_raw_effort=0., max_abs_applied_effort=0.)

    def mapping(tau):
        raw = original_mapping(tau)
        applied = response.apply(raw, env.sim_dt)
        diagnostics['calls'] += 1
        diagnostics['selected_clipped_steps'] += int(np.any(response.clipped))
        if len(selected):
            diagnostics['max_abs_raw_effort'] = max(diagnostics['max_abs_raw_effort'], float(np.max(np.abs(raw[selected]))))
            diagnostics['max_abs_applied_effort'] = max(diagnostics['max_abs_applied_effort'], float(np.max(np.abs(applied[selected]))))
        return applied

    env._map_body_torques_to_actuators = mapping
    try:
        prediction = simulate(env, initial, trial, active, cycles=cycles)
    finally:
        env._map_body_torques_to_actuators = original_mapping
    if diagnostics['calls'] != len(prediction):
        raise ValueError('Motor response did not execute exactly once per physics step')
    if abs(env.mj_data.time - len(prediction)*env.sim_dt) > 1e-7:
        raise ValueError('Simulation time reset or skipped steps')
    if any(w.number for w in env.mj_data.warning):
        raise ValueError('MuJoCo warning during replay')
    return prediction, diagnostics


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--side', choices=('left', 'right', 'both'), default='both')
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--read-original-csv', action='store_true')
    args = parser.parse_args(argv)
    if not 2 <= args.cycles <= 6:
        parser.error('--cycles must be between 2 and 6')
    args.output.mkdir(parents=True, exist_ok=False)
    candidate = json.loads(args.candidate.read_text())
    parameters = candidate['config']
    ankle = json.loads(IDENTIFICATION.read_text())
    if sha256(Path(ankle['source'])) != ankle['source_sha256']:
        raise ValueError('Official model changed since original ankle identification')
    report = dict(status='running', candidate=candidate, candidate_sha256=sha256(args.candidate),
        script_sha256=sha256(Path(__file__)), ankle_identification=ankle,
        ankle_identification_sha256=sha256(IDENTIFICATION), cycles=args.cycles,
        input='actual q/dq/kp/kd/tau_ff command ZOH; no encoder reference; no measured state feedback injection',
        fixture='complete official closed-chain model; pelvis weld, no contact, other joints PD-held',
        response='MotorResponse on actual motor group AFTER closed-chain torque mapping, reset each independent trial; original ramp retained',
        baseline='existing selected bilateral ankle bristle parameters; no additional motor response',
        candidate_scope='conditional fitted left-knee effective response, not calibrated hardware parameters',
        caveat='Two steady cycles only; same source session already used for ankle model development, not a blind holdout',
        sources={}, sides={}, trials=[])
    save(args.output/'report.json', report)
    rows = []
    started = time.monotonic()
    sides = ('left', 'right') if args.side == 'both' else (args.side,)
    for side in sides:
        source = SOURCES[side]
        metadata, trials, provenance = get_trials(source, not args.read_original_csv)
        report['sources'][side] = provenance
        active = metadata['active_index']
        config = make_config(73, source/'metadata.json')
        config.update(ANKLE_PITCH_RESPONSE=ankle['parameters'], H2_MOTOR_RESPONSE={},
                      PRINT_SCENE_INFORMATION=False, DIAGNOSTICS_DIR=None)
        if config.get('FIXTURE_LOCK_NON_ANKLE'):
            raise ValueError('Knees must remain PD-held and dynamic, not constrained')
        env = FixtureEnv(config, onscreen=False)
        if env.ankle_response is None or env.motor_response is not None:
            raise ValueError('Unexpected ankle/native motor-response initialization')
        initial = env.mj_data.qpos.copy()
        model_hash = model_fingerprint(env.mj_model)
        transmissions = {g['name']: g for g in env.closed_loop_transmissions}
        groups = {name: transmissions[transmission]['motor_positions'].tolist()
                  for name, transmission in H2_MOTOR_GROUP_TRANSMISSIONS.items()}
        report['sides'][side] = dict(physics_timestep_s=env.sim_dt,
            equality_count=env.mj_model.neq, model_fingerprint=model_hash,
            motor_groups=groups, body_joint_order=env.body_joint_names,
            nominal_motor_effort_limits=env.body_torque_limit.tolist(),
            initial_qpos=initial.tolist(), config=config)
        for number, trial in enumerate(trials, 1):
            name = f"{side}_{trial['spec']['amplitude_deg']:g}deg_{trial['spec']['frequency_hz']:g}Hz"
            trial_report = dict(side=side, name=name, spec=trial['spec'], variants={})
            trajectories = dict(commands=trial['commands'], real=trial['states'])
            for variant, settings in [('baseline', {}), ('candidate', parameters)]:
                prediction, diagnostics = run_prediction(env, initial, trial, active, settings, groups, args.cycles)
                measurement = compare(trial, trial['states'], prediction, active)
                measurement['runtime'] = diagnostics
                trial_report['variants'][variant] = measurement
                trajectories[variant] = prediction
                row = dict(side=side, trial=name, variant=variant, **trial['spec'])
                row.update({k: v for k, v in measurement.items() if k not in ('real', 'predicted', 'runtime')})
                row.update({f'{kind}_{key}': value for kind in ('real', 'predicted') for key, value in measurement[kind].items()})
                rows.append(row)
                if model_fingerprint(env.mj_model) != model_hash:
                    raise ValueError('Native equality/passive/actuator/contact properties changed')
            report['trials'].append(trial_report)
            np.savez_compressed(args.output/(name+'.npz'), **trajectories)
            save(args.output/'report.json', report)
            before = trial_report['variants']['baseline']['ac_q_rmse_deg']
            after = trial_report['variants']['candidate']['ac_q_rmse_deg']
            print(f'{side} {number}/{len(trials)} {name}: AC RMSE {before:.4f} -> {after:.4f} deg; elapsed {time.monotonic()-started:.1f}s', flush=True)
        del env
    with (args.output/'trials.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for side in sides:
        summary[side] = {}
        for variant in ('baseline', 'candidate'):
            selected = [r for r in rows if r['side'] == side and r['variant'] == variant]
            summary[side][variant] = dict(conditions=len(selected),
                median_ac_q_rmse_deg=float(np.median([r['ac_q_rmse_deg'] for r in selected])),
                max_ac_q_rmse_deg=float(max(r['ac_q_rmse_deg'] for r in selected)),
                gain_mae=float(np.mean([r['gain_absolute_error'] for r in selected])),
                max_absolute_mean_error_deg=float(max(abs(r['mean_error_deg']) for r in selected)),
                reliable_phase_mae_deg=float(np.mean([r['phase_absolute_error_deg'] for r in selected if r['phase_reliable']])),
                reliable_phase_conditions=sum(r['phase_reliable'] for r in selected))
        deltas = [t['variants']['candidate']['ac_q_rmse_deg'] - t['variants']['baseline']['ac_q_rmse_deg']
                  for t in report['trials'] if t['side'] == side]
        summary[side]['max_ac_rmse_increase_deg'] = float(max(deltas))
        summary[side]['regressions_over_0p1_deg'] = sum(d > .1 for d in deltas)
    report.update(status='completed', elapsed_wall_s=time.monotonic()-started, summary=summary)
    save(args.output/'report.json', report)
    save(args.output/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
