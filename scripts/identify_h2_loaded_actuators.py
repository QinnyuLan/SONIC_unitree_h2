#!/usr/bin/env python3
"""Offline, conditional short-window identification; never initializes DDS.

Missing root translation/contact state is an explicit nuisance hypothesis.
Recorded commands are inputs, not encoder references or replayed measured forces.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'external_dependencies/unitree_sdk2_python'))

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
import yaml

from analyze_h2_squat_comparison import columns, load_deploy
from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import H2_JOINT_NAMES_DDS
from gear_sonic.utils.mujoco_sim.motor_response import MotorResponse

LOGS = ROOT / 'gear_sonic_deploy/logs/squat_compare'
CONFIG = ROOT / 'gear_sonic/utils/mujoco_sim/wbc_configs/h2_native_sonic_unitree_ros_loop.yaml'
AXES = [5, 11, 3, 9]
GROUPS = ('left_ankle', 'right_ankle', 'left_knee', 'right_knee')


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def csv_rows(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(run, output, duration=.25):
    source = LOGS / run
    deploy = load_deploy(source)
    begin = deploy['t'][deploy['groups'][0][0]] - 2000
    end = deploy['t'][deploy['groups'][-1][-1]] + 1000
    snames = ['time_monotonic_ms', 'mode_pr', 'crc_ok']
    snames += [f'{p}_{i}' for p in ('q', 'dq') for i in range(31)]
    snames += [f'imu_quat_wxyz_{a}' for a in 'wxyz'] + [f'imu_gyro_{a}' for a in 'xyz']
    cnames = ['time_monotonic_ms', 'mode_pr', 'crc_ok']
    cnames += [f'{p}_{i}' for p in ('q_target', 'dq_target', 'kp', 'kd', 'tau_ff') for i in range(31)]
    state = columns(source / 'dds/lowstate.csv', snames, stop_ms=end)
    command = columns(source / 'dds/lowcmd.csv', cnames, stop_ms=end)
    state = state[(state[:, 0] >= begin) & (state[:, 0] <= end)]
    command = command[(command[:, 0] >= begin) & (command[:, 0] <= end)]
    for a in (state, command):
        if not np.isfinite(a).all() or np.any(np.diff(a[:, 0]) < 0) or np.any(a[:, 1] != 0):
            raise ValueError('invalid/non-PR/nonmonotonic recorded samples')
        if run == 'real01' and np.any(a[:, 2] != 1):
            raise ValueError('hardware CRC failure')
    start_list, repeats, labels = [], [], []
    for repeat, indices in enumerate(deploy['groups'], 1):
        knee = np.rad2deg(deploy['q'][indices][:, [3, 9]].mean(axis=1))
        peaks, _ = find_peaks(knee, prominence=30, distance=40)
        if len(peaks) != 3:
            raise ValueError('expected three squat peaks per playback')
        times = [deploy['t'][indices[0]] - 300]
        times += [deploy['t'][indices[p]] for p in peaks]
        for cycle, t in enumerate(times):
            start_list.append(float(state[np.searchsorted(state[:, 0], t), 0]))
            repeats.append(repeat)
            labels.append(f'rep{repeat}_' + ('settled' if cycle == 0 else f'rise{cycle}'))
    origin = state[0, 0]
    state[:, 0] = (state[:, 0] - origin) / 1000
    command[:, 0] = (command[:, 0] - origin) / 1000
    starts = (np.asarray(start_list) - origin) / 1000
    np.savez_compressed(output / 'dataset.npz', state=state, command=command,
                        starts=starts, repeats=repeats, labels=labels, duration=duration)
    save(output / 'dataset.json', dict(source=str(source), duration_s=duration,
        windows=labels, train_repetitions=[1, 2], validation_repetitions=[3],
        split='within-session repetition holdout, not independent/blind validation',
        input='actual DDS LowCmd, causal receive-time ZOH; q,dq,kp,kd,tau unchanged',
        unavailable=['root position/linear velocity', 'contact forces', 'motor torque calibration'],
        origin_monotonic_ms=origin,
        hashes={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (source/'dds/lowcmd.csv', source/'dds/lowstate.csv', CONFIG)}))


def make_env():
    config = yaml.safe_load(CONFIG.read_text())
    config.update(ENABLE_ELASTIC_BAND=False, PRINT_SCENE_INFORMATION=False,
                  DIAGNOSTICS_DIR=None, INTERFACE=None, USE_JOYSTICK=False)
    config['ANKLE_PITCH_RESPONSE'] = json.loads((ROOT /
        'gear_sonic/data/assets/robot_description/h2_ankle_sysid/identification.json').read_text())['parameters']
    return DefaultEnv(config, onscreen=False)


def motor_groups(env):
    return {name: next(g['motor_positions'].tolist() for g in env.closed_loop_transmissions
                       if g['name'] == name + '_loop') for name in GROUPS}


def sole_points(env):
    m, d = env.mj_model, env.mj_data
    result = []
    for side in ('left', 'right'):
        body = m.body(side + '_ankle_pitch_link').id
        points = []
        for g in np.flatnonzero(m.geom_bodyid == body):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = m.geom_dataid[g]
                a, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
                vertices = m.mesh_vert[a:a+n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                points.append(vertices[np.argmin(vertices[:, 2])])
        if not points:
            raise ValueError('expected official foot collision meshes')
        result.append((body, min(points, key=lambda v: v[2]).copy()))
    return result


def initialize(env, q_dds, dq_dds, quat, gyro, *, velocity='both', clearance=0.,
               root_position=None, root_velocity=None):
    """One initialization only; hidden linkage states satisfy equality constraints."""
    m, d = env.mj_model, env.mj_data
    if not hasattr(env, '_loaded_replay_pose_seed'):
        env._loaded_replay_pose_seed = d.qpos.copy()
    mujoco.mj_resetData(m, d)
    d.qpos[:] = env._loaded_replay_pose_seed
    env._closed_loop_jacobian_cache_time = None
    if env.ankle_response is not None:
        env.ankle_response.reset()
    order = np.array([H2_JOINT_NAMES_DDS.index(n) for n in env.body_joint_names])
    q, dq = np.asarray(q_dds)[order], np.asarray(dq_dds)[order]
    d.qpos[:3] = [0, 0, 1]
    d.qpos[3:7] = np.asarray(quat) / np.linalg.norm(quat)
    # Continue from the official ready assembly branch, never solve a deep
    # squat in one jump from an arbitrary zeroed linkage configuration.
    previous = d.qpos[env.body_qpos_index].copy()
    closed = set(int(i) for g in env.closed_loop_transmissions for i in g['output_positions'])
    direct = np.array([i for i in range(len(q)) if i not in closed])
    steps = max(1, int(np.ceil(np.max(np.abs(q-previous))/.15)))
    for alpha in np.linspace(0, 1, steps+1)[1:]:
        target = previous + alpha*(q-previous)
        d.qpos[env.body_qpos_index[direct]] = target[direct]
        mujoco.mj_forward(m, d)
        env._project_closed_loop_default_pose(target)
    if root_position is not None:
        d.qpos[:3] = root_position
    else:
        points = sole_points(env)
        d.qpos[2] += clearance - min(v[2] for _, v in points)
    mujoco.mj_forward(m, d)
    d.qvel[:] = 0
    d.qvel[env.body_dof_index] = dq
    d.qvel[3:6] = gyro
    rows = np.flatnonzero(d.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
    jac = env._dense_constraint_jacobian()[rows]
    known = np.r_[np.arange(6), env.body_dof_index]
    unknown = np.setdiff1d(np.arange(m.nv), known)
    d.qvel[unknown] = np.linalg.lstsq(jac[:, unknown], -jac[:, known] @ d.qvel[known], rcond=None)[0]
    points = sole_points(env)
    jp = []
    for body, point in points:
        a = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, a, None, point, body)
        jp.append(a)
    if root_velocity is not None:
        d.qvel[:3] = root_velocity
    elif velocity != 'zero':
        chosen = jp if velocity == 'both' else [jp[0 if velocity == 'left' else 1]]
        a = np.vstack(chosen)
        d.qvel[:3] = np.linalg.lstsq(a[:, :3], -a[:, 3:] @ d.qvel[3:], rcond=None)[0]
    eq_error = float(np.max(np.abs(d.efc_pos[rows])))
    veq_error = float(np.linalg.norm(jac @ d.qvel))
    if eq_error > 1e-7 or veq_error > 1e-6:
        raise ValueError(f'invalid reconstruction equality q={eq_error}, dq={veq_error}')
    np.testing.assert_allclose(d.qpos[env.body_qpos_index], q, atol=1e-8, rtol=0)
    np.testing.assert_allclose(d.qvel[env.body_dof_index], dq, atol=1e-10, rtol=0)
    mujoco.mj_forward(m, d)
    env._closed_loop_jacobian_cache_time = None
    return dict(equality_q_error=eq_error, equality_dq_error=veq_error,
        root_z=float(d.qpos[2]), root_velocity=d.qvel[:3].tolist(),
        foot_bottom_z=[float(v[2]) for _, v in points],
        assumed_contact_point_speed=[float(np.linalg.norm(a @ d.qvel)) for a in jp],
        initial_contacts=int(d.ncon))


class Replay:
    def __init__(self, dataset):
        self.a = np.load(dataset)
        self.env = make_env()
        self.order = np.array([H2_JOINT_NAMES_DDS.index(n) for n in self.env.body_joint_names])
        self.to_dds = np.argsort(self.order)
        self.groups = motor_groups(self.env)
        self.initial_cache = {}
        self.windows = []
        s, c = self.a['state'], self.a['command']
        duration = float(self.a['duration'])
        for start, repeat, label in zip(self.a['starts'], self.a['repeats'], self.a['labels']):
            times = start + np.arange(round(duration / self.env.sim_dt)) * self.env.sim_dt
            indices = np.searchsorted(c[:, 0], times, side='right') - 1
            if indices.min() < 0:
                raise ValueError('no causal command at window start')
            state = s[np.searchsorted(s[:, 0], start)]
            target = c[indices, 3:].reshape(-1, 5, 31)[:, :, self.order]
            measured = np.column_stack([np.interp(times[::5], s[:, 0], s[:, k]) for k in range(3, 65)])
            quats = np.column_stack([np.interp(times[::5], s[:, 0], s[:, k]) for k in range(65, 69)])
            pitch = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_euler('xyz')[:, 1]
            self.windows.append(dict(label=str(label), repeat=int(repeat), start=float(start),
                initial=state, target=target, measured=measured, pitch=pitch, times=times-times[0]))

    def run(self, config, window, velocity='both', clearance=0.):
        env, w = self.env, window
        m, d = env.mj_model, env.mj_data
        initial = w['initial']
        key = (w['label'], velocity, clearance)
        if key not in self.initial_cache:
            init = initialize(env, initial[3:34], initial[34:65], initial[65:69], initial[69:72],
                              velocity=velocity, clearance=clearance)
            self.initial_cache[key] = (d.qpos.copy(), d.qvel.copy(), init)
        else:
            mujoco.mj_resetData(m, d)
            d.qpos[:], d.qvel[:], init = self.initial_cache[key]
            if env.ankle_response is not None:
                env.ankle_response.reset()
            mujoco.mj_forward(m, d)
            env._closed_loop_jacobian_cache_time = None
        model = MotorResponse(env.body_torque_limit, self.groups, config)
        qdes, dqdes, kp, kd, ff = w['target'][0]
        raw = env._map_body_torques_to_actuators(ff+kp*(qdes-d.qpos[env.body_qpos_index])+kd*(dqdes-d.qvel[env.body_dof_index]))
        model.reset(model.gain * raw)
        output, effort = [], []
        for k, (qdes, dqdes, kp, kd, ff) in enumerate(w['target']):
            if k % 5 == 0:
                pitch = Rotation.from_quat(d.qpos[[4, 5, 6, 3]]).as_euler('xyz')[1]
                output.append(np.r_[d.qpos[env.body_qpos_index][self.to_dds],
                                     d.qvel[env.body_dof_index][self.to_dds], pitch])
            pd = ff+kp*(qdes-d.qpos[env.body_qpos_index])+kd*(dqdes-d.qvel[env.body_dof_index])
            raw = env._map_body_torques_to_actuators(pd)
            applied = model.apply(raw, env.sim_dt)
            env.torques.fill(0.)
            env.torques[env.body_actuator_index] = applied
            d.ctrl[:] = np.clip(env.torques, -env.torque_limit, env.torque_limit)
            if env.ankle_response is not None:
                env.ankle_response.apply(m, d)
            if k % 5 == 0:
                effort.append(np.r_[raw[self.to_dds], applied[self.to_dds]])
            mujoco.mj_step(m, d)
            if (abs(d.time-(k+1)*env.sim_dt) > 1e-8 or
                    any(warning.number for warning in d.warning)):
                raise ValueError(f'{w["label"]}: simulation warning/reset at step {k}; invalid replay')
        prediction = np.asarray(output)
        if not np.isfinite(prediction).all():
            raise ValueError('nonfinite replay')
        qerr = prediction[:, AXES] - w['measured'][:, AXES]
        verr = prediction[:, np.array(AXES)+31] - w['measured'][:, np.array(AXES)+31]
        perr = prediction[:, 62] - w['pitch']
        residual = np.r_[qerr.ravel(), .035*verr.ravel(), .4*perr] / np.deg2rad(5)
        row = dict(window=w['label'], repeat=w['repeat'], velocity=velocity, clearance=clearance,
                   score=float(np.mean(residual**2)), pitch_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(perr**2)))))
        for j, name in enumerate(GROUPS):
            row[name+'_q_rmse_deg'] = float(np.rad2deg(np.sqrt(np.mean(qerr[:, j]**2))))
            row[name+'_dq_rmse_deg_s'] = float(np.rad2deg(np.sqrt(np.mean(verr[:, j]**2))))
        return row, residual, prediction, init, np.asarray(effort)

    def evaluate(self, config, *, indices=None, velocity='both', clearance=0.):
        rows, residuals = [], []
        for i in (range(len(self.windows)) if indices is None else indices):
            row, residual, *_ = self.run(config, self.windows[i], velocity, clearance)
            rows.append(row)
            residuals.append(residual)
        return rows, np.concatenate(residuals)


def candidates():
    yield 'nominal', {}
    for group in GROUPS:
        for key, values in [('gain', [.7, .85, 1.1]), ('time_constant_ms', [5., 15., 30.]),
                            ('limit_scale', [.6])]:
            for value in values:
                yield f'{group}_{key}_{value:g}', {group: {key: value}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', default='real01')
    parser.add_argument('--stage', choices=['prepare', 'screen', 'fit', 'nuisance'], default='prepare')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--candidate', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.stage == 'prepare':
        prepare(args.run, args.output)
        return
    replay = Replay(args.dataset)
    start = time.monotonic()
    if args.stage == 'screen':
        rows, records = [], []
        for name, config in candidates():
            result, _ = replay.evaluate(config)
            train = np.mean([r['score'] for r in result if r['repeat'] < 3])
            validation = np.mean([r['score'] for r in result if r['repeat'] == 3])
            records.append(dict(name=name, config=config, train_score=float(train), validation_score=float(validation)))
            rows += [dict(candidate=name, **r) for r in result]
            print(name, 'train', round(train, 5), 'validation', round(validation, 5),
                  'elapsed', round(time.monotonic()-start, 1), flush=True)
            save(args.output/'progress.json', records)
        csv_rows(args.output/'windows.csv', rows)
        save(args.output/'screen.json', sorted(records, key=lambda r: r['train_score']))
    elif args.stage == 'fit':
        initial = json.loads(args.candidate.read_text())
        if isinstance(initial, list):
            initial = min(initial, key=lambda r: r['train_score'])
        config = initial.get('config', initial)
        if len(config) != 1:
            raise ValueError('refine a single screened group first')
        group = next(iter(config))
        x0 = [config[group].get('gain', 1.), config[group].get('time_constant_ms', 5.)]
        train = [i for i, w in enumerate(replay.windows) if w['repeat'] < 3]
        history = []
        def fun(x):
            candidate = {group: dict(gain=float(x[0]), time_constant_ms=float(x[1]))}
            _, r = replay.evaluate(candidate, indices=train)
            history.append(dict(config=candidate, train_score=float(np.mean(r*r))))
            save(args.output/'progress.json', history)
            print('fit', len(history), history[-1], flush=True)
            return r
        result = least_squares(fun, x0, bounds=([.55, 0], [1.15, 35]),
                               diff_step=.05, max_nfev=8, ftol=.005, xtol=.005)
        best = min(history, key=lambda r: r['train_score'])
        rows, _ = replay.evaluate(best['config'])
        best.update(validation_score=float(np.mean([r['score'] for r in rows if r['repeat'] == 3])),
                    optimizer_success=bool(result.success), optimizer_message=result.message,
                    scope='conditional short-window candidate, not validated hardware parameters')
        save(args.output/'candidate.json', best)
        csv_rows(args.output/'windows.csv', rows)
    else:
        record = json.loads(args.candidate.read_text())
        config = record.get('config', record)
        rows, initializations, traces = [], [], {}
        for velocity, clearance in [('both', 0.), ('zero', 0.), ('left', 0.), ('right', 0.),
                                    ('both', -.001), ('both', .002)]:
            for name, candidate in [('nominal', {}), ('candidate', config)]:
                for window in replay.windows:
                    row, _, prediction, init, effort = replay.run(candidate, window, velocity, clearance)
                    rows.append(dict(candidate=name, **row))
                    initializations.append(dict(candidate=name, window=window['label'],
                                               velocity=velocity, clearance=clearance, **init))
                    if velocity == 'both' and clearance == 0:
                        traces[name+'_'+window['label']] = prediction
                        traces[name+'_'+window['label']+'_effort'] = effort
            print('nuisance', velocity, clearance, 'elapsed', round(time.monotonic()-start, 1), flush=True)
        csv_rows(args.output/'windows.csv', rows)
        save(args.output/'initializations.json', initializations)
        np.savez_compressed(args.output/'traces.npz', **traces)
    save(args.output/'provenance.json', dict(dataset=str(args.dataset.resolve()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        elapsed_s=time.monotonic()-start, timestep_s=replay.env.sim_dt,
        nuisance='root height from minimum sole vertex; root velocity from no-slip assumption; zero bristle initial state',
        motor_initial_state='steady initial requested motor effort; not a measured torque state',
        safety='offline MuJoCo only, no DDS or real robot commands'))


if __name__ == '__main__':
    main()
