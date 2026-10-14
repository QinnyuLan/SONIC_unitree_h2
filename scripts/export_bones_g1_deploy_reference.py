#!/usr/bin/env python3
"""Export selected BONES G1 CSVs to typed, 50 Hz deployment references."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import tempfile

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / 'gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml'
BODY_INDEXES = [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]
FPS = 50


def joint_names():
    tree = ast.parse((ROOT / 'gear_sonic/envs/env_utils/joint_utils.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == 'G1_ISAACLab_ORDER' for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError('G1 IsaacLab joint order missing')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(path, values, headers):
    values = values.reshape(len(values), -1)
    if values.shape[1] != len(headers) or not np.isfinite(values).all():
        raise ValueError(f'Invalid CSV: {path}')
    np.savetxt(path, values, delimiter=',', header=','.join(headers), comments='', fmt='%.10f')


def angular_velocity(quats):
    rotations = Rotation.from_quat(quats[:, [1, 2, 3, 0]])
    result = np.empty((len(quats), 3))
    result[0] = (rotations[1] * rotations[0].inv()).as_rotvec() * FPS
    result[-1] = (rotations[-1] * rotations[-2].inv()).as_rotvec() * FPS
    result[1:-1] = (rotations[2:] * rotations[:-2].inv()).as_rotvec() * (FPS / 2)
    return result


def export_motion(record, source_fps, output, model, names):
    source = Path(record['source_csv'])
    if sha256(source) != record['sha256']:
        raise ValueError(f'Source hash changed: {source}')
    with source.open() as file:
        headers = file.readline().strip().split(',')
        raw = np.loadtxt(file, delimiter=',', ndmin=2)
    if raw.shape != (record['frames'], 36) or not np.isfinite(raw).all():
        raise ValueError(f'Invalid source shape or values: {source}')
    if not np.all(np.diff(raw[:, 0]) == 1):
        raise ValueError(f'Non-contiguous source frames: {source}')
    source_times = np.arange(len(raw)) / source_fps
    times = np.arange(int(np.ceil(source_times[-1] * FPS)) + 1) / FPS
    # Preserve the exact final source pose on a regular 50 Hz grid, holding
    # it for less than one deployment frame when durations are not divisible.
    sample_times = np.minimum(times, source_times[-1])
    interpolate = lambda x: np.column_stack([
        np.interp(sample_times, source_times, column) for column in x.T
    ])
    joints = interpolate(np.deg2rad(raw[:, [headers.index(n + '_dof') for n in names]]))
    positions = interpolate(raw[:, 1:4] / 100)
    roots = Slerp(source_times, Rotation.from_euler('xyz', raw[:, 4:7], degrees=True))(
        sample_times).as_quat()[:, [3, 0, 1, 2]]
    data = mujoco.MjData(model)
    addresses = [model.joint(n).qposadr[0] for n in names]
    bodies = [model.body('pelvis').id if i == 0 else
              int(model.joint(names[i-1]).bodyid[0]) for i in BODY_INDEXES]
    body_pos = np.empty((len(times), 14, 3))
    body_quat = np.empty((len(times), 14, 4))
    feet = [model.body(s + '_ankle_roll_link').id for s in ('left', 'right')]
    spheres = [np.flatnonzero((model.geom_bodyid == b) &
        (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)) for b in feet]
    sole_height, ankle_pos, torso_tilt = [], [], []
    for i in range(len(times)):
        data.qpos[:3] = positions[i]
        data.qpos[3:7] = roots[i]
        data.qpos[addresses] = joints[i]
        mujoco.mj_kinematics(model, data)
        body_pos[i] = data.xpos[bodies]
        body_quat[i] = data.xquat[bodies]
        sole_height.append([(data.geom_xpos[s, 2] - model.geom_size[s, 0]).min() for s in spheres])
        ankle_pos.append(data.xpos[feet].copy())
        torso_tilt.append(np.degrees(np.arccos(np.clip(data.xmat[model.body('torso_link').id, 8], -1, 1))))
    # q and -q encode the same pose; keep adjacent quaternion signs continuous.
    flips = np.where(np.sum(body_quat[1:] * body_quat[:-1], axis=2) < 0, -1, 1)
    body_quat[1:] *= np.cumprod(flips, axis=0)[:, :, None]
    joint_vel = np.diff(joints, axis=0) * FPS
    joint_vel = np.concatenate([joint_vel, joint_vel[-1:]], axis=0)
    body_lin_vel = np.gradient(body_pos, 1 / FPS, axis=0)
    body_ang_vel = np.stack([angular_velocity(body_quat[:, b]) for b in range(14)], axis=1)

    knee_angles = np.abs(np.rad2deg(joints[:, [names.index(s + '_knee_joint') for s in ('left', 'right')]]))
    root_tilt = np.degrees(np.arccos(np.clip(
        Rotation.from_quat(roots[:, [1, 2, 3, 0]]).as_matrix()[:, 2, 2], -1, 1)))
    standing = ((positions[:, 2] >= .70) & (root_tilt <= 20) &
        (knee_angles.max(axis=1) <= 35) & (np.linalg.norm(body_lin_vel[:, 0], axis=1) <= .25) &
        (np.abs(sole_height).max(axis=1) <= .035) & (np.asarray(torso_tilt) <= 25) &
        (np.linalg.norm(np.gradient(ankle_pos, 1 / FPS, axis=0), axis=2).max(axis=1) <= .25))
    if not (standing[:13].all() and standing[-13:].all()):
        raise ValueError(f'Resampled endpoints fail standing check: {source}')

    dest = output / record['filename']
    dest.mkdir()
    arrays = dict(joint_pos=joints, joint_vel=joint_vel, body_pos=body_pos,
                  body_quat=body_quat, body_lin_vel=body_lin_vel, body_ang_vel=body_ang_vel)
    for key, values in arrays.items():
        if key.startswith('joint'):
            labels = [f'{key}_{i}' for i in range(29)]
        else:
            axes = 'wxyz' if key == 'body_quat' else 'xyz'
            labels = [f'{key}_{b}_{axis}' for b in range(14) for axis in axes]
        save_csv(dest / (key + '.csv'), values, labels)
    metadata = [f'Metadata for: {record["filename"]}', '==============================', '',
        'Body part indexes:', '[' + ' '.join(map(str, BODY_INDEXES)) + ']', '',
        f'Total timesteps: {len(times)}', 'FPS: 50', 'Reference type: g1', 'Encoder mode: 3',
        'Joint order: G1_ISAACLab_ORDER (29 native G1 joints)',
        'Quaternion order: wxyz', 'Units: m, rad, m/s, rad/s',
        f'Source CSV: {source}', f'Source SHA256: {record["sha256"]}',
        f'Source sample rate: {source_fps:g} Hz', 'Endpoint standing check: passed (13 frames at each end)', '']
    (dest / 'metadata.txt').write_text('\n'.join(metadata))
    return dict(motion_name=record['filename'], reference_type='g1', encoder_mode=3,
        description=record['description_zh'], fps=FPS, frames=len(times),
        duration_s=float(times[-1]), source_csv=str(source), source_sha256=record['sha256'],
        source_fps=source_fps, source_frames=len(raw), source_duration_s=float(source_times[-1]),
        final_pose_hold_s=float(times[-1] - source_times[-1]),
        start_standing=True, end_standing=True,
        body_names=[model.body(b).name for b in bodies],
        output_sha256={p.name: sha256(p) for p in sorted(dest.iterdir())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=Path(
        '/home/qingyul/Datasets/bones_seed/sim2sim_selection_20260908/recommendations.json'))
    parser.add_argument('--output', type=Path, default=ROOT / 'gear_sonic_deploy/reference/example_g1_explore')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Output already exists: {args.output}')
    selection = json.loads(args.selection.read_text())
    model = mujoco.MjModel.from_xml_path(str(ASSET))
    names = joint_names()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.g1_export_', dir=args.output.parent) as tmp:
        stage = Path(tmp) / 'bundle'
        stage.mkdir()
        motions = [export_motion(r, selection['source_fps'], stage, model, names)
                   for r in selection['recommendations']]
        manifest = dict(schema='sonic_typed_reference_manifest_v1', reference_type='g1', fps=FPS,
            description='BONES-SEED G1 exploration demos with standing starts and ends.',
            source_selection=str(args.selection.resolve()), source_selection_sha256=sha256(args.selection),
            fk_asset=str(ASSET), fk_asset_sha256=sha256(ASSET), joint_names=names,
            body_part_indexes=BODY_INDEXES,
            resampling='120 to 50 Hz: linear joints/root position; SLERP root rotation; FK at 50 Hz; final pose hold < 20 ms',
            velocity_method='forward joint difference (last repeats); central world linear/angular differences',
            motions=motions)
        (stage / 'h2_lora_reference_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
        summary = [f'G1 exploration references: {len(motions)} motions, 50 Hz, reference_type=g1', '']
        summary += [f'{i}. {r["motion_name"]}: {r["frames"]} frames, {r["duration_s"]:.2f} s — {r["description"]}'
                    for i, r in enumerate(motions, 1)]
        (stage / 'motion_summary.txt').write_text('\n'.join(summary) + '\n')
        readme = ['# G1 exploration references', '',
            f'{len(motions)} 条 G1 原生参考动作。每条在清单与 metadata 中标记为 `g1`，对应通用编码器 selector 3。', '',
            '在 gear_sonic_deploy 目录运行部署程序时，设置：', '',
            '```text', '--motion-data reference/example_g1_explore', '```', '',
            '加载顺序：', '', *summary[2:], '',
            '数据为 50 Hz、29 维 G1 IsaacLab 关节顺序，包含与 reference/example 一致的 14 个身体部位。',
            '位置单位 m，角度 rad，四元数 wxyz，身体速度在世界坐标系。',
            '源 CSV 的 120 Hz 轨迹线性/SLERP 重采样；末尾最多保持不足 20 ms 以保留原始终点。',
            '重采样后首尾各 13 帧通过站姿、足底高度和速度检查。尚未进行策略闭环测试。',
            '需要支持 G1 参考分支的模型/配置；H2-only 编码器无法读取此参考类型。', '',
            '复现：在仓库根目录运行 python3 scripts/export_bones_g1_deploy_reference.py --output <新目录>。',
            '来源、帧数、关节与身体顺序、SHA-256 见 h2_lora_reference_manifest.json。', '']
        (stage / 'README.md').write_text('\n'.join(readme))
        stage.rename(args.output)
    print(f'Exported {len(motions)} native G1 motions to {args.output}')
    print(f'Total frames: {sum(r["frames"] for r in motions)}; all endpoints passed standing checks')


if __name__ == '__main__':
    main()
