#!/usr/bin/env python3
"""Export H2LoRA MotionLib PKLs into C++ deploy reference CSVs.

The H2LoRA deploy encoder defaults to SONIC's joint-reference branch.  The
reference files therefore need to contain the same 29D G1-semantic joint space
used by training observations, while still being generated from H2 retargeted
MotionLib data.  This script performs that conversion and writes the CSV layout
consumed by ``MotionDataReader`` in ``gear_sonic_deploy``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import joblib
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from gear_sonic.envs.env_utils import h2lora_alignment, joint_utils  # noqa: E402

DEFAULT_ROBOT_ROOT = Path("/mnt/data/seed/h2lora_motion_lib/seed_5m_balanced_mixed_50hz_schema6/robot")
DEFAULT_SMPL_ROOT = Path("/mnt/data/seed/h2lora_motion_lib/seed_5m_balanced_mixed_50hz_schema6/smpl")
DEFAULT_OUTPUT_DIR = Path("gear_sonic_deploy/reference/h2_lora_seed5m_schema6")
DEFAULT_H2LORA_WBC_CONFIG = Path("gear_sonic/utils/mujoco_sim/wbc_configs/h2_lora_sonic.yaml")
MANIFEST_NAME = "h2_lora_reference_manifest.json"

H2_MJCF_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_pitch_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _unwrap_motion_pkl(path: Path) -> tuple[str, dict[str, Any]]:
    data = joblib.load(path)
    if isinstance(data, dict) and len(data) == 1:
        key, value = next(iter(data.items()))
        if isinstance(value, dict):
            return str(key), value
    if isinstance(data, dict):
        return path.stem, data
    raise TypeError(f"Unsupported motion PKL payload in {path}")


def _finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(arr, dtype=np.float64)
    if arr.shape[0] <= 1:
        return out.astype(np.float32)
    dt = 1.0 / float(fps)
    out[0] = (arr[1] - arr[0]) / dt
    out[-1] = (arr[-1] - arr[-2]) / dt
    if arr.shape[0] > 2:
        out[1:-1] = (arr[2:] - arr[:-2]) / (2.0 * dt)
    return out.astype(np.float32)


def _motionlib_forward_dof_velocity(values: np.ndarray, fps: float) -> np.ndarray:
    """Match MotionLib's robot dof_vels generation for deploy references."""
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(arr, dtype=np.float64)
    if arr.shape[0] <= 1:
        return out.astype(np.float32)
    dt = 1.0 / float(fps)
    out[:-1] = (arr[1:] - arr[:-1]) / dt
    out[-1] = out[-2]
    return out.astype(np.float32)


def _normalize_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.where(norm > 1e-12, norm, 1.0)
    return (quat / norm).astype(np.float32)


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_xyzw(quat_xyzw)
    return quat[:, [3, 0, 1, 2]].astype(np.float32)


def _axis_angle_to_wxyz(rotvec: np.ndarray) -> np.ndarray:
    arr = np.asarray(rotvec, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"axis-angle root pose must have shape (T, 3), got {arr.shape}")
    quat_xyzw = Rotation.from_rotvec(arr).as_quat()
    return _xyzw_to_wxyz(quat_xyzw)


def _smpl_root_yup_to_runtime_wxyz(root_quat_wxyz: np.ndarray) -> np.ndarray:
    """Match TrackingCommand.smpl_root_quat_w for smpl_y_up=True."""
    root_xyzw = np.asarray(root_quat_wxyz, dtype=np.float64)[:, [1, 2, 3, 0]]
    yup_to_zup = Rotation.from_rotvec(np.asarray([np.pi / 2.0, 0.0, 0.0]))
    smpl_base_inv = Rotation.from_quat(np.asarray([-0.5, -0.5, -0.5, 0.5], dtype=np.float64))
    runtime_xyzw = (yup_to_zup * Rotation.from_quat(root_xyzw) * smpl_base_inv).as_quat()
    return _xyzw_to_wxyz(runtime_xyzw)


def _angular_velocity_from_wxyz(quat_wxyz: np.ndarray, fps: float) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    out = np.zeros((quat.shape[0], 3), dtype=np.float64)
    if quat.shape[0] <= 1:
        return out.astype(np.float32)

    rot = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
    dt = 1.0 / float(fps)
    for i in range(quat.shape[0]):
        if i == 0:
            rel = rot[1] * rot[0].inv()
            denom = dt
        elif i == quat.shape[0] - 1:
            rel = rot[-1] * rot[-2].inv()
            denom = dt
        else:
            rel = rot[i + 1] * rot[i - 1].inv()
            denom = 2.0 * dt
        out[i] = rel.as_rotvec() / denom
    return out.astype(np.float32)


def _csv_headers(prefix: str, width: int) -> list[str]:
    return [f"{prefix}_{i}" for i in range(width)]


def _csv_headers_3d(prefix: str, count: int) -> list[str]:
    axes = ("x", "y", "z")
    return [f"{prefix}_{i}_{axis}" for i in range(count) for axis in axes]


def _csv_headers_quat(prefix: str, count: int) -> list[str]:
    axes = ("w", "x", "y", "z")
    return [f"{prefix}_{i}_{axis}" for i in range(count) for axis in axes]


def _write_csv(path: Path, data: np.ndarray, headers: list[str]) -> None:
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"{path} expects a 2D array, got shape {arr.shape}")
    if arr.shape[1] != len(headers):
        raise ValueError(f"{path} header count {len(headers)} does not match width {arr.shape[1]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in arr:
            writer.writerow([f"{float(v):.9f}" for v in row])


def _sanitize_motion_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", name).strip("_")


def _build_stem_index(root: Path) -> dict[str, Path]:
    return {path.stem: path for path in sorted(root.rglob("*.pkl"))}


def _resolve_robot_paths(args: argparse.Namespace) -> list[Path]:
    motion_root = Path(args.motion_root)
    if not motion_root.exists():
        raise FileNotFoundError(f"Motion root does not exist: {motion_root}")

    stem_index = _build_stem_index(motion_root)
    selected: list[Path] = []
    if args.motion_key:
        missing: list[str] = []
        for key in args.motion_key:
            path = stem_index.get(key)
            if path is None:
                missing.append(key)
            else:
                selected.append(path)
        if missing:
            raise FileNotFoundError(f"Motion keys not found under {motion_root}: {missing}")
    elif args.include_regex:
        pattern = re.compile(args.include_regex)
        selected = [path for stem, path in stem_index.items() if pattern.search(stem)]
    elif args.all:
        selected = list(stem_index.values())
    else:
        raise ValueError("Select motions with --motion-key, --include-regex, or --all.")

    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("No motions selected for export.")
    return selected


def _resolve_smpl_path(robot_path: Path, motion_root: Path, smpl_root: Path | None) -> Path | None:
    if smpl_root is None or not smpl_root.exists():
        return None
    rel = robot_path.relative_to(motion_root)
    direct = smpl_root / rel
    if direct.exists():
        return direct
    matches = sorted(smpl_root.rglob(robot_path.name))
    return matches[0] if matches else None


def _h2_physical_to_g1_semantic(dof: np.ndarray, joint_names: list[str]) -> np.ndarray:
    missing = [name for name in joint_utils.H2LORA_BODY_JOINTS_IN_G1_ORDER if name not in joint_names]
    if missing:
        raise ValueError(f"H2 motion is missing joints required by H2LoRA: {missing}")
    body_indices = [joint_names.index(name) for name in joint_utils.H2LORA_BODY_JOINTS_IN_G1_ORDER]
    h2_body = np.asarray(dof[:, body_indices], dtype=np.float32)
    with torch.no_grad():
        semantic = h2lora_alignment.h2_to_g1_semantic(torch.from_numpy(h2_body)).cpu().numpy()
    return semantic.astype(np.float32)


def _h2_physical_to_mjcf_order(dof: np.ndarray, joint_names: list[str]) -> np.ndarray:
    missing = [name for name in H2_MJCF_JOINT_NAMES if name not in joint_names]
    if missing:
        raise ValueError(f"H2 motion is missing physical joints required by deploy reset: {missing}")
    indices = [joint_names.index(name) for name in H2_MJCF_JOINT_NAMES]
    h2_local = np.asarray(dof[:, indices], dtype=np.float32).copy()
    # H2LoRA policy never controls the head.  Keep the deploy reset/init sidecar
    # aligned with the direct default head command used in training and sim2sim.
    h2_local[:, 15:17] = np.asarray([0.0, 0.0], dtype=np.float32)
    return h2_local


def _fit_frames(values: np.ndarray, expected_frames: int, *, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape[0] == expected_frames:
        return arr
    if arr.shape[0] <= 0:
        raise ValueError(f"{label} has no frames")
    if arr.shape[0] > expected_frames:
        return arr[:expected_frames].copy()
    pad = np.repeat(arr[-1:], expected_frames - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def _export_smpl(smpl_path: Path, motion_dir: Path, expected_frames: int) -> dict[str, Any]:
    key, smpl = _unwrap_motion_pkl(smpl_path)
    exported: dict[str, Any] = {"source_key": key, "source_path": str(smpl_path)}

    if "smpl_joints" in smpl:
        smpl_joints = np.asarray(smpl["smpl_joints"], dtype=np.float32)
        if smpl_joints.ndim != 3 or smpl_joints.shape[1:] != (24, 3):
            raise ValueError(f"smpl_joints must have shape (T, 24, 3), got {smpl_joints.shape} in {smpl_path}")
        source_smpl_joint_frames = int(smpl_joints.shape[0])
        smpl_joints = _fit_frames(smpl_joints, expected_frames, label=f"smpl_joints in {smpl_path}")
        _write_csv(
            motion_dir / "smpl_joint.csv",
            smpl_joints.reshape(expected_frames, -1),
            _csv_headers_3d("smpl_joint", smpl_joints.shape[1]),
        )
        exported["smpl_joint_shape"] = list(smpl_joints.shape)
        exported["source_smpl_joint_frames"] = source_smpl_joint_frames

    root_quat_wxyz: np.ndarray | None = None
    if "pose_aa" in smpl:
        pose_aa = np.asarray(smpl["pose_aa"], dtype=np.float32)
        if pose_aa.ndim == 2 and pose_aa.shape[1] == 72:
            pose_aa = pose_aa.reshape(pose_aa.shape[0], 24, 3)
        if pose_aa.ndim == 3 and pose_aa.shape[1] >= 22 and pose_aa.shape[2] == 3:
            source_smpl_pose_frames = int(pose_aa.shape[0])
            pose_aa = _fit_frames(pose_aa, expected_frames, label=f"pose_aa in {smpl_path}")
            root_quat_wxyz = _smpl_root_yup_to_runtime_wxyz(_axis_angle_to_wxyz(pose_aa[:, 0, :]))
            smpl_pose = pose_aa[:, 1:22, :]
            _write_csv(
                motion_dir / "smpl_pose.csv",
                smpl_pose.reshape(expected_frames, -1),
                _csv_headers_3d("smpl_pose", smpl_pose.shape[1]),
            )
            exported["smpl_pose_shape"] = list(smpl_pose.shape)
            exported["source_smpl_pose_frames"] = source_smpl_pose_frames

    if root_quat_wxyz is not None:
        _write_csv(
            motion_dir / "smpl_root_quat.csv",
            root_quat_wxyz,
            _csv_headers_quat("smpl_root", 1),
        )
        exported["smpl_root_quat_shape"] = list(root_quat_wxyz.shape)
        exported["smpl_root_quat_format"] = "wxyz_after_smpl_yup_to_zup_and_remove_base_rot"

    return exported


def _write_deploy_csv_bundle(
    *,
    motion_dir: Path,
    motion_name: str,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    h2_joint_pos: np.ndarray | None,
    h2_joint_vel: np.ndarray | None,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    fps: float,
    metadata_lines: list[str],
) -> None:
    frames = joint_pos.shape[0]
    body_pos = root_pos[:, None, :]
    body_quat = root_quat_wxyz[:, None, :]
    body_lin_vel = _finite_difference(root_pos, fps)[:, None, :]
    body_ang_vel = _angular_velocity_from_wxyz(root_quat_wxyz, fps)[:, None, :]

    _write_csv(motion_dir / "joint_pos.csv", joint_pos, _csv_headers("joint", joint_pos.shape[1]))
    _write_csv(motion_dir / "joint_vel.csv", joint_vel, _csv_headers("joint_vel", joint_vel.shape[1]))
    if h2_joint_pos is not None:
        _write_csv(
            motion_dir / "h2_joint_pos.csv",
            h2_joint_pos,
            _csv_headers("h2_joint", h2_joint_pos.shape[1]),
        )
    if h2_joint_vel is not None:
        _write_csv(
            motion_dir / "h2_joint_vel.csv",
            h2_joint_vel,
            _csv_headers("h2_joint_vel", h2_joint_vel.shape[1]),
        )
    _write_csv(motion_dir / "body_pos.csv", body_pos.reshape(frames, -1), _csv_headers_3d("body", 1))
    _write_csv(motion_dir / "body_quat.csv", body_quat.reshape(frames, -1), _csv_headers_quat("body", 1))
    _write_csv(motion_dir / "body_lin_vel.csv", body_lin_vel.reshape(frames, -1), _csv_headers_3d("body_lin_vel", 1))
    _write_csv(motion_dir / "body_ang_vel.csv", body_ang_vel.reshape(frames, -1), _csv_headers_3d("body_ang_vel", 1))

    metadata = [
        f"Metadata for: {motion_name}",
        "=" * 30,
        "",
        "Body part indexes:",
        "[0]",
        "",
        f"Total timesteps: {frames}",
        f"FPS: {int(fps)}",
    ]
    metadata.extend(metadata_lines)
    metadata.append("")
    (motion_dir / "metadata.txt").write_text("\n".join(metadata), encoding="utf-8")


def _export_default_stand(output_root: Path, *, frames: int, fps: float) -> dict[str, Any]:
    with DEFAULT_H2LORA_WBC_CONFIG.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    h2_default = np.asarray(config["DEFAULT_DOF_ANGLES"], dtype=np.float32)
    root_pos_single = np.asarray(config.get("DEFAULT_BASE_POS", [0.0, 0.0, 1.04]), dtype=np.float32)
    root_quat_single = np.asarray(config.get("DEFAULT_BASE_QUAT", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
    if h2_default.shape != (31,):
        raise ValueError(f"Expected 31 H2 default DOF angles, got {h2_default.shape}")
    if root_quat_single.shape != (4,):
        raise ValueError(f"Expected DEFAULT_BASE_QUAT to have 4 values, got {root_quat_single.shape}")
    root_quat_single = root_quat_single / max(float(np.linalg.norm(root_quat_single)), 1e-12)

    motion_name = "h2_default_stand"
    motion_dir = output_root / motion_name
    motion_dir.mkdir(parents=True, exist_ok=True)

    dof = np.repeat(h2_default[None, :], frames, axis=0)
    joint_pos = _h2_physical_to_g1_semantic(dof, H2_MJCF_JOINT_NAMES)
    joint_vel = np.zeros_like(joint_pos, dtype=np.float32)
    root_pos = np.repeat(root_pos_single[None, :], frames, axis=0).astype(np.float32)
    root_quat_wxyz = np.repeat(root_quat_single[None, :], frames, axis=0).astype(np.float32)

    _write_deploy_csv_bundle(
        motion_dir=motion_dir,
        motion_name=motion_name,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        h2_joint_pos=dof,
        h2_joint_vel=np.zeros_like(dof, dtype=np.float32),
        root_pos=root_pos,
        root_quat_wxyz=root_quat_wxyz,
        fps=fps,
        metadata_lines=[
            "Deploy source: h2_lora",
            "Reference joint space: g1_semantic_29d_from_h2_physical_headless",
            "Synthetic reference: true",
            f"Source config: {DEFAULT_H2LORA_WBC_CONFIG}",
        ],
    )

    info = {
        "motion_name": motion_name,
        "source_key": motion_name,
        "source_robot_pkl": None,
        "source_smpl_pkl": None,
        "fps": int(fps),
        "frames": int(frames),
        "joint_space": "g1_semantic_29d_from_h2_physical_headless",
        "source_joint_space": "h2_physical_31d_mujoco_default_pose",
        "joint_pos_shape": list(joint_pos.shape),
        "h2_joint_pos_shape": list(dof.shape),
        "body_quat_format": "wxyz",
        "body_count": 1,
        "smpl": None,
        "h2_default_angles": h2_default.tolist(),
        "root_pos": root_pos_single.tolist(),
        "root_quat_wxyz": root_quat_single.tolist(),
        "note": "H2LoRA nominal default is a bent-knee standing pose: knee=0.669 rad, hip_pitch=-0.312 rad, ankle_pitch=-0.363 rad.",
    }
    (motion_dir / "h2_lora_reference.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def _export_motion(
    robot_path: Path,
    motion_root: Path,
    smpl_root: Path | None,
    output_root: Path,
) -> dict[str, Any]:
    motion_key, motion = _unwrap_motion_pkl(robot_path)
    fps = float(motion.get("fps", 50))
    if abs(fps - 50.0) > 1e-4:
        raise ValueError(f"H2LoRA deploy reference expects 50 Hz data, got fps={fps} in {robot_path}")

    dof = np.asarray(motion.get("dof", motion.get("dof_pos")), dtype=np.float32)
    if dof.ndim != 2 or dof.shape[1] != 31:
        raise ValueError(f"H2 robot dof must have shape (T, 31), got {dof.shape} in {robot_path}")
    joint_names = list(motion.get("joint_names", []))
    if len(joint_names) != 31:
        raise ValueError(f"H2 robot motion must carry 31 joint_names, got {len(joint_names)} in {robot_path}")

    root_pos = np.asarray(motion.get("root_trans_offset", motion.get("root_pos")), dtype=np.float32)
    root_rot_xyzw = np.asarray(motion["root_rot"], dtype=np.float32)
    if root_pos.shape != (dof.shape[0], 3):
        raise ValueError(f"root position shape mismatch in {robot_path}: {root_pos.shape}")
    if root_rot_xyzw.shape != (dof.shape[0], 4):
        raise ValueError(f"root_rot shape mismatch in {robot_path}: {root_rot_xyzw.shape}")

    motion_name = _sanitize_motion_name(motion_key)
    motion_dir = output_root / motion_name
    motion_dir.mkdir(parents=True, exist_ok=True)

    joint_pos = _h2_physical_to_g1_semantic(dof, joint_names)
    joint_vel = _motionlib_forward_dof_velocity(joint_pos, fps)
    h2_joint_pos = _h2_physical_to_mjcf_order(dof, joint_names)
    h2_joint_vel = _motionlib_forward_dof_velocity(h2_joint_pos, fps)
    root_quat_wxyz = _xyzw_to_wxyz(root_rot_xyzw)
    frames = joint_pos.shape[0]

    _write_deploy_csv_bundle(
        motion_dir=motion_dir,
        motion_name=motion_name,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        h2_joint_pos=h2_joint_pos,
        h2_joint_vel=h2_joint_vel,
        root_pos=root_pos,
        root_quat_wxyz=root_quat_wxyz,
        fps=fps,
        metadata_lines=[
            "Deploy source: h2_lora",
            "Reference joint space: g1_semantic_29d_from_h2_physical_headless",
            f"Source robot pkl: {robot_path}",
            f"Source smpl pkl: {_resolve_smpl_path(robot_path, motion_root, smpl_root) or ''}",
        ],
    )

    smpl_path = _resolve_smpl_path(robot_path, motion_root, smpl_root)
    smpl_info = None
    if smpl_path is not None:
        smpl_info = _export_smpl(smpl_path, motion_dir, frames)

    info = {
        "motion_name": motion_name,
        "source_key": motion_key,
        "source_robot_pkl": str(robot_path),
        "source_smpl_pkl": str(smpl_path) if smpl_path else None,
        "fps": int(fps),
        "frames": int(frames),
        "joint_space": "g1_semantic_29d_from_h2_physical_headless",
        "source_joint_space": "h2_physical_31d_mujoco",
        "joint_pos_shape": list(joint_pos.shape),
        "h2_joint_pos_shape": list(h2_joint_pos.shape),
        "body_quat_format": "wxyz",
        "body_count": 1,
        "smpl": smpl_info,
    }
    (motion_dir / "h2_lora_reference.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def _scalar_npz_value(npz: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in npz.files:
        return default
    value = npz[key]
    if getattr(value, "shape", None) == ():
        return value.item()
    return value


def _infer_encoder_name_from_label(label: str, motion_name: str) -> str:
    lowered = f"{label} {motion_name}".lower()
    if re.search(r"(^|[_\s-])smpl($|[_\s-])", lowered):
        return "smpl"
    if re.search(r"(^|[_\s-])teleop($|[_\s-])", lowered):
        return "teleop"
    return "g1"


def _encoder_mode_from_name(encoder_name: str) -> int:
    modes = {"g1": 0, "teleop": 1, "smpl": 2}
    if encoder_name not in modes:
        raise ValueError(f"Unknown H2LoRA encoder name: {encoder_name}")
    return modes[encoder_name]


def _resolve_smpl_path_for_key(motion_key: str, smpl_root: Path | None) -> Path | None:
    if smpl_root is None or not smpl_root.exists():
        return None
    direct_matches = sorted(smpl_root.rglob(f"{motion_key}.pkl"))
    if direct_matches:
        return direct_matches[0]
    return None


def _export_trajectory_npz(npz_path: Path, output_root: Path, smpl_root: Path | None = None) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        fps = float(_scalar_npz_value(data, "fps", 50.0))
        if abs(fps - 50.0) > 1e-4:
            raise ValueError(f"H2LoRA deploy reference expects 50 Hz data, got fps={fps} in {npz_path}")
        motion_key = str(_scalar_npz_value(data, "motion_key", npz_path.stem))
        label = str(_scalar_npz_value(data, "label", motion_key))
        motion_name = _sanitize_motion_name(label or motion_key)
        encoder_name = _infer_encoder_name_from_label(label, motion_name)
        encoder_mode = _encoder_mode_from_name(encoder_name)

        if "ref_dof_mujoco" not in data.files:
            raise ValueError(f"{npz_path} does not contain ref_dof_mujoco")
        h2_joint_pos = np.asarray(data["ref_dof_mujoco"], dtype=np.float32)
        if h2_joint_pos.ndim != 2 or h2_joint_pos.shape[1] != 31:
            raise ValueError(f"ref_dof_mujoco must have shape (T, 31), got {h2_joint_pos.shape} in {npz_path}")
        h2_joint_pos = h2_joint_pos.copy()
        h2_joint_pos[:, 15:17] = np.asarray([0.0, 0.0], dtype=np.float32)

        # The visualization NPZ stores ref_dof_mujoco in the local H2 MJCF
        # order.  Its robot_joint_names metadata is IsaacLab order, so using it
        # here would silently corrupt the G1-semantic encoder reference while
        # leaving h2_joint_pos itself looking correct.
        joint_pos = _h2_physical_to_g1_semantic(h2_joint_pos, H2_MJCF_JOINT_NAMES)
        joint_vel = _motionlib_forward_dof_velocity(joint_pos, fps)
        h2_joint_vel = _motionlib_forward_dof_velocity(h2_joint_pos, fps)

        if "ref_root_pos" not in data.files or "ref_root_quat" not in data.files:
            raise ValueError(f"{npz_path} must contain ref_root_pos and ref_root_quat")
        root_pos = np.asarray(data["ref_root_pos"], dtype=np.float32)
        root_quat = np.asarray(data["ref_root_quat"], dtype=np.float32)
        if root_pos.shape != (h2_joint_pos.shape[0], 3):
            raise ValueError(f"ref_root_pos shape mismatch in {npz_path}: {root_pos.shape}")
        if root_quat.shape != (h2_joint_pos.shape[0], 4):
            raise ValueError(f"ref_root_quat shape mismatch in {npz_path}: {root_quat.shape}")

        # visualize_h2lora_ref_tracking.py stores quaternions in IsaacLab wxyz
        # order.  The deploy CSV bundle also expects wxyz, so only normalize.
        root_quat_wxyz = np.asarray(root_quat, dtype=np.float64)
        norm = np.linalg.norm(root_quat_wxyz, axis=-1, keepdims=True)
        root_quat_wxyz = (root_quat_wxyz / np.where(norm > 1e-12, norm, 1.0)).astype(np.float32)

        action_source = str(_scalar_npz_value(data, "action_source", ""))
        onnx_encoder = str(_scalar_npz_value(data, "onnx_encoder", ""))
        onnx_decoder = str(_scalar_npz_value(data, "onnx_decoder", ""))
        checkpoint = str(_scalar_npz_value(data, "checkpoint", ""))
        done = np.asarray(data["done"], dtype=bool) if "done" in data.files else None
        joint_mae = np.asarray(data["joint_mae"], dtype=np.float64) if "joint_mae" in data.files else None
        root_error = np.asarray(data["root_error"], dtype=np.float64) if "root_error" in data.files else None

    motion_dir = output_root / motion_name
    motion_dir.mkdir(parents=True, exist_ok=True)
    _write_deploy_csv_bundle(
        motion_dir=motion_dir,
        motion_name=motion_name,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        h2_joint_pos=h2_joint_pos,
        h2_joint_vel=h2_joint_vel,
        root_pos=root_pos,
        root_quat_wxyz=root_quat_wxyz,
        fps=fps,
        metadata_lines=[
            "Deploy source: h2_lora_trajectory_npz",
            "Reference joint space: g1_semantic_29d_from_h2_physical_headless",
            f"Encoder mode: {encoder_mode}",
            f"Encoder name: {encoder_name}",
            f"Source trajectory npz: {npz_path}",
            f"Action source: {action_source}",
            f"ONNX encoder: {onnx_encoder}",
            f"ONNX decoder: {onnx_decoder}",
        ],
    )

    smpl_path = _resolve_smpl_path_for_key(motion_key, smpl_root)
    smpl_info = None
    if encoder_name == "smpl":
        if smpl_path is None:
            raise FileNotFoundError(
                f"Trajectory NPZ '{npz_path}' is labeled as SMPL encoder input, "
                f"but no matching '{motion_key}.pkl' was found under --smpl-root={smpl_root}. "
                "SMPL deploy references must include smpl_joint.csv to match the ONNX "
                "IsaacSim visualization path."
            )
        smpl_info = _export_smpl(smpl_path, motion_dir, joint_pos.shape[0])

    info = {
        "motion_name": motion_name,
        "source_key": motion_key,
        "source_robot_pkl": str(npz_path),
        "source_smpl_pkl": str(smpl_path) if smpl_path else None,
        "encoder_mode": int(encoder_mode),
        "encoder_name": encoder_name,
        "fps": int(fps),
        "frames": int(joint_pos.shape[0]),
        "joint_space": "g1_semantic_29d_from_h2_physical_headless",
        "source_joint_space": "trajectory_npz_ref_dof_mujoco_31d",
        "joint_pos_shape": list(joint_pos.shape),
        "h2_joint_pos_shape": list(h2_joint_pos.shape),
        "body_quat_format": "wxyz",
        "body_count": 1,
        "smpl": smpl_info,
        "trajectory_npz": {
            "path": str(npz_path),
            "label": label,
            "action_source": action_source,
            "onnx_encoder": onnx_encoder,
            "onnx_decoder": onnx_decoder,
            "checkpoint": checkpoint,
            "done_any": bool(done.any()) if done is not None else None,
            "joint_mae_mean": float(joint_mae.mean()) if joint_mae is not None else None,
            "joint_mae_max": float(joint_mae.max()) if joint_mae is not None else None,
            "root_error_mean": float(root_error.mean()) if root_error is not None else None,
            "root_error_max": float(root_error.max()) if root_error is not None else None,
        },
    }
    (motion_dir / "h2_lora_reference.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def export_references(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    trajectory_npz_paths = [Path(path).resolve() for path in (args.trajectory_npz or [])]
    if args.default_stand_only or trajectory_npz_paths:
        robot_paths = []
    else:
        motion_root = Path(args.motion_root).resolve()
        smpl_root = Path(args.smpl_root).resolve() if args.smpl_root else None
        robot_paths = _resolve_robot_paths(args)
    motions: list[dict[str, Any]] = []
    if args.default_stand_only or args.add_default_stand:
        motions.append(_export_default_stand(output_root, frames=args.stand_frames, fps=50.0))
    if trajectory_npz_paths:
        smpl_root = Path(args.smpl_root).resolve() if args.smpl_root else None
        motions.extend(_export_trajectory_npz(path, output_root, smpl_root) for path in trajectory_npz_paths)
        motion_root_for_manifest = [str(path) for path in trajectory_npz_paths]
        smpl_root_for_manifest = str(smpl_root) if smpl_root else None
        description = (
            "H2LoRA trajectory NPZ references exported for deploy. joint_pos/joint_vel are "
            "G1 semantic 29D derived from ref_dof_mujoco in IsaacSim visualization NPZs."
        )
    else:
        motion_root = Path(args.motion_root).resolve()
        smpl_root = Path(args.smpl_root).resolve() if args.smpl_root else None
        motions.extend(_export_motion(path, motion_root, smpl_root, output_root) for path in robot_paths)
        motion_root_for_manifest = str(motion_root)
        smpl_root_for_manifest = str(smpl_root) if smpl_root else None
        description = (
            "H2 MotionLib references exported for H2LoRA deploy. joint_pos/joint_vel are "
            "G1 semantic 29D derived from H2 physical headless joints."
        )
    manifest = {
        "schema": "h2_lora_deploy_reference_v1",
        "description": description,
        "motion_root": motion_root_for_manifest,
        "smpl_root": smpl_root_for_manifest,
        "output_dir": str(output_root),
        "fps": 50,
        "num_motions": len(motions),
        "total_frames": int(sum(item["frames"] for item in motions)),
        "motions": motions,
    }
    (output_root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_lines = [
        "H2LoRA Deploy Reference Summary",
        "=" * 36,
        f"Schema: {manifest['schema']}",
        f"Motions: {manifest['num_motions']}",
        f"Total frames: {manifest['total_frames']}",
        "",
    ]
    for item in motions:
        summary_lines.append(f"- {item['motion_name']}: {item['frames']} frames, source={item['source_robot_pkl']}")
    (output_root / "motion_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", type=Path, default=DEFAULT_ROBOT_ROOT)
    parser.add_argument("--smpl-root", type=Path, default=DEFAULT_SMPL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--motion-key", action="append", help="Exact MotionLib PKL stem/key to export. Can be repeated.")
    parser.add_argument("--trajectory-npz", type=Path, action="append", help="Export an IsaacSim H2LoRA trajectory_ref_tracking.npz. Can be repeated.")
    parser.add_argument("--include-regex", help="Export PKL stems matching this regex.")
    parser.add_argument("--all", action="store_true", help="Export every PKL under --motion-root.")
    parser.add_argument("--limit", type=int, help="Limit selected motions after sorting.")
    parser.add_argument("--add-default-stand", action="store_true", help="Also export a synthetic static H2 default-stand reference.")
    parser.add_argument("--default-stand-only", action="store_true", help="Export only the synthetic static H2 default-stand reference.")
    parser.add_argument("--stand-frames", type=int, default=1000, help="Number of 50 Hz frames for the synthetic stand reference.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = export_references(args)
    print(f"Exported {manifest['num_motions']} H2LoRA deploy references to {manifest['output_dir']}")
    print(f"Total frames: {manifest['total_frames']}")
    print(f"Manifest: {Path(manifest['output_dir']) / MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
