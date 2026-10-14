#!/usr/bin/env python3
"""Build H2LoRA MotionLib PKLs from the local SEED H2/SMPL-X dataset."""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import json
import multiprocessing as mp
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import joblib
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import transform
from tqdm import tqdm

from gear_sonic.envs.env_utils import joint_utils

ASSET_ROOT = Path("gear_sonic/data/assets/robot_description")
H2_MJCF = ASSET_ROOT / "mjcf/h2.xml"

H2_CSV_JOINT_NAMES = [
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

_SMPL_MODEL = None
_HUMAN_JOINTS_FN = None
_SMPL_DEVICE = "cpu"
_SMPL_MODE = "soma-bvh"
_SMPL_ROOT_MODE = "soma-bvh-compatible"
_SMPL_BVH_ROOT = Path("/mnt/data/seed/soma_uniform/bvh")
_SMPL_BATCH_FRAMES = 512
H2LORA_SEED_SMPL_SCHEMA_VERSION = 6

SMPL24_FROM_SOMA_NAMES = [
    "Hips",
    "LeftLeg",
    "RightLeg",
    "Spine1",
    "LeftShin",
    "RightShin",
    "Spine2",
    "LeftFoot",
    "RightFoot",
    "Chest",
    "LeftToeBase",
    "RightToeBase",
    "Neck1",
    "LeftShoulder",
    "RightShoulder",
    "Head",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
    "LeftHandThumb1",
    "RightHandThumb1",
]


def _parse_vec(text: str | None, n: int, default: float = 0.0) -> np.ndarray:
    if text is None:
        return np.full(n, default, dtype=np.float32)
    return np.asarray([float(x) for x in text.split()], dtype=np.float32)


@lru_cache(maxsize=1)
def parse_h2_mjcf(
    mjcf_path: Path = H2_MJCF,
) -> tuple[list[str], dict[str, str], dict[str, np.ndarray]]:
    """Return MJCF body order and joint axis metadata used by Humanoid_Batch."""
    root = ET.parse(mjcf_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF has no worldbody: {mjcf_path}")

    body_names: list[str] = []
    joint_to_body: dict[str, str] = {}
    joint_axes: dict[str, np.ndarray] = {}

    def walk(body: ET.Element) -> None:
        body_name = body.get("name")
        if body_name is None:
            return
        body_names.append(body_name)
        for joint in body.findall("joint"):
            joint_name = joint.get("name")
            if joint_name is None or joint_name == "floating_base_joint":
                continue
            joint_to_body[joint_name] = body_name
            joint_axes[joint_name] = _parse_vec(joint.get("axis"), 3)
        for child in body.findall("body"):
            walk(child)

    for body in worldbody.findall("body"):
        walk(body)
    return body_names, joint_to_body, joint_axes


def downsample_array(array: np.ndarray, fps_source: int, fps_target: int) -> np.ndarray:
    if fps_source == fps_target:
        return array.astype(np.float32, copy=False)
    if fps_source % fps_target == 0:
        jump = int(fps_source / fps_target)
        if jump > 1:
            return array[::jump].astype(np.float32, copy=False)

    # 120 Hz -> 50 Hz is not an integer stride.  Use time-domain linear
    # resampling so the stored MotionLib fps matches the declared fps exactly.
    n_src = array.shape[0]
    duration = (n_src - 1) / float(fps_source)
    source_t = np.arange(n_src, dtype=np.float64) / float(fps_source)
    target_t = np.arange(0.0, duration + 1e-9, 1.0 / float(fps_target), dtype=np.float64)
    if target_t.size == 0:
        target_t = np.asarray([0.0], dtype=np.float64)
    if target_t[-1] > source_t[-1]:
        target_t[-1] = source_t[-1]
    interpolator = interp1d(
        source_t,
        array,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=(array[0], array[-1]),
        assume_sorted=True,
    )
    return interpolator(target_t).astype(np.float32, copy=False)


def resample_quat_xyzw(quat_xyzw: np.ndarray, fps_source: int, fps_target: int) -> np.ndarray:
    """Resample root quaternions with Slerp for non-integer rate changes."""
    if fps_source == fps_target:
        out = quat_xyzw.astype(np.float32, copy=False)
    elif fps_source % fps_target == 0:
        jump = int(fps_source / fps_target)
        out = quat_xyzw[::jump].astype(np.float32, copy=False) if jump > 1 else quat_xyzw
    else:
        n_src = quat_xyzw.shape[0]
        duration = (n_src - 1) / float(fps_source)
        source_t = np.arange(n_src, dtype=np.float64) / float(fps_source)
        target_t = np.arange(0.0, duration + 1e-9, 1.0 / float(fps_target), dtype=np.float64)
        if target_t.size == 0:
            target_t = np.asarray([0.0], dtype=np.float64)
        if target_t[-1] > source_t[-1]:
            target_t[-1] = source_t[-1]
        slerp = transform.Slerp(source_t, transform.Rotation.from_quat(quat_xyzw))
        out = slerp(target_t).as_quat().astype(np.float32)
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return (out / np.clip(norm, 1e-8, None)).astype(np.float32, copy=False)


def _smpl_root_axis_angle_yup_to_zup(root_axis_angle: np.ndarray) -> np.ndarray:
    """Match runtime ``smpl_root_ytoz_up`` for SMPL root axis-angle values."""
    root_rot = transform.Rotation.from_rotvec(
        root_axis_angle.reshape(-1, 3).astype(np.float64, copy=False)
    )
    y_to_z = transform.Rotation.from_rotvec(np.asarray([np.pi / 2.0, 0.0, 0.0]))
    return (y_to_z * root_rot).as_rotvec().astype(np.float32).reshape(root_axis_angle.shape)


def _smpl_yup_points_to_zup(points: np.ndarray) -> np.ndarray:
    """Convert SMPL/SEED Y-up points to Isaac/robot Z-up points."""
    out = points.astype(np.float32, copy=True)
    out[..., 1] = -points[..., 2]
    out[..., 2] = points[..., 1]
    return out


def load_h2_csv(csv_path: Path, fps_source: int, fps_target: int) -> dict:
    body_names, joint_to_body, joint_axes = parse_h2_mjcf()
    body_index = {name: idx for idx, name in enumerate(body_names)}

    columns: dict[str, list[float]] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        required_cols = [
            "root_translateX",
            "root_translateY",
            "root_translateZ",
            "root_rotateX",
            "root_rotateY",
            "root_rotateZ",
            *[f"{name}_dof" for name in H2_CSV_JOINT_NAMES],
        ]
        missing = [col for col in required_cols if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"{csv_path} missing columns: {missing[:8]}")
        columns = {col: [] for col in required_cols}
        for row in reader:
            for col in required_cols:
                columns[col].append(float(row[col]))

    if not columns["root_translateX"]:
        raise ValueError(f"CSV has no frames: {csv_path}")

    root_trans = (
        np.stack(
            [
                columns["root_translateX"],
                columns["root_translateY"],
                columns["root_translateZ"],
            ],
            axis=1,
        ).astype(np.float32)
        / 100.0
    )
    root_euler = np.stack(
        [
            columns["root_rotateX"],
            columns["root_rotateY"],
            columns["root_rotateZ"],
        ],
        axis=1,
    ).astype(np.float64)
    root_quat_xyzw = transform.Rotation.from_euler(
        "xyz", root_euler, degrees=True
    ).as_quat().astype(np.float32)
    root_rotvec = transform.Rotation.from_quat(root_quat_xyzw).as_rotvec().astype(np.float32)

    dof = np.stack(
        [np.deg2rad(columns[f"{name}_dof"]) for name in H2_CSV_JOINT_NAMES],
        axis=1,
    ).astype(np.float32)
    pose_aa = np.zeros((dof.shape[0], len(body_names), 3), dtype=np.float32)
    pose_aa[:, 0, :] = root_rotvec
    for joint_idx, joint_name in enumerate(H2_CSV_JOINT_NAMES):
        pose_aa[:, body_index[joint_to_body[joint_name]], :] = (
            joint_axes[joint_name][None, :] * dof[:, joint_idx : joint_idx + 1]
        )

    return {
        "root_trans_offset": downsample_array(root_trans, fps_source, fps_target),
        "pose_aa": downsample_array(pose_aa, fps_source, fps_target),
        "dof": downsample_array(dof, fps_source, fps_target),
        "root_rot": resample_quat_xyzw(root_quat_xyzw, fps_source, fps_target),
        "smpl_joints": np.zeros(
            (downsample_array(dof, fps_source, fps_target).shape[0], 24, 3),
            dtype=np.float32,
        ),
        "fps": int(fps_target),
        "joint_names": list(H2_CSV_JOINT_NAMES),
        "joint_order": "h2_mjcf",
        "dof_order": "mujoco",
        "body_names": body_names,
        "body_order": "h2_mjcf",
        "head_joint_names": list(joint_utils.H2_HEAD_JOINTS),
    }


def _init_worker(
    smpl_mode: str,
    smpl_body_model_dir: str,
    smpl_device: str,
    smpl_batch_frames: int,
    smpl_bvh_root: str,
    smpl_root_mode: str,
):
    global _HUMAN_JOINTS_FN, _SMPL_BATCH_FRAMES, _SMPL_BVH_ROOT, _SMPL_DEVICE
    global _SMPL_MODE, _SMPL_MODEL, _SMPL_ROOT_MODE
    _SMPL_MODE = smpl_mode
    _SMPL_DEVICE = smpl_device
    _SMPL_BATCH_FRAMES = smpl_batch_frames
    _SMPL_BVH_ROOT = Path(smpl_bvh_root)
    _SMPL_ROOT_MODE = smpl_root_mode
    _HUMAN_JOINTS_FN = None
    if smpl_mode != "smplx-fk":
        _SMPL_MODEL = None
        return
    import torch

    from gear_sonic.trl.utils.smplx.body_model.smplx_lite import SmplxLiteSmplN24

    model_path = Path(smpl_body_model_dir) / "smplx"
    _SMPL_MODEL = SmplxLiteSmplN24(model_path=str(model_path)).to(smpl_device)
    _SMPL_MODEL.eval()
    torch.set_grad_enabled(False)


def _soma_bvh_root_quat_to_smpl_pose_rotvec(root_quat_xyzw: np.ndarray) -> np.ndarray:
    """Encode a SOMA/BVH root quat into SMPL pose root space for SONIC runtime.

    Runtime SMPL root processing applies ``smpl_root_ytoz_up`` then
    ``remove_smpl_base_rot``.  The SOMA/BVH path applies ``smpl_root_ytoz_up``
    then ``remove_bvh_base_rot``.  The default mode right-multiplies the stored
    root by the SMPL base rotation twice so both runtime paths produce the same
    Z-up root orientation.
    """
    root_rot = transform.Rotation.from_quat(root_quat_xyzw.astype(np.float64, copy=False))
    if _SMPL_ROOT_MODE == "raw-bvh":
        return root_rot.as_rotvec().astype(np.float32)
    if _SMPL_ROOT_MODE != "soma-bvh-compatible":
        raise ValueError(f"Unsupported SMPL root mode: {_SMPL_ROOT_MODE}")
    smpl_base = transform.Rotation.from_quat(np.asarray([0.5, 0.5, 0.5, 0.5]))
    return (root_rot * smpl_base * smpl_base).as_rotvec().astype(np.float32)


def _resolve_soma_bvh_path(npz_path: Path, source_bvh_relpath: str | None) -> Path:
    if source_bvh_relpath:
        candidate = _SMPL_BVH_ROOT / source_bvh_relpath
        if candidate.exists():
            return candidate
    candidate = _SMPL_BVH_ROOT / npz_path.parent.name / f"{npz_path.stem}.bvh"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Cannot resolve SOMA BVH for {npz_path}; tried relpath={source_bvh_relpath!r} "
        f"under {_SMPL_BVH_ROOT}"
    )


def _smpl24_from_soma_bvh(
    npz_path: Path,
    source_bvh_relpath: str | None,
    fps_source: int,
    fps_target: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    """Build SONIC's SMPL-encoder contract from the SEED SOMA/BVH skeleton."""
    from gear_sonic.data_process.extract_soma_joints_from_bvh import (
        compute_fk_selected,
        parse_bvh,
    )

    bvh_path = _resolve_soma_bvh_path(npz_path, source_bvh_relpath)
    joints, channel_order, motion_data, _n_frames, frame_time = parse_bvh(str(bvh_path))
    actual_fps = int(round(1.0 / frame_time))
    if abs(actual_fps - int(fps_source)) > 1:
        raise ValueError(f"BVH fps {actual_fps} does not match requested source fps {fps_source}")

    positions, root_quat_xyzw = compute_fk_selected(
        joints, channel_order, motion_data, SMPL24_FROM_SOMA_NAMES
    )
    if positions.shape[1] != len(SMPL24_FROM_SOMA_NAMES):
        raise ValueError(
            f"Expected {len(SMPL24_FROM_SOMA_NAMES)} SOMA joints, got {positions.shape[1]}"
        )

    positions_m = positions.astype(np.float32, copy=False) / 100.0
    transl = positions_m[:, 0, :].copy()
    positions_local = positions_m - transl[:, None, :]
    smpl_joints = _smpl_yup_points_to_zup(positions_local)

    smpl_joints = downsample_array(smpl_joints, fps_source, fps_target)
    transl = downsample_array(transl, fps_source, fps_target)
    root_quat_xyzw = resample_quat_xyzw(root_quat_xyzw, fps_source, fps_target)
    root_rotvec = _soma_bvh_root_quat_to_smpl_pose_rotvec(root_quat_xyzw)

    pose_aa = np.zeros((root_rotvec.shape[0], 72), dtype=np.float32)
    pose_aa[:, :3] = root_rotvec
    metadata = {
        "source_bvh_path": str(bvh_path),
        "smpl_pose_body_source": "zeros_nonroot_because_seed_smplx_poses_are_somax_77_not_smpl24",
        "smpl_joints_source": "soma_bvh_fk_name_mapped_to_smpl24",
        "smpl_joints_source_names": ",".join(SMPL24_FROM_SOMA_NAMES),
        "smpl_root_mode": _SMPL_ROOT_MODE,
    }
    return pose_aa, smpl_joints.astype(np.float32), transl.astype(np.float32), metadata


def _smpl_joints_24(poses: np.ndarray, transl: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Generate SMPL 24 joints in the same frame consumed by SONIC's SMPL encoder.

    ``pose_aa`` is stored in the original SMPL/SEED Y-up convention because the
    runtime command term converts root orientation through ``smpl_y_up``.  The
    MotionLib tokenizer observation then rotates stored ``smpl_joints`` by
    ``inv(remove_smpl_base_rot(smpl_root_ytoz_up(root)))``.  Following the SOMA
    data chain, the stored joints are Z-up, pelvis-centered body-local vectors
    that are still oriented by the SMPL root before runtime canonicalization.
    They must not carry global translation or the SMPL rest-pelvis offset.
    """
    if _SMPL_MODE == "zeros":
        return np.zeros((poses.shape[0], 24, 3), dtype=np.float32)
    if _SMPL_MODE == "human-fk":
        return _human_fk_joints_24(poses)
    if _SMPL_MODE != "smplx-fk":
        raise ValueError(f"Unsupported SMPL mode: {_SMPL_MODE}")
    if _SMPL_MODEL is None:
        raise RuntimeError("SMPL-X FK model was not initialized")

    import torch

    chunks = []
    beta = np.asarray(betas, dtype=np.float32).reshape(1, -1)[:, :10]
    if beta.shape[-1] < 10:
        beta = np.pad(beta, ((0, 0), (0, 10 - beta.shape[-1])))
    with torch.no_grad():
        for start in range(0, poses.shape[0], _SMPL_BATCH_FRAMES):
            end = min(start + _SMPL_BATCH_FRAMES, poses.shape[0])
            pose_chunk = poses[start:end].astype(np.float32, copy=False)
            global_orient_zup = _smpl_root_axis_angle_yup_to_zup(pose_chunk[:, 0, :])
            global_orient = torch.from_numpy(global_orient_zup).to(_SMPL_DEVICE)[None]
            body_pose = torch.from_numpy(pose_chunk[:, 1:22, :].reshape(end - start, 63)).to(
                _SMPL_DEVICE
            )[None]
            transl_tensor = torch.zeros(1, end - start, 3, device=_SMPL_DEVICE)
            betas_tensor = (
                torch.from_numpy(beta)
                .to(_SMPL_DEVICE)
                .view(1, 1, 10)
                .expand(1, end - start, 10)
            )
            joints, _ = _SMPL_MODEL(body_pose, betas_tensor, global_orient, transl_tensor)
            joints = joints[0]
            joints = joints - joints[:, 0:1, :]
            chunks.append(joints.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def _human_fk_joints_24(poses: np.ndarray) -> np.ndarray:
    """Compute the same 22 body + 2 thumb SMPL joints used by live teleop."""
    global _HUMAN_JOINTS_FN
    if _HUMAN_JOINTS_FN is None:
        from gear_sonic.trl.utils.torch_transform import compute_human_joints

        _HUMAN_JOINTS_FN = compute_human_joints

    import torch

    chunks = []
    with torch.no_grad():
        for start in range(0, poses.shape[0], _SMPL_BATCH_FRAMES):
            end = min(start + _SMPL_BATCH_FRAMES, poses.shape[0])
            pose_chunk = poses[start:end].astype(np.float32, copy=False)
            global_orient_zup = _smpl_root_axis_angle_yup_to_zup(pose_chunk[:, 0, :])
            global_orient = torch.from_numpy(global_orient_zup).to(_SMPL_DEVICE)
            body_pose = torch.from_numpy(pose_chunk[:, 1:22, :].reshape(end - start, 63)).to(
                _SMPL_DEVICE
            )
            joints = _HUMAN_JOINTS_FN(body_pose=body_pose, global_orient=global_orient)
            joints = joints - joints[:, 0:1, :]
            chunks.append(joints.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def _smpl_joints_source_name() -> str:
    if _SMPL_MODE == "soma-bvh":
        return "soma_bvh_fk_name_mapped_to_smpl24"
    if _SMPL_MODE == "human-fk":
        return "compute_human_joints_22_body_plus_2_thumb"
    if _SMPL_MODE == "smplx-fk":
        return "smpl_neutral_24_regressor"
    return _SMPL_MODE


def load_smplx_npz(npz_path: Path, fps_source: int, fps_target: int) -> dict:
    with np.load(npz_path, allow_pickle=True) as data:
        poses = np.asarray(data["poses"], dtype=np.float32)
        transl = np.asarray(data["transl"], dtype=np.float32)
        source_bvh_relpath = str(data["source_bvh_relpath"].item()) if "source_bvh_relpath" in data.files else None
        if "identity_coeffs" in data.files:
            betas = np.asarray(data["identity_coeffs"], dtype=np.float32)
        else:
            betas = np.zeros((1, 10), dtype=np.float32)

    extra_metadata: dict[str, str] = {}
    if _SMPL_MODE == "soma-bvh":
        pose_aa, smpl_joints, transl, extra_metadata = _smpl24_from_soma_bvh(
            npz_path, source_bvh_relpath, fps_source, fps_target
        )
    else:
        poses = downsample_array(poses, fps_source, fps_target)
        transl = downsample_array(transl, fps_source, fps_target)
        pose_aa = poses[:, :24, :].reshape(poses.shape[0], 72).astype(np.float32)
        smpl_joints = _smpl_joints_24(poses, transl, betas)
    return {
        "pose_aa": pose_aa,
        "smpl_joints": smpl_joints,
        "transl": transl.astype(np.float32, copy=False),
        "fps": int(fps_target),
        "source_path": str(npz_path),
        "source_format": "smplx_uniform_npz",
        "schema_version": H2LORA_SEED_SMPL_SCHEMA_VERSION,
        "smpl_pose_frame": "smpl_y_up_axis_angle_root_only_for_soma_bvh",
        "smpl_joints_frame": "z_up_pelvis_centered_world_oriented_no_translation",
        "smpl_joints_root_orientation": "smpl_root_ytoz_up(pose_aa_root), base_rotation_not_removed",
        "smpl_joints_source": _smpl_joints_source_name(),
        "smpl_joints_root_center": "joint0/pelvis subtracted after FK",
        "smpl_joints_runtime_contract": (
            "tokenizer observation applies inv(remove_smpl_base_rot(smpl_root_ytoz_up(root)))"
        ),
        "smpl_transl_frame": "smplx_y_up_world",
        "smpl_y_up": True,
        "smpl_joint_count": 24,
        **extra_metadata,
    }


def _index_files(root: Path, suffix: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates = []
    for path in root.rglob(f"*{suffix}"):
        key = path.stem
        if key in index:
            duplicates.append(key)
        index[key] = path
    if duplicates:
        sample = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(f"Duplicate motion basenames under {root}: {sample}")
    return index


def _select_keys(keys: list[str], include_regex: str | None, keyword: str | None, limit: int | None):
    selected = sorted(keys)
    if keyword:
        selected = [key for key in selected if keyword.lower() in key.lower()]
    if include_regex:
        pattern = re.compile(include_regex)
        selected = [key for key in selected if pattern.search(key)]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _count_csv_data_rows(csv_path: Path) -> int:
    """Count non-header CSV rows without parsing the full file."""
    rows = 0
    with csv_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            rows += chunk.count(b"\n")
    # All seed CSV files have a header.  If the last line has no trailing newline,
    # counting newlines still includes the header line but misses the final data row.
    with csv_path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        if size == 0:
            return 0
        f.seek(size - 1)
        if f.read(1) != b"\n":
            rows += 1
    return max(0, rows - 1)


def _estimate_resampled_frames(num_source_frames: int, fps_source: int, fps_target: int) -> int:
    if num_source_frames <= 0:
        return 0
    if fps_source == fps_target:
        return num_source_frames
    if fps_source % fps_target == 0:
        jump = int(fps_source / fps_target)
        return int((num_source_frames + jump - 1) // jump)
    duration = (num_source_frames - 1) / float(fps_source)
    return int(np.floor(duration * float(fps_target) + 1e-9)) + 1


def _select_keys_by_target_frames(
    keys: list[str],
    h2_index: dict[str, Path],
    fps_source: int,
    fps_target: int,
    target_frames: int,
    random_seed: int,
) -> tuple[list[str], dict[str, int]]:
    """Uniformly shuffle motions and keep full clips until target frame count is met."""
    rng = np.random.default_rng(random_seed)
    shuffled = list(keys)
    rng.shuffle(shuffled)

    selected: list[str] = []
    frame_counts: dict[str, int] = {}
    total_frames = 0
    for key in tqdm(shuffled, desc="Selecting motions", unit="motion"):
        source_frames = _count_csv_data_rows(h2_index[key])
        target_count = _estimate_resampled_frames(source_frames, fps_source, fps_target)
        if target_count <= 0:
            continue
        selected.append(key)
        frame_counts[key] = target_count
        total_frames += target_count
        if total_frames >= target_frames:
            break

    if total_frames < target_frames:
        raise ValueError(
            f"Target frame count {target_frames} cannot be reached; selected {total_frames} frames"
        )
    return selected, frame_counts


def _process_one(task):
    key, h2_path, smpl_path, robot_out, smpl_out, fps_source, fps_target, skip_existing = task
    robot_out = Path(robot_out)
    smpl_out = Path(smpl_out)
    robot_path = robot_out / h2_path.parent.name / f"{key}.pkl"
    smpl_path_out = smpl_out / h2_path.parent.name / f"{key}.pkl"
    if (
        skip_existing
        and robot_path.exists()
        and smpl_path_out.exists()
        and _existing_robot_entry_is_current(robot_path, key, fps_target)
        and _existing_smpl_schema_is_current(
            smpl_path_out,
            key,
            fps_target=fps_target,
            smpl_mode=_SMPL_MODE,
            smpl_root_mode=_SMPL_ROOT_MODE,
        )
    ):
        return key, True, "skipped"

    robot_path.parent.mkdir(parents=True, exist_ok=True)
    smpl_path_out.parent.mkdir(parents=True, exist_ok=True)
    robot_entry = load_h2_csv(Path(h2_path), fps_source, fps_target)
    robot_entry["source_path"] = str(h2_path)
    robot_entry["paired_smplx_path"] = str(smpl_path)
    smpl_entry = load_smplx_npz(Path(smpl_path), fps_source, fps_target)
    joblib.dump({key: robot_entry}, robot_path, compress=True)
    joblib.dump({key: smpl_entry}, smpl_path_out, compress=True)
    return key, True, "converted"


def _unwrap_existing_entry(data, key: str):
    entry = data.get(key, data) if isinstance(data, dict) else {}
    if isinstance(entry, dict) and len(entry) == 1 and key in entry:
        entry = entry[key]
    return entry


def _existing_robot_entry_is_current(path: Path, key: str, fps_target: int) -> bool:
    """Return True only for robot PKLs matching the requested target fps/order."""
    try:
        data = joblib.load(path)
    except Exception:
        return False
    entry = _unwrap_existing_entry(data, key)
    if not isinstance(entry, dict):
        return False
    if int(entry.get("fps", -1)) != int(fps_target):
        return False
    dof = entry.get("dof")
    if not isinstance(dof, np.ndarray) or dof.ndim != 2 or dof.shape[-1] != len(H2_CSV_JOINT_NAMES):
        return False
    return list(entry.get("joint_names", [])) == H2_CSV_JOINT_NAMES


def _existing_smpl_schema_is_current(
    path: Path,
    key: str,
    fps_target: int | None = None,
    smpl_mode: str | None = None,
    smpl_root_mode: str | None = None,
) -> bool:
    """Return True only for SMPL PKLs built with the current H2LoRA schema."""
    try:
        data = joblib.load(path)
    except Exception:
        return False
    entry = _unwrap_existing_entry(data, key)
    if not isinstance(entry, dict):
        return False
    if entry.get("schema_version") != H2LORA_SEED_SMPL_SCHEMA_VERSION:
        return False
    if fps_target is not None and int(entry.get("fps", -1)) != int(fps_target):
        return False
    if smpl_mode is not None and entry.get("smpl_joints_source") != _smpl_joints_source_name():
        return False
    if smpl_mode == "soma-bvh" and smpl_root_mode is not None:
        if entry.get("smpl_root_mode") != smpl_root_mode:
            return False
    return True


def _write_manifest(
    output_root: Path,
    selected_keys: list[str],
    args: argparse.Namespace,
    selected_frame_counts: dict[str, int] | None = None,
    total_paired_motions: int | None = None,
) -> None:
    selected_frame_counts = selected_frame_counts or {}
    category_counts: dict[str, int] = {}
    for key in selected_keys:
        category = key.split("_", 1)[0] if "_" in key else "unknown"
        category_counts[category] = category_counts.get(category, 0) + 1
    manifest = {
        "seed_root": str(args.seed_root),
        "output_root": str(output_root),
        "fps_source": args.fps_source,
        "fps": args.fps,
        "selection_mode": args.selection_mode,
        "random_seed": args.random_seed,
        "target_frames": args.target_frames,
        "selected_total_frames": int(sum(selected_frame_counts.values())),
        "total_paired_motions": total_paired_motions,
        "num_motions": len(selected_keys),
        "smpl_schema_version": H2LORA_SEED_SMPL_SCHEMA_VERSION,
        "smpl_mode": args.smpl_mode,
        "smpl_bvh_root": str(args.smpl_bvh_root),
        "smpl_root_mode": args.smpl_root_mode,
        "robot_motion_file": str(output_root / "robot"),
        "smpl_motion_file": str(output_root / "smpl"),
        "category_motion_counts": dict(sorted(category_counts.items())),
        "selected_frame_counts": selected_frame_counts,
        "keys": selected_keys,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, default=Path("/mnt/data/seed"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps-source", type=int, default=120)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--keyword", type=str, default=None)
    parser.add_argument("--include-regex", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--selection-mode",
        choices=["sorted", "random"],
        default="sorted",
        help="sorted keeps legacy deterministic order; random shuffles paired motions uniformly.",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=None,
        help="Select full motions until this many post-resample robot frames are covered.",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--smpl-mode",
        choices=["soma-bvh", "human-fk", "smplx-fk", "zeros"],
        default="soma-bvh",
    )
    parser.add_argument(
        "--smpl-bvh-root",
        type=Path,
        default=Path("/mnt/data/seed/soma_uniform/bvh"),
        help="Root of SEED SOMA BVH clips used by --smpl-mode soma-bvh.",
    )
    parser.add_argument(
        "--smpl-root-mode",
        choices=["soma-bvh-compatible", "raw-bvh"],
        default="soma-bvh-compatible",
    )
    parser.add_argument(
        "--smpl-body-model-dir",
        type=Path,
        default=Path("/home/sky/workspace/tools/GMR/assets/body_models"),
    )
    parser.add_argument("--smpl-device", default="cpu")
    parser.add_argument("--smpl-batch-frames", type=int, default=512)
    args = parser.parse_args()

    seed_root = args.seed_root
    h2_root = seed_root / "h2/csv"
    smpl_root = seed_root / "smpl-x/uniform"
    if not h2_root.exists():
        raise FileNotFoundError(h2_root)
    if not smpl_root.exists():
        raise FileNotFoundError(smpl_root)
    if args.smpl_mode == "smplx-fk" and not (args.smpl_body_model_dir / "smplx").exists():
        raise FileNotFoundError(args.smpl_body_model_dir / "smplx")

    h2_index = _index_files(h2_root, ".csv")
    smpl_index = _index_files(smpl_root, ".npz")
    common_keys = sorted(set(h2_index) & set(smpl_index))
    filtered_keys = _select_keys(common_keys, args.include_regex, args.keyword, None)
    selected_frame_counts: dict[str, int] = {}
    if args.target_frames is not None:
        if args.target_frames <= 0:
            raise ValueError("--target-frames must be positive")
        if args.selection_mode != "random":
            raise ValueError("--target-frames currently requires --selection-mode random")
        selected_keys, selected_frame_counts = _select_keys_by_target_frames(
            filtered_keys,
            h2_index,
            args.fps_source,
            args.fps,
            args.target_frames,
            args.random_seed,
        )
        if args.limit is not None:
            selected_keys = selected_keys[: args.limit]
            selected_frame_counts = {
                key: selected_frame_counts[key] for key in selected_keys if key in selected_frame_counts
            }
    else:
        selected_keys = list(filtered_keys)
        if args.selection_mode == "random":
            rng = np.random.default_rng(args.random_seed)
            rng.shuffle(selected_keys)
        if args.limit is not None:
            selected_keys = selected_keys[: args.limit]
    if not selected_keys:
        raise ValueError("No paired H2/SMPL-X motions matched the requested filters")
    if not selected_frame_counts:
        selected_frame_counts = {
            key: _estimate_resampled_frames(
                _count_csv_data_rows(h2_index[key]),
                args.fps_source,
                args.fps,
            )
            for key in tqdm(selected_keys, desc="Counting selected frames", unit="motion")
        }

    output_root = args.output_root
    robot_out = output_root / "robot"
    smpl_out = output_root / "smpl"
    output_root.mkdir(parents=True, exist_ok=True)
    _write_manifest(
        output_root,
        selected_keys,
        args,
        selected_frame_counts=selected_frame_counts,
        total_paired_motions=len(common_keys),
    )

    tasks = [
        (
            key,
            h2_index[key],
            smpl_index[key],
            robot_out,
            smpl_out,
            args.fps_source,
            args.fps,
            args.skip_existing,
        )
        for key in selected_keys
    ]

    num_workers = max(1, min(args.num_workers, 16, len(tasks)))
    print(
        f"Building H2LoRA MotionLib: {len(tasks)} motions, fps {args.fps_source}->{args.fps}, "
        f"workers={num_workers}, smpl_mode={args.smpl_mode}, "
        f"target_frames={args.target_frames}, selected_frames={sum(selected_frame_counts.values())}"
    )
    print(f"  robot: {robot_out}")
    print(f"  smpl : {smpl_out}")

    if num_workers == 1:
        _init_worker(
            args.smpl_mode,
            str(args.smpl_body_model_dir),
            args.smpl_device,
            args.smpl_batch_frames,
            str(args.smpl_bvh_root),
            args.smpl_root_mode,
        )
        iterator = (_process_one(task) for task in tasks)
        results = list(tqdm(iterator, total=len(tasks)))
    else:
        with mp.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(
                args.smpl_mode,
                str(args.smpl_body_model_dir),
                args.smpl_device,
                args.smpl_batch_frames,
                str(args.smpl_bvh_root),
                args.smpl_root_mode,
            ),
        ) as pool:
            results = list(tqdm(pool.imap_unordered(_process_one, tasks), total=len(tasks)))

    failed = [result for result in results if not result[1]]
    if failed:
        for key, _, message in failed[:10]:
            print(f"FAILED {key}: {message}", file=sys.stderr)
        return 1

    converted = sum(1 for _, _, message in results if message == "converted")
    skipped = sum(1 for _, _, message in results if message == "skipped")
    print(f"Done: converted={converted}, skipped={skipped}, output={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
