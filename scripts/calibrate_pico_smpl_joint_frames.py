#!/usr/bin/env python3
"""Fit constant XRT-to-SMPL upper-body joint-frame corrections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from gear_sonic.utils.teleop.smpl_joint_frame_calibration import (
    default_joint_frame_quaternions,
)
from gear_sonic.utils.teleop.xr.offline_xrt import load_tracking_data_recording


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDING = REPO_ROOT / "data/trackingData_20260807_162702.txt"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "gear_sonic/config/teleop/pico_smpl_joint_frames_20260807_162702.json"
)
DEFAULT_SMPL_JOINTS = REPO_ROOT / "gear_sonic/data/human/human_joints_info.pkl"

JOINTS = (
    ("left_shoulder", 16, 18),
    ("left_elbow", 18, 20),
    ("right_shoulder", 17, 19),
    ("right_elbow", 19, 21),
)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("cannot normalize a zero-length bone vector")
    return vector / norm


def _angles_degrees(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    cosine = np.sum(_normalize(first) * _normalize(second), axis=-1)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _rotation_from_to(first: np.ndarray, second: np.ndarray) -> Rotation:
    first = _normalize(np.asarray(first, dtype=np.float64))
    second = _normalize(np.asarray(second, dtype=np.float64))
    cross = np.cross(first, second)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.dot(first, second))
    if sine < 1e-10:
        if cosine > 0.0:
            return Rotation.identity()
        axis = np.asarray([1.0, 0.0, 0.0])
        axis -= first * float(np.dot(axis, first))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.asarray([0.0, 1.0, 0.0])
            axis -= first * float(np.dot(axis, first))
        return Rotation.from_rotvec(np.pi * _normalize(axis))
    axis = cross / sine
    return Rotation.from_rotvec(np.arctan2(sine, cosine) * axis)


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90.0)),
        "max": float(np.max(values)),
    }


def _load_rest_joints(path: Path) -> np.ndarray:
    info = torch.load(path, map_location="cpu", weights_only=False)
    joints = np.asarray(info["J"], dtype=np.float64)
    if joints.ndim == 3:
        joints = joints[0]
    if joints.shape[0] < 24 or joints.shape[1:] != (3,):
        raise ValueError(f"unexpected SMPL rest-joint shape: {joints.shape}")
    return joints[:24]


def _validation_mask(frame_count: int, fps: float, block_seconds: float) -> np.ndarray:
    block_frames = max(1, round(fps * block_seconds))
    block_index = np.arange(frame_count) // block_frames
    return block_index % 5 == 4


def _fit_joint(
    xrt_rotations: np.ndarray,
    positions: np.ndarray,
    rest_joints: np.ndarray,
    joint_index: int,
    child_index: int,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    default_correction: Rotation,
) -> tuple[Rotation, dict]:
    observed_direction = _normalize(
        positions[:, child_index] - positions[:, joint_index]
    )
    direction_in_xrt_frame = np.einsum(
        "nij,nj->ni",
        xrt_rotations[:, joint_index].transpose(0, 2, 1),
        observed_direction,
    )
    mean_local_direction = np.mean(direction_in_xrt_frame[train_mask], axis=0)
    concentration = float(np.linalg.norm(mean_local_direction))
    target_local_direction = _normalize(mean_local_direction)

    rest_direction = _normalize(rest_joints[child_index] - rest_joints[joint_index])
    default_local_direction = default_correction.apply(rest_direction)
    delta = _rotation_from_to(default_local_direction, target_local_direction)
    correction = delta * default_correction

    default_prediction = np.einsum(
        "nij,j->ni",
        xrt_rotations[:, joint_index],
        default_local_direction,
    )
    calibrated_prediction = np.einsum(
        "nij,j->ni",
        xrt_rotations[:, joint_index],
        correction.apply(rest_direction),
    )
    default_error = _angles_degrees(
        default_prediction[validation_mask], observed_direction[validation_mask]
    )
    calibrated_error = _angles_degrees(
        calibrated_prediction[validation_mask], observed_direction[validation_mask]
    )
    return correction, {
        "train_local_direction_concentration": concentration,
        "delta_from_original_degrees": float(delta.magnitude() * 180.0 / np.pi),
        "validation_direction_error_degrees": {
            "original": _stats(default_error),
            "calibrated": _stats(calibrated_error),
        },
    }


def _elbow_report(
    positions: np.ndarray,
    xrt_rotations: np.ndarray,
    rest_joints: np.ndarray,
    corrections: dict[int, Rotation],
    validation_mask: np.ndarray,
    default_correction: Rotation,
) -> dict:
    result = {}
    for side, shoulder, elbow, wrist in (
        ("left", 16, 18, 20),
        ("right", 17, 19, 21),
    ):
        raw_angle = _angles_degrees(
            positions[:, shoulder] - positions[:, elbow],
            positions[:, wrist] - positions[:, elbow],
        )
        upper_rest = _normalize(rest_joints[elbow] - rest_joints[shoulder])
        forearm_rest = _normalize(rest_joints[wrist] - rest_joints[elbow])

        def predicted_angle(shoulder_correction: Rotation, elbow_correction: Rotation):
            upper = np.einsum(
                "nij,j->ni",
                xrt_rotations[:, shoulder],
                shoulder_correction.apply(upper_rest),
            )
            forearm = np.einsum(
                "nij,j->ni",
                xrt_rotations[:, elbow],
                elbow_correction.apply(forearm_rest),
            )
            return _angles_degrees(-upper, forearm)

        original_angle = predicted_angle(default_correction, default_correction)
        calibrated_angle = predicted_angle(
            corrections[shoulder], corrections[elbow]
        )
        selected = validation_mask
        result[side] = {
            "raw_xrt": _stats(raw_angle[selected]),
            "original_fk": _stats(original_angle[selected]),
            "calibrated_fk": _stats(calibrated_angle[selected]),
            "original_absolute_error": _stats(
                np.abs(original_angle[selected] - raw_angle[selected])
            ),
            "calibrated_absolute_error": _stats(
                np.abs(calibrated_angle[selected] - raw_angle[selected])
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, default=DEFAULT_RECORDING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smpl-joints", type=Path, default=DEFAULT_SMPL_JOINTS)
    parser.add_argument("--validation-block-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.validation_block_seconds <= 0.0:
        parser.error("--validation-block-seconds must be positive")

    recording = load_tracking_data_recording(args.recording)
    body_poses = np.asarray(recording["body_joints_pose"], dtype=np.float64)
    positions = body_poses[:, :, :3]
    xrt_rotations = Rotation.from_quat(
        body_poses[:, :, 3:].reshape(-1, 4)
    ).as_matrix().reshape(body_poses.shape[0], 24, 3, 3)
    rest_joints = _load_rest_joints(args.smpl_joints)
    fps = float(recording["fps"])
    validation_mask = _validation_mask(
        body_poses.shape[0], fps, args.validation_block_seconds
    )
    train_mask = ~validation_mask
    if not np.any(train_mask) or not np.any(validation_mask):
        raise ValueError("recording is too short for the requested train/validation split")

    default_quat = default_joint_frame_quaternions()[0]
    default_correction = Rotation.from_quat(default_quat, scalar_first=True)
    corrections: dict[int, Rotation] = {}
    joint_metrics = {}
    joint_entries = []
    for name, joint_index, child_index in JOINTS:
        correction, metrics = _fit_joint(
            xrt_rotations,
            positions,
            rest_joints,
            joint_index,
            child_index,
            train_mask,
            validation_mask,
            default_correction,
        )
        corrections[joint_index] = correction
        joint_metrics[name] = metrics
        joint_entries.append(
            {
                "joint_name": name,
                "joint_index": joint_index,
                "child_index": child_index,
                "quat_wxyz": correction.as_quat(scalar_first=True).tolist(),
            }
        )

    payload = {
        "format_version": 1,
        "description": "Per-joint right corrections: R_smpl_global = R_xrt_global * C_joint",
        "source_recording": str(args.recording.resolve()),
        "frame_count": int(body_poses.shape[0]),
        "fps": fps,
        "split": {
            "method": "four temporal blocks train, one temporal block validation",
            "block_seconds": args.validation_block_seconds,
            "train_frames": int(np.sum(train_mask)),
            "validation_frames": int(np.sum(validation_mask)),
        },
        "default_joint_frame_quat_wxyz": default_quat.tolist(),
        "joint_corrections": joint_entries,
        "metrics": {
            "joint_direction": joint_metrics,
            "elbow_angle_degrees": _elbow_report(
                positions,
                xrt_rotations,
                rest_joints,
                corrections,
                validation_mask,
                default_correction,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    for name, metrics in joint_metrics.items():
        original = metrics["validation_direction_error_degrees"]["original"]
        calibrated = metrics["validation_direction_error_degrees"]["calibrated"]
        print(
            f"{name}: mean {original['mean']:.2f}->{calibrated['mean']:.2f} deg, "
            f"p90 {original['p90']:.2f}->{calibrated['p90']:.2f} deg"
        )


if __name__ == "__main__":
    main()
