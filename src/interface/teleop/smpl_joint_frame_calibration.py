"""Load and apply per-joint XRT-to-SMPL frame corrections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation


JOINT_COUNT = 24
DEFAULT_XRT_TO_SMPL_Y_DEGREES = 180.0


def default_joint_frame_quaternions() -> np.ndarray:
    """Return the original uniform XRT-to-SMPL correction in WXYZ order."""
    quat = Rotation.from_euler(
        "y", DEFAULT_XRT_TO_SMPL_Y_DEGREES, degrees=True
    ).as_quat(scalar_first=True)
    return np.repeat(quat[None, :], JOINT_COUNT, axis=0)


def _validated_quaternion(value, *, context: str) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError(f"{context} must be a finite WXYZ quaternion")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        raise ValueError(f"{context} must not be a zero quaternion")
    return quat / norm


def load_joint_frame_calibration(path: str | Path) -> np.ndarray:
    """Load 24 right-multiplied joint-frame corrections in WXYZ order."""
    calibration_path = Path(path)
    with calibration_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError("SMPL joint-frame calibration must be a JSON object")
    if int(payload.get("format_version", 0)) != 1:
        raise ValueError("unsupported SMPL joint-frame calibration format_version")

    default_quat = _validated_quaternion(
        payload.get(
            "default_joint_frame_quat_wxyz",
            default_joint_frame_quaternions()[0],
        ),
        context="default_joint_frame_quat_wxyz",
    )
    result = np.repeat(default_quat[None, :], JOINT_COUNT, axis=0)
    corrections = payload.get("joint_corrections", [])
    if not isinstance(corrections, list):
        raise ValueError("joint_corrections must be a list")

    seen: set[int] = set()
    for item in corrections:
        if not isinstance(item, Mapping):
            raise ValueError("each joint correction must be a JSON object")
        index = int(item.get("joint_index", -1))
        if index < 0 or index >= JOINT_COUNT:
            raise ValueError(f"joint_index must be in [0, {JOINT_COUNT - 1}], got {index}")
        if index in seen:
            raise ValueError(f"duplicate joint correction for index {index}")
        seen.add(index)
        result[index] = _validated_quaternion(
            item.get("quat_wxyz"), context=f"joint {index} quat_wxyz"
        )
    return result


def corrected_xrt_global_rotations(
    body_poses: np.ndarray,
    joint_frame_quats_wxyz: np.ndarray | None = None,
) -> Rotation:
    """Convert XRT global frames using uniform or calibrated right corrections."""
    poses = np.asarray(body_poses, dtype=np.float64)
    if poses.shape != (JOINT_COUNT, 7) or not np.isfinite(poses).all():
        raise ValueError(f"XRT body poses must have shape ({JOINT_COUNT}, 7)")
    xrt_quats_wxyz = poses[:, [6, 3, 4, 5]]
    norms = np.linalg.norm(xrt_quats_wxyz, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError("XRT body poses contain a zero quaternion")
    xrt_quats_wxyz /= norms[:, None]

    corrections = (
        default_joint_frame_quaternions()
        if joint_frame_quats_wxyz is None
        else np.asarray(joint_frame_quats_wxyz, dtype=np.float64)
    )
    if corrections.shape != (JOINT_COUNT, 4) or not np.isfinite(corrections).all():
        raise ValueError(f"joint-frame quaternions must have shape ({JOINT_COUNT}, 4)")
    correction_norms = np.linalg.norm(corrections, axis=1)
    if np.any(correction_norms < 1e-8):
        raise ValueError("joint-frame calibration contains a zero quaternion")
    corrections = corrections / correction_norms[:, None]
    return Rotation.from_quat(xrt_quats_wxyz, scalar_first=True) * Rotation.from_quat(
        corrections, scalar_first=True
    )


__all__ = [
    "DEFAULT_XRT_TO_SMPL_Y_DEGREES",
    "JOINT_COUNT",
    "corrected_xrt_global_rotations",
    "default_joint_frame_quaternions",
    "load_joint_frame_calibration",
]
