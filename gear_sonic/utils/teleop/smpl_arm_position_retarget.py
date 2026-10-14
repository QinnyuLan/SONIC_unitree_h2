"""Constrain SMPL arm bone directions with XR full-body joint positions."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

XRT_TO_ROBOT = np.asarray(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)

SMPL_ARM_CHAINS = (
    (16, 18, 20, 22),  # left shoulder, elbow, wrist, hand
    (17, 19, 21, 23),  # right shoulder, elbow, wrist, hand
)


def _validated_body_poses(body_poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(body_poses, dtype=np.float64)
    if poses.shape != (24, 7):
        raise ValueError(f"XRT body poses must have shape (24, 7), got {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError("XRT body poses contain non-finite values")
    if np.linalg.norm(poses[0, 3:]) < 1e-8:
        raise ValueError("XRT pelvis quaternion is zero")
    return poses


def xrt_joint_positions_in_pelvis_frame(body_poses: np.ndarray) -> np.ndarray:
    """Return XRT joint positions in a robot-axis, pelvis-local frame."""
    poses = _validated_body_poses(body_poses)
    root_relative_robot = (XRT_TO_ROBOT @ (poses[:, :3] - poses[0, :3]).T).T
    pelvis_rotation_robot = XRT_TO_ROBOT @ Rotation.from_quat(poses[0, 3:]).as_matrix() @ XRT_TO_ROBOT.T
    return (pelvis_rotation_robot.T @ root_relative_robot.T).T


def _unit_direction(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm >= 1e-8:
        return vector / norm
    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm < 1e-8:
        raise ValueError("SMPL arm contains a zero-length bone")
    return fallback / fallback_norm


def retarget_smpl_arms_from_xrt_positions(
    body_poses: np.ndarray,
    smpl_joints_local: np.ndarray,
) -> np.ndarray:
    """Replace SMPL arm directions while preserving the fixed SMPL bone lengths.

    XRT rotations and SMPL local joint frames do not share a reliable arm-axis
    convention. XRT positions do share the required geometric information, so
    shoulder-to-elbow and elbow-to-wrist directions are taken from positions.
    The hand joint continues the forearm direction to avoid controller/hand-frame
    noise at joints 22 and 23.
    """
    joints = np.asarray(smpl_joints_local, dtype=np.float64)
    if joints.shape != (24, 3):
        raise ValueError(f"SMPL joints must have shape (24, 3), got {joints.shape}")
    if not np.isfinite(joints).all():
        raise ValueError("SMPL joints contain non-finite values")

    xrt_local = xrt_joint_positions_in_pelvis_frame(body_poses)
    corrected = joints.copy()
    for shoulder, elbow, wrist, hand in SMPL_ARM_CHAINS:
        upper_fallback = joints[elbow] - joints[shoulder]
        forearm_fallback = joints[wrist] - joints[elbow]
        upper_direction = _unit_direction(xrt_local[elbow] - xrt_local[shoulder], upper_fallback)
        forearm_direction = _unit_direction(xrt_local[wrist] - xrt_local[elbow], forearm_fallback)

        upper_length = float(np.linalg.norm(upper_fallback))
        forearm_length = float(np.linalg.norm(forearm_fallback))
        hand_length = float(np.linalg.norm(joints[hand] - joints[wrist]))
        corrected[elbow] = corrected[shoulder] + upper_length * upper_direction
        corrected[wrist] = corrected[elbow] + forearm_length * forearm_direction
        corrected[hand] = corrected[wrist] + hand_length * forearm_direction

    return corrected.astype(np.float32)


def smpl_joints_world_from_local(smpl_joints_local: np.ndarray, smpl_root_quat_wxyz: np.ndarray) -> np.ndarray:
    """Undo deploy's root-localization input transform for streamed SMPL data."""
    joints = np.asarray(smpl_joints_local, dtype=np.float64)
    root_quat = np.asarray(smpl_root_quat_wxyz, dtype=np.float64)
    if joints.shape != (24, 3):
        raise ValueError(f"SMPL joints must have shape (24, 3), got {joints.shape}")
    if root_quat.shape != (4,):
        raise ValueError(f"SMPL root quaternion must have shape (4,), got {root_quat.shape}")
    if not np.isfinite(joints).all() or not np.isfinite(root_quat).all():
        raise ValueError("SMPL joint transform contains non-finite values")
    if np.linalg.norm(root_quat) < 1e-8:
        raise ValueError("SMPL root quaternion is zero")
    return Rotation.from_quat(root_quat, scalar_first=True).apply(joints).astype(np.float32)


__all__ = [
    "SMPL_ARM_CHAINS",
    "XRT_TO_ROBOT",
    "retarget_smpl_arms_from_xrt_positions",
    "smpl_joints_world_from_local",
    "xrt_joint_positions_in_pelvis_frame",
]
