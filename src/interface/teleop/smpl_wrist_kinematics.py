"""Project SMPL wrist rotations into robot wrist joint coordinates."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from interface.math.rotation_conversion import decompose_rotation_aa

H2_SEMANTIC_WRIST_INDICES = (23, 24, 25, 26, 27, 28)

# H2 wrist limits from the MJCF used by the paired SOMA retargeting dataset.
H2_WRIST_LOWER = np.asarray([-2.618, -2.618, -0.576, -0.576, -1.22, -1.22])
H2_WRIST_UPPER = -H2_WRIST_LOWER


def _smpl_body_pose(body_pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(body_pose, dtype=np.float64)
    if pose.size != 63:
        raise ValueError(f"SMPL body pose must contain 21x3 values, got shape {pose.shape}")
    pose = pose.reshape(21, 3)
    if not np.isfinite(pose).all():
        raise ValueError("SMPL body pose contains non-finite values")
    return pose


def g1_wrist_reference_from_smpl_pose(body_pose: np.ndarray) -> np.ndarray:
    """Return the original G1 PoseStreamer 29D wrist-only reference."""
    pose = _smpl_body_pose(body_pose).reshape(1, 21, 3)
    _, left_swing = decompose_rotation_aa(pose[:, 17], np.asarray([0.0, 1.0, 0.0]))
    _, right_swing = decompose_rotation_aa(pose[:, 18], np.asarray([0.0, 1.0, 0.0]))
    left_swing_euler = Rotation.from_quat(left_swing[:, [1, 2, 3, 0]]).as_euler("XYZ")
    right_swing_euler = Rotation.from_quat(right_swing[:, [1, 2, 3, 0]]).as_euler("XYZ")
    left_wrist_euler = Rotation.from_rotvec(pose[:, 19]).as_euler("XYZ")
    right_wrist_euler = Rotation.from_rotvec(pose[:, 20]).as_euler("XYZ")

    reference = np.zeros(29, dtype=np.float32)
    reference[23] = left_swing_euler[0, 0] + left_wrist_euler[0, 0]
    reference[25] = left_wrist_euler[0, 1]
    reference[27] = left_swing_euler[0, 2] + left_wrist_euler[0, 2]
    reference[24] = -(right_swing_euler[0, 0] + right_wrist_euler[0, 0])
    reference[26] = -right_wrist_euler[0, 1]
    reference[28] = right_swing_euler[0, 2] + right_wrist_euler[0, 2]
    return reference


def h2_wrist_reference_from_smpl_pose(body_pose: np.ndarray) -> np.ndarray:
    """Return a stateless local-kinematic approximation of H2LoRA's W input.

    SMPL joints 20 and 21 are the local hand-to-forearm rotations. H2 realizes
    the same relative rotation with an intrinsic roll-pitch-yaw chain. The
    right arm uses mirrored roll and pitch coordinates in the paired H2 data.

    Training W labels came from whole-body SOMA optimization, so this local
    projection intentionally approximates rather than reproduces those labels.
    """
    pose = _smpl_body_pose(body_pose)
    left = Rotation.from_rotvec(pose[19]).as_euler("XYZ")
    right = Rotation.from_rotvec(pose[20]).as_euler("XYZ")
    wrist = np.asarray(
        [left[0], -right[0], left[1], -right[1], left[2], right[2]],
        dtype=np.float64,
    )
    wrist = np.clip(wrist, H2_WRIST_LOWER, H2_WRIST_UPPER)
    reference = np.zeros(29, dtype=np.float32)
    reference[list(H2_SEMANTIC_WRIST_INDICES)] = wrist.astype(np.float32)
    return reference
