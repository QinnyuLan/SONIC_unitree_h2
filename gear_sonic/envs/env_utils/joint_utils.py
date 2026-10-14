"""Joint utility functions and constants for humanoid robots.

This module provides joint ordering constants and helper functions for mapping
between motion library data and robot joints.
"""

from __future__ import annotations

import torch

# G1 body joint names in IsaacLab order (29 DOF)
G1_ISAACLab_ORDER = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

# G1 hand joint names (14 DOF) - order from g1_43dof.yaml
G1_HAND_JOINTS = [
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]

H2_HEAD_JOINTS = [
    "head_pitch_joint",
    "head_yaw_joint",
]

# H2 body joints, ordered in the 29-DoF semantic convention expected by the
# released G1 SONIC policy.  H2's physical ankle order is roll->pitch in the
# MJCF/URDF, so this list intentionally keeps the G1 policy order.
H2LORA_BODY_JOINTS_IN_G1_ORDER = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

# Indices into the full 31-DoF H2 IsaacLab joint order that produce the
# headless 29-DoF order above.  This strips head_pitch/head_yaw and also fixes
# the H2 ankle roll/pitch placement to match the released G1 policy interface.
H2LORA_FULL_H2_ISAACLAB_DOF_INDICES = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    12,
    13,
    19,
    20,
    17,
    18,
    14,
    15,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
]

HIP_JOINT_PAIRS = {
    "left": ("left_hip_pitch_joint", "left_hip_roll_joint"),
    "right": ("right_hip_pitch_joint", "right_hip_roll_joint"),
}

# Caches for joint indices
_body_joint_indices_cache = {}
_hand_joint_indices_cache = {}
_head_joint_indices_cache = {}


def _get_joint_indices_by_names(asset, joint_names: list, cache: dict) -> torch.Tensor:
    """Get indices of specified joints in the robot's joint list."""
    cache_key = (id(asset), tuple(joint_names))
    if cache_key in cache:
        return cache[cache_key]

    robot_joint_names = asset.joint_names
    indices = [robot_joint_names.index(n) for n in joint_names if n in robot_joint_names]
    indices_tensor = torch.tensor(indices, dtype=torch.long, device=asset.device)
    cache[cache_key] = indices_tensor
    return indices_tensor


def get_joint_indices_by_names(asset, joint_names: list[str], *, strict: bool = True) -> torch.Tensor:
    """Get joint indices by name in the asset's native order.

    Args:
        asset: IsaacLab articulation asset.
        joint_names: Joint names to look up.
        strict: If True, raise when any requested joint is absent.

    Returns:
        Tensor of indices on ``asset.device``.
    """
    robot_joint_names = asset.joint_names
    missing = [name for name in joint_names if name not in robot_joint_names]
    if missing and strict:
        raise ValueError(f"Missing joints in asset '{getattr(asset, 'name', 'robot')}': {missing}")
    indices = [robot_joint_names.index(name) for name in joint_names if name in robot_joint_names]
    return torch.tensor(indices, dtype=torch.long, device=asset.device)


def get_body_joint_indices(asset) -> torch.Tensor:
    """Get indices of body joints (29 DOF) using G1_ISAACLab_ORDER."""
    return _get_joint_indices_by_names(asset, G1_ISAACLab_ORDER, _body_joint_indices_cache)


def get_hand_joint_indices(asset) -> torch.Tensor:
    """Get indices of hand joints (14 DOF) using G1_HAND_JOINTS."""
    return _get_joint_indices_by_names(asset, G1_HAND_JOINTS, _hand_joint_indices_cache)


def get_h2lora_body_joint_indices(asset) -> torch.Tensor:
    """Get H2 body joint indices in the G1 policy's 29-DoF order."""
    return get_joint_indices_by_names(asset, H2LORA_BODY_JOINTS_IN_G1_ORDER, strict=True)


def get_h2_head_joint_indices(asset) -> torch.Tensor:
    """Get H2 head joint indices for direct default command injection."""
    return _get_joint_indices_by_names(asset, H2_HEAD_JOINTS, _head_joint_indices_cache)
