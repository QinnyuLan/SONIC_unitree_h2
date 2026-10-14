"""H2 + Inspire feature contract shared by data conversion and inference clients."""

import numpy as np


# Keep the SONIC projector's semantic group order, but use physical H2 values.
# In particular this swaps the DDS ankle roll/pitch slots without an axis transform.
BODY_NAMES = (
    [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ]
    + [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
    + [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    ]
)
HAND_NAMES = [
    f"{side}_{joint}"
    for side in ("left", "right")
    for joint in ("pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotate")
]
STATE_NAMES = BODY_NAMES + HAND_NAMES + ["gravity_x", "gravity_y", "gravity_z"]
STATE_SLICES = {
    "left_leg": (0, 6),
    "right_leg": (6, 12),
    "waist": (12, 15),
    "left_arm": (15, 22),
    "right_arm": (22, 29),
    "left_hand": (29, 35),
    "right_hand": (35, 41),
    "projected_gravity": (41, 44),
}
ACTION_SLICES = {
    "motion_token": (0, 64),
    "left_hand_joints": (64, 70),
    "right_hand_joints": (70, 76),
}


def projected_gravity(quaternion_wxyz):
    """World down in the pelvis frame, using the lowstate IMU (wxyz)."""
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if q.shape[-1] != 4 or not np.isfinite(q).all():
        raise ValueError("Expected finite wxyz quaternions")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Zero-norm IMU quaternion")
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    return np.stack(
        (2 * (w * y - z * x), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)), axis=-1
    ).astype(np.float32)


def pack_state(raw_state, joint_names, quaternion_wxyz):
    """Remove head joints, reorder by name, preserve measured 6D hand feedback."""
    raw = np.asarray(raw_state, dtype=np.float32)
    expected = set(BODY_NAMES + HAND_NAMES + ["head_pitch_joint", "head_yaw_joint"])
    if len(joint_names) != len(expected) or set(joint_names) != expected:
        raise ValueError(
            "Raw state must name all 31 H2 joints and 12 Inspire actuators exactly once"
        )
    if raw.shape[-1] != len(joint_names) or not np.isfinite(raw).all():
        raise ValueError("Invalid raw H2 state")
    indices = [joint_names.index(name) for name in BODY_NAMES + HAND_NAMES]
    return np.concatenate((raw[..., indices], projected_gravity(quaternion_wxyz)), axis=-1)


def pack_action(tokens, raw_action, joint_names):
    """Keep executed SONIC tokens and observed DDS hand setpoints, including ramps."""
    tokens, raw = np.asarray(tokens, dtype=np.float32), np.asarray(raw_action, dtype=np.float32)
    if tokens.shape[-1] != 64 or raw.shape[-1] != len(joint_names):
        raise ValueError("Invalid H2 action dimensions")
    hands = raw[..., [joint_names.index(name) for name in HAND_NAMES]]
    if not np.isfinite(tokens).all() or not np.isfinite(hands).all():
        raise ValueError("Non-finite H2 actions")
    if np.abs(tokens).max() > 1.25 or np.any((hands < -1e-6) | (hands > 1 + 1e-6)):
        raise ValueError("H2 token/hand action outside the recorded interface bounds")
    return np.concatenate((tokens, hands), axis=-1)


def modality_metadata():
    return {
        "state": {key: {"start": start, "end": end} for key, (start, end) in STATE_SLICES.items()},
        "action": {
            key: {"start": start, "end": end} for key, (start, end) in ACTION_SLICES.items()
        },
        "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
