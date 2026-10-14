#!/usr/bin/env python3
"""Generate a synthetic VR3Point replay for the H2 straight-arm reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.data.robot_model.instantiation.h2 import (  # noqa: E402
    H2_DEFAULT_ANGLES_LOCAL,
    h2_configuration_from_local_joints,
    instantiate_h2_robot_model,
)
from scripts.make_h2_standing_arms_forward_reference import (  # noqa: E402
    build_reference_pose,
)

DEFAULT_OUTPUT = REPO_ROOT / "data/h2_vr3pt_arms_forward_straight.npz"
XRT_TO_ROBOT = np.asarray(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
WRIST_OFFSETS = {
    "left": np.asarray([0.18, -0.025, 0.0], dtype=np.float64),
    "right": np.asarray([0.18, 0.025, 0.0], dtype=np.float64),
}


def h2_vr3_pose(joint_values: np.ndarray) -> np.ndarray:
    """Return left hand, right hand, and head targets in robot coordinates."""
    model = instantiate_h2_robot_model()
    configuration = h2_configuration_from_local_joints(model, joint_values)
    model.cache_forward_kinematics(configuration, auto_clip=False)

    result = np.zeros((3, 7), dtype=np.float64)
    for row, side in enumerate(("left", "right")):
        placement = model.frame_placement(f"{side}_wrist_yaw_link")
        result[row, :3] = placement.translation + placement.rotation @ WRIST_OFFSETS[side]
        result[row, 3:] = Rotation.from_matrix(placement.rotation).as_quat(scalar_first=True)

    torso = model.frame_placement("torso_link")
    result[2, :3] = torso.translation + torso.rotation @ np.asarray([0.0, 0.0, 0.35])
    result[2, 3:] = Rotation.from_matrix(torso.rotation).as_quat(scalar_first=True)
    return result


def robot_pose_to_xrt(pose_wxyz: np.ndarray) -> np.ndarray:
    """Invert the manager's XRT-to-robot three-point coordinate transform."""
    pose = np.asarray(pose_wxyz, dtype=np.float64)
    if pose.shape != (3, 7):
        raise ValueError(f"robot VR3 pose must have shape (3, 7), got {pose.shape}")
    result = np.zeros((3, 7), dtype=np.float64)
    for index in range(3):
        result[index, :3] = XRT_TO_ROBOT.T @ pose[index, :3]
        rotation_robot = Rotation.from_quat(pose[index, 3:], scalar_first=True).as_matrix()
        rotation_xrt = XRT_TO_ROBOT.T @ rotation_robot @ XRT_TO_ROBOT
        result[index, 3:] = Rotation.from_matrix(rotation_xrt).as_quat()
    return result.astype(np.float32)


def _interpolate_pose(start: np.ndarray, end: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    output = np.zeros((alpha.size, 3, 7), dtype=np.float64)
    output[:, :, :3] = (1.0 - alpha[:, None, None]) * start[None, :, :3] + alpha[:, None, None] * end[None, :, :3]
    for point in range(3):
        rotations = Rotation.from_quat(np.stack([start[point, 3:], end[point, 3:]]), scalar_first=True)
        output[:, point, 3:] = Slerp([0.0, 1.0], rotations)(alpha).as_quat(scalar_first=True)
    return output


def generate_arrays(
    fps: float = 50.0,
    neutral_seconds: float = 2.0,
    transition_seconds: float = 2.0,
    hold_seconds: float = 16.0,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    durations = (neutral_seconds, transition_seconds, hold_seconds)
    if any(duration <= 0.0 for duration in durations):
        raise ValueError("all phase durations must be positive")

    straight_joints, geometry = build_reference_pose()
    neutral_pose = h2_vr3_pose(H2_DEFAULT_ANGLES_LOCAL)
    target_pose = h2_vr3_pose(straight_joints)

    neutral_frames = round(neutral_seconds * fps)
    transition_frames = round(transition_seconds * fps)
    hold_frames = round(hold_seconds * fps)
    smooth_alpha = np.linspace(0.0, 1.0, transition_frames, dtype=np.float64)
    smooth_alpha = smooth_alpha * smooth_alpha * (3.0 - 2.0 * smooth_alpha)
    robot_poses = np.concatenate(
        [
            np.repeat(neutral_pose[None, :, :], neutral_frames, axis=0),
            _interpolate_pose(neutral_pose, target_pose, smooth_alpha),
            np.repeat(target_pose[None, :, :], hold_frames, axis=0),
        ],
        axis=0,
    )
    xrt_poses = np.stack([robot_pose_to_xrt(pose) for pose in robot_poses])
    frame_count = robot_poses.shape[0]
    timestamps_ns = np.rint(np.arange(frame_count) * 1e9 / fps).astype(np.int64)
    arrays = {
        "left_controller_pose": xrt_poses[:, 0],
        "right_controller_pose": xrt_poses[:, 1],
        "headset_pose": xrt_poses[:, 2],
        "timestamps_ns": timestamps_ns,
        "fps": np.asarray(fps, dtype=np.float64),
    }
    metadata = {
        "description": "H2 VR3Point neutral-to-horizontal-straight-arms replay",
        "fps": fps,
        "frame_count": frame_count,
        "phases_seconds": {
            "neutral": neutral_seconds,
            "transition": transition_seconds,
            "hold": hold_seconds,
        },
        "encoder_mode": 1,
        "neutral_robot_vr3_pose_wxyz": neutral_pose.tolist(),
        "target_robot_vr3_pose_wxyz": target_pose.tolist(),
        "source_h2_geometry": geometry,
    }
    return arrays, metadata


def generate(output: Path, **kwargs) -> Path:
    arrays, metadata = generate_arrays(**kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--neutral-seconds", type=float, default=2.0)
    parser.add_argument("--transition-seconds", type=float, default=2.0)
    parser.add_argument("--hold-seconds", type=float, default=16.0)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    generated = generate(
        output,
        fps=args.fps,
        neutral_seconds=args.neutral_seconds,
        transition_seconds=args.transition_seconds,
        hold_seconds=args.hold_seconds,
    )
    print(f"Generated {generated}")


if __name__ == "__main__":
    main()
