#!/usr/bin/env python3
"""Generate a mode-0 H2 reference with both arms straight forward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.data.robot_model.instantiation.h2 import (  # noqa: E402
    H2_DEFAULT_ANGLES_LOCAL,
    h2_configuration_from_local_joints,
    instantiate_h2_robot_model,
)
from scripts.export_h2lora_deploy_reference import (  # noqa: E402
    H2_MJCF_JOINT_NAMES,
    _h2_physical_to_g1_semantic,
    _write_deploy_csv_bundle,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "gear_sonic_deploy/reference/h2_standing_arms_forward_mode0"
MOTION_NAME = "h2_stand_arms_forward_straight"
FPS = 50.0

# Solved against H2_loop.urdf. Both shoulder-to-wrist chains point along +X,
# remain at constant height, and have a 180-degree internal elbow angle.
STRAIGHT_ARM_JOINTS = {
    "left_shoulder_pitch_joint": -1.56279221,
    "left_shoulder_roll_joint": 0.10740987,
    "left_shoulder_yaw_joint": -2.20692140,
    "left_elbow_joint": 1.64762524,
    "right_shoulder_pitch_joint": -1.56279221,
    "right_shoulder_roll_joint": -0.10740987,
    "right_shoulder_yaw_joint": 2.20692140,
    "right_elbow_joint": 1.64762524,
}

ARM_CHAINS = {
    "left": ("left_shoulder_pitch_link", "left_elbow_link", "left_wrist_yaw_link"),
    "right": ("right_shoulder_pitch_link", "right_elbow_link", "right_wrist_yaw_link"),
}


def _internal_angle_degrees(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
    upper = shoulder - elbow
    lower = wrist - elbow
    cosine = np.dot(upper, lower) / (np.linalg.norm(upper) * np.linalg.norm(lower))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def build_reference_pose() -> tuple[np.ndarray, dict[str, dict[str, object]]]:
    joints = H2_DEFAULT_ANGLES_LOCAL.copy()
    for name, value in STRAIGHT_ARM_JOINTS.items():
        joints[H2_MJCF_JOINT_NAMES.index(name)] = value

    model = instantiate_h2_robot_model()
    configuration = h2_configuration_from_local_joints(model, joints)
    model.cache_forward_kinematics(configuration, auto_clip=False)

    geometry: dict[str, dict[str, object]] = {}
    for side, frame_names in ARM_CHAINS.items():
        shoulder, elbow, wrist = [
            model.frame_placement(frame_name).translation.copy() for frame_name in frame_names
        ]
        direction = wrist - shoulder
        direction /= np.linalg.norm(direction)
        elbow_angle = _internal_angle_degrees(shoulder, elbow, wrist)
        if not np.allclose(direction, [1.0, 0.0, 0.0], atol=1e-6):
            raise RuntimeError(f"{side} arm direction is not +X: {direction}")
        if abs(elbow_angle - 180.0) > 1e-5:
            raise RuntimeError(f"{side} elbow is not straight: {elbow_angle:.8f} deg")
        geometry[side] = {
            "shoulder_xyz": shoulder.tolist(),
            "elbow_xyz": elbow.tolist(),
            "wrist_xyz": wrist.tolist(),
            "shoulder_to_wrist_direction": direction.tolist(),
            "elbow_internal_angle_deg": elbow_angle,
        }
    return joints.astype(np.float32), geometry


def generate(output_dir: Path, frames: int) -> Path:
    if frames < 46:
        raise ValueError("mode-0 10-frame lookahead requires at least 46 frames")

    h2_joints, geometry = build_reference_pose()
    h2_joint_pos = np.repeat(h2_joints[None, :], frames, axis=0)
    h2_joint_vel = np.zeros_like(h2_joint_pos)
    joint_pos = _h2_physical_to_g1_semantic(h2_joint_pos, H2_MJCF_JOINT_NAMES)
    joint_vel = np.zeros_like(joint_pos)
    root_pos_single = np.asarray([0.0, 0.0, 1.04], dtype=np.float32)
    root_quat_single = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    root_pos = np.repeat(root_pos_single[None, :], frames, axis=0)
    root_quat = np.repeat(root_quat_single[None, :], frames, axis=0)

    motion_dir = output_dir / MOTION_NAME
    motion_dir.mkdir(parents=True, exist_ok=True)
    _write_deploy_csv_bundle(
        motion_dir=motion_dir,
        motion_name=MOTION_NAME,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        h2_joint_pos=h2_joint_pos,
        h2_joint_vel=h2_joint_vel,
        root_pos=root_pos,
        root_quat_wxyz=root_quat,
        fps=FPS,
        metadata_lines=[
            "Encoder mode: 0",
            "Deploy source: synthetic_h2_native_joint_reference",
            "Robot profile: h2_native",
            "Reference joint space: h2_physical_31d_mujoco",
            "Synthetic reference: true",
            "Arm target: shoulder-to-wrist +X, elbow internal angle 180 deg",
            "SMPL data: none",
        ],
    )

    info = {
        "motion_name": MOTION_NAME,
        "source_key": MOTION_NAME,
        "source_robot_pkl": None,
        "source_smpl_pkl": None,
        "fps": int(FPS),
        "frames": frames,
        "encoder_mode": 0,
        "robot_profile": "h2_native",
        "joint_space": "g1_semantic_29d_from_h2_physical_headless",
        "source_joint_space": "h2_physical_31d_mujoco",
        "joint_pos_shape": list(joint_pos.shape),
        "h2_joint_pos_shape": list(h2_joint_pos.shape),
        "body_quat_format": "wxyz",
        "body_count": 1,
        "smpl": None,
        "h2_joint_angles": h2_joints.tolist(),
        "root_pos": root_pos_single.tolist(),
        "root_quat_wxyz": root_quat_single.tolist(),
        "geometry": geometry,
        "note": "Static H2-native mode-0 reference; no SMPL encoder input is present.",
    }
    (motion_dir / "h2_lora_reference.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema": "h2_lora_deploy_reference_v1",
        "description": "Static H2-native mode-0 standing reference with straight forward arms.",
        "motion_root": None,
        "smpl_root": None,
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "fps": int(FPS),
        "num_motions": 1,
        "total_frames": frames,
        "motions": [info],
    }
    (output_dir / "h2_lora_reference_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "motion_summary.txt").write_text(
        "\n".join(
            [
                "H2-native Direct Reference Summary",
                "==================================",
                "Encoder mode: 0 (h2_native)",
                "Motions: 1",
                f"Total frames: {frames}",
                "",
                f"- {MOTION_NAME}: static stand, straight arms along +X",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return motion_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames", type=int, default=1000)
    args = parser.parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    motion_dir = generate(output_dir, args.frames)
    print(f"Generated {motion_dir}")


if __name__ == "__main__":
    main()
