#!/usr/bin/env python3
"""Generate internally consistent static SMPL mode-2 arm references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from export_h2lora_deploy_reference import (  # noqa: E402
    _csv_headers_3d,
    _csv_headers_quat,
    _write_csv,
    _write_deploy_csv_bundle,
)

from gear_sonic.isaac_utils.rotations import (  # noqa: E402
    remove_smpl_base_rot,
    smpl_root_ytoz_up,
)
from gear_sonic.trl.utils.torch_transform import (  # noqa: E402
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
)
from gear_sonic.utils.teleop.smpl_wrist_kinematics import (  # noqa: E402
    g1_wrist_reference_from_smpl_pose,
    h2_wrist_reference_from_smpl_pose,
)

FPS = 50.0
DEFAULT_TEMPLATE_MOTION = REPO_ROOT / "gear_sonic_deploy/reference/example_h2/deploy/forward_lunge_R_001__A359_M"
ARM_CHAINS = {
    "left": (16, 18, 20, 22),
    "right": (17, 19, 21, 23),
}
ARM_POSE_JOINTS = (12, 13, 15, 16, 17, 18, 19, 20)

# Each row is a local SMPL body-joint rotation in axis-angle form. These poses
# were solved against compute_human_joints, so the shoulder-elbow-wrist-thumb
# chains remain on the SMPL kinematic manifold instead of editing FK points.
POSES = {
    "arms_overhead_straight": {
        "target": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        "arm_pose": np.asarray(
            [
                [-0.80634307, -0.51853900, 0.67400975],
                [-0.68066769, 0.49045187, -0.67016682],
                [-0.51688297, -0.75027313, 0.72629821],
                [-0.56945002, 0.68840384, -0.65788667],
                [-0.74809281, 0.21261845, -0.28525890],
                [-0.69888823, -0.08688992, 0.05516129],
                [-0.36933705, 0.71825324, 0.19627866],
                [-0.34528322, -0.80654219, -0.12721563],
            ],
            dtype=np.float32,
        ),
    },
    "arms_forward_straight": {
        "target": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "arm_pose": np.asarray(
            [
                [-0.69449792, -0.55736598, -0.39674687],
                [-0.66595861, 0.61066047, 0.43102503],
                [-0.69238727, -0.59323792, -0.46828853],
                [-0.59378934, 0.60060470, 0.53593889],
                [1.08026303, 0.03669404, -0.42240805],
                [0.12752509, -0.02541212, 0.12533401],
                [0.08658409, 0.69166045, 0.36104367],
                [0.11610777, -0.74622555, -0.28396138],
            ],
            dtype=np.float32,
        ),
    },
}


def _load_csv_frame(path: Path, width: int, frame: int) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != width:
        raise ValueError(f"Expected {width} columns in {path}, got {values.shape[1]}")
    if not 0 <= frame < values.shape[0]:
        raise IndexError(f"Template frame {frame} is outside {path} with {values.shape[0]} frames")
    return values[frame].copy()


def _load_robot_template(motion_dir: Path, frame: int) -> dict[str, np.ndarray]:
    required = {
        "h2_joint_pos": ("h2_joint_pos.csv", 31),
        "root_pos": ("body_pos.csv", 3),
    }
    missing = [filename for filename, _ in required.values() if not (motion_dir / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"Template motion {motion_dir} is missing: {', '.join(missing)}")
    return {
        name: _load_csv_frame(motion_dir / filename, width, frame) for name, (filename, width) in required.items()
    }


def _runtime_root_frames() -> tuple[torch.Tensor, torch.Tensor]:
    root_y_up = angle_axis_to_quaternion(torch.zeros((1, 3), dtype=torch.float32))
    root_z_up = smpl_root_ytoz_up(root_y_up)
    root_runtime = remove_smpl_base_rot(root_z_up, w_last=False)
    return root_z_up, root_runtime


def _wrist_targets_from_smpl_pose(body_pose: np.ndarray) -> np.ndarray:
    """Match PoseStreamer wrist conversion in pico_manager_thread_server.py."""
    return g1_wrist_reference_from_smpl_pose(body_pose)


def _pose_and_geometry(pose_spec: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    body_pose = np.zeros((21, 3), dtype=np.float32)
    body_pose[np.asarray(ARM_POSE_JOINTS)] = pose_spec["arm_pose"]
    root_z_up, root_runtime = _runtime_root_frames()

    with torch.no_grad():
        body_pose_tensor = torch.from_numpy(body_pose.reshape(1, 63))
        global_orient = quaternion_to_angle_axis(root_z_up)
        joints_raw_tensor = compute_human_joints(body_pose_tensor, global_orient)
        joints_raw_tensor = joints_raw_tensor - joints_raw_tensor[:, :1, :].clone()
        joints_local_tensor = quat_apply(
            quat_inv(root_runtime).unsqueeze(1).repeat(1, 24, 1),
            joints_raw_tensor,
        )

    joints_raw = joints_raw_tensor[0].cpu().numpy().astype(np.float32)
    joints_local = joints_local_tensor[0].cpu().numpy().astype(np.float32)
    target = pose_spec["target"]
    geometry: dict[str, object] = {
        "target_direction_local": target.tolist(),
        "smpl_root_quat_wxyz": root_runtime[0].cpu().numpy().tolist(),
        "sides": {},
    }

    for side, chain in ARM_CHAINS.items():
        directions = []
        lengths = []
        for start, end in zip(chain[:-1], chain[1:], strict=True):
            vector = joints_local[end] - joints_local[start]
            length = float(np.linalg.norm(vector))
            direction = vector / length
            if np.linalg.norm(direction - target) > 5e-4:
                raise RuntimeError(f"{side} segment {start}->{end} misses target: {direction}")
            directions.append(direction.tolist())
            lengths.append(length)

        shoulder, elbow, wrist, thumb = chain
        upper = joints_local[shoulder] - joints_local[elbow]
        lower = joints_local[wrist] - joints_local[elbow]
        cosine = np.dot(upper, lower) / (np.linalg.norm(upper) * np.linalg.norm(lower))
        elbow_angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        if elbow_angle < 179.9:
            raise RuntimeError(f"{side} elbow is not straight: {elbow_angle:.4f} deg")
        geometry["sides"][side] = {
            "segment_directions": directions,
            "segment_lengths_m": lengths,
            "elbow_angle_deg": elbow_angle,
            "shoulder_local": joints_local[shoulder].tolist(),
            "elbow_local": joints_local[elbow].tolist(),
            "wrist_local": joints_local[wrist].tolist(),
            "thumb_local": joints_local[thumb].tolist(),
        }

    wrist_targets = _wrist_targets_from_smpl_pose(body_pose)
    return body_pose, joints_raw, wrist_targets, geometry


def generate(
    output_dir: Path,
    frames: int,
    template_motion: Path,
    template_frame: int,
    robot_profile: str = "h2_lora",
) -> dict:
    if frames < 10:
        raise ValueError("frames must be at least 10 for the SMPL mode-2 observation")
    if robot_profile not in {"g1", "h2_lora"}:
        raise ValueError(f"Unsupported robot profile: {robot_profile}")
    output_dir.mkdir(parents=True, exist_ok=True)

    template_motion = template_motion.resolve()
    template = _load_robot_template(template_motion, template_frame)
    _, root_runtime_tensor = _runtime_root_frames()
    root_runtime = root_runtime_tensor[0].cpu().numpy().astype(np.float32)

    motions = []
    geometry_report = {}
    for motion_name, pose_spec in POSES.items():
        motion_dir = output_dir / motion_name
        motion_dir.mkdir(parents=True, exist_ok=True)
        body_pose, smpl_joints_raw, wrist_target, geometry = _pose_and_geometry(pose_spec)
        geometry_report[motion_name] = geometry
        if robot_profile == "h2_lora":
            wrist_target = h2_wrist_reference_from_smpl_pose(body_pose)
            template_h2 = template["h2_joint_pos"].copy()
            template_h2[[21, 28, 22, 29, 23, 30]] = wrist_target[[23, 24, 25, 26, 27, 28]]
            h2_joint_pos = np.repeat(template_h2[None, :], frames, axis=0)
            root_pos = np.repeat(template["root_pos"][None, :], frames, axis=0)
            body_root_quat = np.repeat(root_runtime[None, :], frames, axis=0)
            reference_joint_space = "h2_local_kinematic_wrist_in_g1_semantic_29d"
            wrist_contract = "stateless SMPL local wrist to H2 roll-pitch-yaw projection"
        else:
            h2_joint_pos = np.repeat(template["h2_joint_pos"][None, :], frames, axis=0)
            root_pos = np.repeat(template["root_pos"][None, :], frames, axis=0)
            body_root_quat = np.repeat(root_runtime[None, :], frames, axis=0)
            reference_joint_space = "g1_semantic_29d_vr_wrist_targets_only"
            wrist_contract = "original G1 PoseStreamer FK and wrist conversion"
        joint_pos = np.repeat(wrist_target[None, :], frames, axis=0)

        _write_deploy_csv_bundle(
            motion_dir=motion_dir,
            motion_name=motion_name,
            joint_pos=joint_pos,
            joint_vel=np.zeros_like(joint_pos),
            h2_joint_pos=h2_joint_pos,
            h2_joint_vel=np.zeros_like(h2_joint_pos),
            root_pos=root_pos,
            root_quat_wxyz=body_root_quat,
            fps=FPS,
            metadata_lines=[
                "Encoder mode: 2",
                "Deploy source: synthetic_static_smpl_fk",
                f"Robot profile: {robot_profile}",
                f"Reference joint space: {reference_joint_space}",
                "Synthetic reference: true",
                "SMPL stored frame: z_up_pelvis_centered_world_oriented_no_translation",
                f"SMPL local target direction: {pose_spec['target'].tolist()}",
                f"SMPL/wrist contract: {wrist_contract}",
            ],
        )

        smpl_pose = np.repeat(body_pose[None, :, :], frames, axis=0)
        smpl_joints = np.repeat(smpl_joints_raw[None, :, :], frames, axis=0)
        smpl_roots = np.repeat(root_runtime[None, :], frames, axis=0)
        _write_csv(
            motion_dir / "smpl_pose.csv",
            smpl_pose.reshape(frames, -1),
            _csv_headers_3d("smpl_pose", 21),
        )
        _write_csv(
            motion_dir / "smpl_joint.csv",
            smpl_joints.reshape(frames, -1),
            _csv_headers_3d("smpl_joint", 24),
        )
        _write_csv(
            motion_dir / "smpl_root_quat.csv",
            smpl_roots,
            _csv_headers_quat("smpl_root", 1),
        )

        info = {
            "motion_name": motion_name,
            "source_key": motion_name,
            "fps": int(FPS),
            "frames": frames,
            "encoder_mode": 2,
            "robot_profile": robot_profile,
            "joint_space": reference_joint_space,
            "source_joint_space": wrist_contract,
            "template_motion": str(template_motion),
            "template_frame": template_frame,
            "joint_pos_shape": list(joint_pos.shape),
            "h2_joint_pos_shape": list(h2_joint_pos.shape),
            "body_quat_format": "wxyz",
            "body_count": 1,
            "smpl": {
                "synthetic": True,
                "kinematically_consistent": True,
                "smpl_joint_shape": list(smpl_joints.shape),
                "smpl_pose_shape": list(smpl_pose.shape),
                "smpl_root_quat_shape": list(smpl_roots.shape),
                "smpl_joint_frame": "z_up_pelvis_centered_world_oriented_no_translation",
                "smpl_root_quat_format": "runtime_wxyz_after_yup_to_zup_and_remove_base_rot",
                "wrist_targets": wrist_target[[23, 24, 25, 26, 27, 28]].tolist(),
                "geometry": geometry,
            },
        }
        (motion_dir / "h2_lora_reference.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        (motion_dir / "info.txt").write_text(f"Static SMPL mode-2 reference: {motion_name}\n", encoding="utf-8")
        motions.append(info)

    manifest = {
        "schema": "h2_lora_deploy_reference_v1",
        "description": "Kinematically consistent static SMPL mode-2 arm references.",
        "generator_contract_version": 6,
        "robot_profile": robot_profile,
        "template_motion": str(template_motion),
        "template_frame": template_frame,
        "motion_root": None,
        "smpl_root": None,
        "output_dir": str(output_dir),
        "fps": int(FPS),
        "num_motions": len(motions),
        "total_frames": frames * len(motions),
        "motions": motions,
    }
    (output_dir / "h2_lora_reference_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "reference_geometry.json").write_text(
        json.dumps(geometry_report, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "motion_summary.txt").write_text(
        "Static SMPL Arm References\n"
        "==========================\n"
        + "\n".join(f"- {name}: {frames} frames, encoder_mode=2" for name in POSES)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gear_sonic_deploy/reference/smpl_static_arm_test"),
    )
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--template-motion", type=Path, default=DEFAULT_TEMPLATE_MOTION)
    parser.add_argument("--template-frame", type=int, default=0)
    parser.add_argument(
        "--robot-profile",
        choices=("g1", "h2_lora"),
        default="h2_lora",
        help="Select the embodiment-specific wrist reference contract",
    )
    args = parser.parse_args()
    manifest = generate(
        args.output_dir,
        args.frames,
        args.template_motion,
        args.template_frame,
        robot_profile=args.robot_profile,
    )
    print(f"Generated {manifest['num_motions']} references in {args.output_dir}")
    for motion in manifest["motions"]:
        print(f"  {motion['motion_name']}: {motion['frames']} frames, mode 2")


if __name__ == "__main__":
    main()
