#!/usr/bin/env python3
"""Stream a static SMPL elbow-angle sweep and measure G1/H2 policy response."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import msgpack
import numpy as np
from scipy.spatial.transform import Rotation
import zmq

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.data.robot_model.instantiation.h2 import instantiate_h2_robot_model
from gear_sonic.utils.teleop.h2_vr3pt_kinematics import (
    h2_configuration_from_g1_compatible_feedback,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCES = {
    "g1": REPO_ROOT
    / "gear_sonic_deploy/reference/smpl_static_arm_test_g1/arms_forward_straight",
    "h2-native": REPO_ROOT
    / "gear_sonic_deploy/reference/smpl_static_arm_test_h2/arms_forward_straight",
    "h2-native-v1-1": REPO_ROOT
    / "gear_sonic_deploy/reference/smpl_static_arm_test_h2/arms_forward_straight",
}
DEFAULT_OUTPUTS = {
    "g1": REPO_ROOT / "test_logs/g1_smpl_elbow_sweep.json",
    "h2-native": REPO_ROOT / "test_logs/h2_native_smpl_elbow_sweep.json",
    "h2-native-v1-1": REPO_ROOT
    / "test_logs/h2_native_v1_1_smpl_elbow_sweep.json",
}
ARM_CHAINS = ((16, 18, 20, 22), (17, 19, 21, 23))


def _load_first_csv_row(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, max_rows=1, ndmin=2)[0]


def load_reference(reference_dir: Path) -> dict[str, np.ndarray]:
    joints_world = _load_first_csv_row(reference_dir / "smpl_joint.csv").reshape(24, 3)
    root_quat = _load_first_csv_row(reference_dir / "smpl_root_quat.csv").reshape(-1, 4)[0]
    joints_local = Rotation.from_quat(root_quat, scalar_first=True).inv().apply(joints_world)
    return {
        "joints_local": joints_local,
        "root_quat": root_quat,
        "body_quat": _load_first_csv_row(reference_dir / "body_quat.csv").reshape(-1, 4)[0],
        "smpl_pose": _load_first_csv_row(reference_dir / "smpl_pose.csv").reshape(21, 3),
        "joint_pos": _load_first_csv_row(reference_dir / "joint_pos.csv").reshape(29),
    }


def joints_for_elbow_angle(reference: dict[str, np.ndarray], angle_degrees: float) -> np.ndarray:
    if not 0.0 < angle_degrees <= 180.0:
        raise ValueError("elbow angle must be in (0, 180]")
    joints = reference["joints_local"].copy()
    angle_radians = math.radians(angle_degrees)
    forearm_direction = np.asarray(
        [-math.cos(angle_radians), 0.0, math.sin(angle_radians)], dtype=np.float64
    )
    for shoulder, elbow, wrist, hand in ARM_CHAINS:
        upper_length = float(np.linalg.norm(joints[elbow] - joints[shoulder]))
        forearm_length = float(np.linalg.norm(joints[wrist] - joints[elbow]))
        hand_length = float(np.linalg.norm(joints[hand] - joints[wrist]))
        joints[elbow] = joints[shoulder] + np.asarray([upper_length, 0.0, 0.0])
        joints[wrist] = joints[elbow] + forearm_length * forearm_direction
        joints[hand] = joints[wrist] + hand_length * forearm_direction
    return joints


def _internal_angle(joints: np.ndarray, shoulder: int, elbow: int, wrist: int) -> float:
    first = joints[shoulder] - joints[elbow]
    second = joints[wrist] - joints[elbow]
    cosine = float(
        np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
    )
    return math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))


def build_pose_window(
    reference: dict[str, np.ndarray],
    angle_degrees: float,
    frame_start: int,
    frame_count: int = 50,
) -> dict[str, np.ndarray]:
    joints_local = joints_for_elbow_angle(reference, angle_degrees)
    joints_world = Rotation.from_quat(
        reference["root_quat"], scalar_first=True
    ).apply(joints_local)
    for shoulder, elbow, wrist, _ in ARM_CHAINS:
        actual = _internal_angle(joints_local, shoulder, elbow, wrist)
        if abs(actual - angle_degrees) > 1e-5:
            raise RuntimeError(f"generated elbow angle is {actual}, expected {angle_degrees}")
    return {
        "smpl_pose": np.repeat(
            reference["smpl_pose"][None], frame_count, axis=0
        ).astype(np.float32),
        "smpl_joints": np.repeat(joints_world[None], frame_count, axis=0).astype(
            np.float32
        ),
        "smpl_root_quat": np.repeat(
            reference["root_quat"][None], frame_count, axis=0
        ).astype(np.float32),
        "body_quat_w": np.repeat(
            reference["body_quat"][None], frame_count, axis=0
        ).astype(np.float32),
        "joint_pos": np.repeat(
            reference["joint_pos"][None], frame_count, axis=0
        ).astype(np.float32),
        "joint_vel": np.zeros((frame_count, 29), dtype=np.float32),
        "frame_index": np.arange(frame_start, frame_start + frame_count, dtype=np.int64),
    }


def _robot_elbow_angles(
    robot, feedback_q: list[float], robot_profile: str
) -> list[float]:
    feedback = np.asarray(feedback_q, dtype=np.float64)
    if robot_profile in {"h2-native", "h2-native-v1-1"}:
        configuration = h2_configuration_from_g1_compatible_feedback(robot, feedback)
    else:
        configuration = robot.get_configuration_from_actuated_joints(
            body_actuated_joint_values=feedback[:29]
        )
    robot.cache_forward_kinematics(configuration, auto_clip=False)
    result = []
    for side in ("left", "right"):
        shoulder = robot.frame_placement(f"{side}_shoulder_pitch_link").translation
        elbow = robot.frame_placement(f"{side}_elbow_link").translation
        wrist = robot.frame_placement(f"{side}_wrist_yaw_link").translation
        first = shoulder - elbow
        second = wrist - elbow
        cosine = float(
            np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
        )
        result.append(math.degrees(math.acos(np.clip(cosine, -1.0, 1.0))))
    return result


def _stats(samples: np.ndarray) -> dict[str, list[float]]:
    return {
        "median": np.median(samples, axis=0).tolist(),
        "mean": np.mean(samples, axis=0).tolist(),
        "p10": np.percentile(samples, 10.0, axis=0).tolist(),
        "p90": np.percentile(samples, 90.0, axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-profile",
        choices=("g1", "h2-native", "h2-native-v1-1"),
        default="h2-native",
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pose-port", type=int, default=5556)
    parser.add_argument("--feedback-port", type=int, default=5557)
    parser.add_argument("--start-angle", type=int, default=90)
    parser.add_argument("--stop-angle", type=int, default=180)
    parser.add_argument("--step-angle", type=int, default=5)
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    parser.add_argument("--settle-seconds", type=float, default=1.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.step_angle <= 0 or args.stop_angle < args.start_angle:
        parser.error("invalid angle range")
    if not 0.0 <= args.settle_seconds < args.hold_seconds:
        parser.error("settle seconds must be non-negative and shorter than hold seconds")

    reference_path = args.reference or DEFAULT_REFERENCES[args.robot_profile]
    output_path = args.output or DEFAULT_OUTPUTS[args.robot_profile]

    angles = list(range(args.start_angle, args.stop_angle + 1, args.step_angle))
    reference = load_reference(reference_path)
    for index, angle in enumerate(angles):
        build_pose_window(reference, angle, index * 50)
    if args.dry_run:
        print(f"Validated {len(angles)} inputs: {angles}")
        return

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind(f"tcp://*:{args.pose_port}")
    feedback = context.socket(zmq.SUB)
    feedback.setsockopt(zmq.SUBSCRIBE, b"g1_debug")
    feedback.setsockopt(zmq.CONFLATE, 1)
    feedback.connect(f"tcp://localhost:{args.feedback_port}")
    robot = (
        instantiate_g1_robot_model()
        if args.robot_profile == "g1"
        else instantiate_h2_robot_model()
    )
    time.sleep(1.0)

    results = []
    frame_index = 0
    try:
        for angle in angles:
            window = build_pose_window(reference, angle, frame_index)
            frame_index += 50
            packed_pose = pack_pose_message(window, topic="pose", version=3)
            started_at = time.monotonic()
            last_command_at = -math.inf
            target_samples = []
            measured_samples = []
            while time.monotonic() - started_at < args.hold_seconds:
                now = time.monotonic()
                publisher.send(packed_pose)
                if now - last_command_at >= 0.2:
                    publisher.send(
                        build_command_message(start=True, stop=False, planner=False)
                    )
                    last_command_at = now
                if feedback.poll(timeout=0):
                    message = feedback.recv(zmq.NOBLOCK)
                    payload = msgpack.unpackb(message[len(b"g1_debug") :], raw=False)
                    if now - started_at >= args.settle_seconds:
                        target_samples.append(
                            _robot_elbow_angles(
                                robot, payload["body_q_target"], args.robot_profile
                            )
                        )
                        measured_samples.append(
                            _robot_elbow_angles(
                                robot, payload["body_q_measured"], args.robot_profile
                            )
                        )
                time.sleep(0.02)
            if not target_samples or not measured_samples:
                raise RuntimeError(f"no H2 feedback samples collected for input {angle}")
            target_array = np.asarray(target_samples)
            measured_array = np.asarray(measured_samples)
            result = {
                "input_elbow_degrees": angle,
                "samples": len(target_samples),
                "target_elbow_degrees": _stats(target_array),
                "measured_elbow_degrees": _stats(measured_array),
            }
            results.append(result)
            print(
                f"input={angle:3d} target(L/R)="
                f"{result['target_elbow_degrees']['median'][0]:6.1f}/"
                f"{result['target_elbow_degrees']['median'][1]:6.1f} measured="
                f"{result['measured_elbow_degrees']['median'][0]:6.1f}/"
                f"{result['measured_elbow_degrees']['median'][1]:6.1f}"
            )
    finally:
        publisher.close()
        feedback.close()
        context.term()

    payload = {
        "robot_profile": args.robot_profile,
        "reference": str(reference_path.resolve()),
        "hold_seconds": args.hold_seconds,
        "settle_seconds": args.settle_seconds,
        "elbow_bend_plane": "forearm rotates from +Z at 90 degrees to +X at 180 degrees",
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
