#!/usr/bin/env python3
"""Measure static SMPL mode-2 arm tracking from deploy CSV logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = REPO_ROOT / "test_logs/static_smpl_mode2_20260724"
G1_MODEL = REPO_ROOT / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
H2_MODEL = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/h2.xml"

MOTIONS = {
    "arms_overhead_straight": np.asarray([0.0, 0.0, 1.0]),
    "arms_forward_straight": np.asarray([1.0, 0.0, 0.0]),
}

G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]
for _side in ("left", "right"):
    G1_JOINT_NAMES.extend(
        f"{_side}_{part}_joint"
        for part in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    )

H2_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_pitch_joint",
    "head_yaw_joint",
]
for _side in ("left", "right"):
    H2_JOINT_NAMES.extend(
        f"{_side}_{part}_joint"
        for part in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    )

ROBOT_SPECS = {
    "g1": (G1_MODEL, G1_JOINT_NAMES),
    "h2_lora": (H2_MODEL, H2_JOINT_NAMES),
}


def _load_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {int(row["index"]): row for row in csv.DictReader(stream)}


def _load_model(path: Path) -> mujoco.MjModel:
    if path != H2_MODEL:
        return mujoco.MjModel.from_xml_path(str(path))

    # The H2 XML's kinematics are self-contained. Removing visual mesh entries
    # makes offline FK independent of the optional sibling meshes directory.
    tree = ET.parse(path)
    root = tree.getroot()
    for asset in root.findall("asset"):
        for child in list(asset):
            if child.tag == "mesh":
                asset.remove(child)
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom" and "mesh" in child.attrib:
                parent.remove(child)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-12:
        raise ValueError("Cannot measure an angle from a zero-length vector")
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _joint_anchor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
) -> np.ndarray:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise KeyError(f"Joint is missing from model: {joint_name}")
    return data.xanchor[joint_id].copy()


def _set_robot_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: list[str],
    q: np.ndarray,
) -> None:
    if len(joint_names) != q.shape[0]:
        raise ValueError(f"Expected {len(joint_names)} joints, got {q.shape[0]}")
    data.qpos[:] = 0.0
    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if free_joints.size:
        root_qpos = int(model.jnt_qposadr[int(free_joints[0])])
        data.qpos[root_qpos + 3] = 1.0
    for name, value in zip(joint_names, q, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise KeyError(f"Joint is missing from model: {name}")
        data.qpos[int(model.jnt_qposadr[joint_id])] = value
    mujoco.mj_forward(model, data)


def analyze_robot(
    robot: str,
    log_dir: Path,
    window_seconds: float,
) -> tuple[dict, dict[str, dict[str, list[float]]]]:
    model_path, joint_names = ROBOT_SPECS[robot]
    model = _load_model(model_path)
    data = mujoco.MjData(model)

    q_rows = _load_rows(log_dir / "q.csv")
    name_rows = _load_rows(log_dir / "motion_name.csv")
    playing_rows = _load_rows(log_dir / "motion_playing.csv")
    mode_rows = _load_rows(log_dir / "encoder_mode.csv")
    metadata = json.loads((log_dir / "metadata.json").read_text(encoding="utf-8"))
    dt = float(metadata["logging"]["dt"])
    window_frames = max(1, round(window_seconds / dt))
    joint_columns = [f"q_{index}" for index in range(len(joint_names))]

    report = {
        "model": str(model_path),
        "log_dir": str(log_dir),
        "window_seconds": window_seconds,
        "motions": {},
    }
    plot_data: dict[str, dict[str, list[float]]] = {}

    for motion_name, target_direction in MOTIONS.items():
        selected = [
            index
            for index in sorted(q_rows)
            if index in name_rows
            and index in playing_rows
            and index in mode_rows
            and name_rows[index]["motion_name"] == motion_name
            and float(playing_rows[index]["playing_0"]) > 0.5
            and round(float(mode_rows[index]["encoder_mode_0"])) == 2
        ][-window_frames:]
        if not selected:
            raise RuntimeError(f"No playing mode-2 samples for {robot}/{motion_name}")

        arm_q = {"left": [], "right": []}
        elbow_joint = {"left": [], "right": []}
        elbow_internal = {"left": [], "right": []}
        target_error = {"left": [], "right": []}
        parallel_error: list[float] = []
        time_seconds: list[float] = []

        for index in selected:
            q = np.asarray([float(q_rows[index][column]) for column in joint_columns])
            _set_robot_qpos(model, data, joint_names, q)
            directions = {}
            for side in ("left", "right"):
                shoulder = _joint_anchor(model, data, f"{side}_shoulder_pitch_joint")
                elbow = _joint_anchor(model, data, f"{side}_elbow_joint")
                wrist = _joint_anchor(model, data, f"{side}_wrist_yaw_joint")
                direction = wrist - shoulder
                directions[side] = direction / np.linalg.norm(direction)
                elbow_internal[side].append(_angle_deg(shoulder - elbow, wrist - elbow))
                target_error[side].append(_angle_deg(direction, target_direction))

                shoulder_index = joint_names.index(f"{side}_shoulder_pitch_joint")
                wrist_index = joint_names.index(f"{side}_wrist_yaw_joint") + 1
                arm_q[side].append(q[shoulder_index:wrist_index])
                elbow_index = joint_names.index(f"{side}_elbow_joint")
                elbow_joint[side].append(math.degrees(q[elbow_index]))

            parallel_error.append(_angle_deg(directions["left"], directions["right"]))
            time_seconds.append(float(q_rows[index]["time_ms"]) / 1000.0)

        all_arm_q = np.concatenate(
            [np.asarray(arm_q["left"]), np.asarray(arm_q["right"])], axis=1
        )
        arm_joint_std_deg = np.degrees(all_arm_q.std(axis=0))
        report["motions"][motion_name] = {
            "sample_count": len(selected),
            "time_range_seconds": [time_seconds[0], time_seconds[-1]],
            "reference": {
                "target_direction": target_direction.tolist(),
                "elbow_internal_angle_deg": 180.0,
                "arms_parallel_error_deg": 0.0,
            },
            "left": {
                "elbow_joint_deg": _stats(elbow_joint["left"]),
                "elbow_internal_angle_deg": _stats(elbow_internal["left"]),
                "target_direction_error_deg": _stats(target_error["left"]),
            },
            "right": {
                "elbow_joint_deg": _stats(elbow_joint["right"]),
                "elbow_internal_angle_deg": _stats(elbow_internal["right"]),
                "target_direction_error_deg": _stats(target_error["right"]),
            },
            "arms_parallel_error_deg": _stats(parallel_error),
            "arm_joint_stability_deg": {
                "mean_std": float(arm_joint_std_deg.mean()),
                "max_std": float(arm_joint_std_deg.max()),
            },
        }
        plot_data[motion_name] = {
            "time": (np.asarray(time_seconds) - time_seconds[0]).tolist(),
            "left": elbow_internal["left"],
            "right": elbow_internal["right"],
        }

    return report, plot_data


def _write_plot(path: Path, plot_data: dict[str, dict[str, dict[str, list[float]]]]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True, constrained_layout=True)
    for row, robot in enumerate(("g1", "h2_lora")):
        for column, motion in enumerate(MOTIONS):
            axis = axes[row, column]
            values = plot_data[robot][motion]
            axis.plot(values["time"], values["left"], label="left", linewidth=1.8)
            axis.plot(values["time"], values["right"], label="right", linewidth=1.8)
            axis.axhline(180.0, color="black", linestyle="--", linewidth=1.0, label="reference")
            axis.set_title(f"{robot}: {motion}")
            axis.set_xlabel("stable-window time (s)")
            axis.set_ylabel("elbow internal angle (deg)")
            axis.set_ylim(60.0, 185.0)
            axis.grid(alpha=0.25)
            axis.legend(loc="lower right")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-logs", type=Path, default=DEFAULT_LOG_ROOT / "g1")
    parser.add_argument("--h2-logs", type=Path, default=DEFAULT_LOG_ROOT / "h2_lora")
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_LOG_ROOT / "tracking_summary.json")
    parser.add_argument("--plot", type=Path, default=DEFAULT_LOG_ROOT / "elbow_tracking.png")
    args = parser.parse_args()

    summary = {"metric_frame": "robot pelvis frame", "robots": {}}
    plots = {}
    for robot, log_dir in (("g1", args.g1_logs), ("h2_lora", args.h2_logs)):
        summary["robots"][robot], plots[robot] = analyze_robot(
            robot,
            log_dir.resolve(),
            args.window_seconds,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_plot(args.plot, plots)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
