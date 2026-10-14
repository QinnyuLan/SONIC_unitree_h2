#!/usr/bin/env python3
"""Visualize the two static SMPL arm references as 24-joint skeletons."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "gear_sonic_deploy/reference/smpl_static_arm_test"
MOTION_NAMES = ("arms_overhead_straight", "arms_forward_straight")
SMPL_PARENTS = (
    -1,
    0,
    0,
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
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
)
LEFT_ARM = (13, 16, 18, 20, 22)
RIGHT_ARM = (14, 17, 19, 21, 23)
KEY_JOINT_LABELS = {
    0: "pelvis",
    16: "L shoulder",
    17: "R shoulder",
    18: "L elbow",
    19: "R elbow",
    20: "L wrist",
    21: "R wrist",
}


def _load_motion(reference_root: Path, motion_name: str, frame: int) -> tuple[np.ndarray, int]:
    path = reference_root / motion_name / "smpl_joint.csv"
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != 72:
        raise ValueError(f"Expected 72 SMPL joint values in {path}, got {values.shape[1]}")
    if not 0 <= frame < values.shape[0]:
        raise IndexError(f"Frame {frame} is outside [0, {values.shape[0] - 1}] for {path}")
    return values[frame].reshape(24, 3), values.shape[0]


def _bone_color(parent: int, child: int) -> str:
    if parent in LEFT_ARM and child in LEFT_ARM:
        return "#d1495b"
    if parent in RIGHT_ARM and child in RIGHT_ARM:
        return "#2878b5"
    return "#444444"


def _draw_skeleton(axis, joints: np.ndarray, motion_name: str, frames: int) -> None:
    for child, parent in enumerate(SMPL_PARENTS):
        if parent < 0:
            continue
        points = joints[[parent, child]]
        axis.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=_bone_color(parent, child),
            linewidth=3.0 if child in LEFT_ARM + RIGHT_ARM else 1.8,
        )

    colors = np.full(24, "#555555", dtype=object)
    colors[list(LEFT_ARM)] = "#d1495b"
    colors[list(RIGHT_ARM)] = "#2878b5"
    axis.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=colors, s=26, depthshade=False)
    for joint_index, label in KEY_JOINT_LABELS.items():
        point = joints[joint_index]
        axis.text(point[0], point[1], point[2], f" {label}", fontsize=7)

    pelvis = joints[0]
    arrow_length = 0.22
    axis.quiver(*pelvis, 0.0, -arrow_length, 0.0, color="#c62828", linewidth=2.0)
    axis.quiver(*pelvis, arrow_length, 0.0, 0.0, color="#2e7d32", linewidth=2.0)
    axis.quiver(*pelvis, 0.0, 0.0, arrow_length, color="#1565c0", linewidth=2.0)
    axis.text(*(pelvis + [0.0, -arrow_length, 0.0]), " -Y forward", color="#c62828", fontsize=8)
    axis.text(*(pelvis + [arrow_length, 0.0, 0.0]), " +X left", color="#2e7d32", fontsize=8)
    axis.text(*(pelvis + [0.0, 0.0, arrow_length]), " +Z up", color="#1565c0", fontsize=8)

    center = 0.5 * (joints.min(axis=0) + joints.max(axis=0))
    radius = 0.55 * float(np.max(joints.max(axis=0) - joints.min(axis=0)))
    radius = max(radius, 0.6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_xlabel("raw SMPL X (left)")
    axis.set_ylabel("raw SMPL Y (-Y forward)")
    axis.set_zlabel("Z (up)")
    axis.set_title(f"{motion_name}\n{frames} identical frames, SMPL mode 2")
    axis.view_init(elev=12.0, azim=-62.0)
    axis.grid(True, alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--motion", choices=("all", *MOTION_NAMES), default="all")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        help="Save a PNG instead of opening an interactive Matplotlib window",
    )
    args = parser.parse_args()

    motion_names = MOTION_NAMES if args.motion == "all" else (args.motion,)
    figure = plt.figure(figsize=(7 * len(motion_names), 7), constrained_layout=True)
    for plot_index, motion_name in enumerate(motion_names, start=1):
        joints, frames = _load_motion(args.reference_root.resolve(), motion_name, args.frame)
        axis = figure.add_subplot(1, len(motion_names), plot_index, projection="3d")
        _draw_skeleton(axis, joints, motion_name, frames)
    figure.suptitle(
        "Stored raw SMPL references (red: left arm, blue: right arm)",
        fontsize=14,
    )

    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
        print(f"Wrote {output}")
    else:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
