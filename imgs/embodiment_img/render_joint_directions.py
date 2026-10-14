#!/usr/bin/env python3
"""Render G1 and H2 joint axes with MuJoCo's native visualizer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent
G1_MODEL_PATH = (
    REPO_ROOT
    / "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"
)
H2_LOOP_CONFIG_PATH = (
    REPO_ROOT
    / "gear_sonic/utils/mujoco_sim/wbc_configs/"
    "h2_native_sonic_unitree_ros_loop.yaml"
)

DISPLAY_POSE = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}

IMAGE_SIZE = (1600, 2000)
CAMERA_SETTINGS = {
    "g1": {"lookat_z": 0.70, "vertical_span": 1.55},
    "h2": {"lookat_z": 1.05, "vertical_span": 2.20},
}


def _matches_pose_pattern(name: str, pattern: str) -> bool:
    if pattern.startswith(".*"):
        return name.endswith(pattern[2:])
    return name == pattern


def _load_g1() -> tuple[mujoco.MjModel, mujoco.MjData, set[str]]:
    model = mujoco.MjModel.from_xml_path(str(G1_MODEL_PATH))
    data = mujoco.MjData(model)
    visible_joints: set[str] = set()

    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        visible_joints.add(name)
        for pattern, value in DISPLAY_POSE.items():
            if _matches_pose_pattern(name, pattern):
                data.qpos[model.jnt_qposadr[joint_id]] = value
                break

    mujoco.mj_forward(model, data)
    return model, data, visible_joints


def _load_h2_loop() -> tuple[mujoco.MjModel, mujoco.MjData, set[str]]:
    with H2_LOOP_CONFIG_PATH.open() as config_file:
        config = yaml.safe_load(config_file)

    config["PRINT_SCENE_INFORMATION"] = False
    for key in ("ROBOT_SCENE", "MJCF_MESHDIR"):
        config[key] = str(REPO_ROOT / config[key])

    # DefaultEnv projects the requested 31-DoF pose onto H2_loop.xml's six
    # equality constraints, keeping the knee, ankle, and waist loops closed.
    environment = DefaultEnv(config, onscreen=False)
    visible_joints = set(config["MUJOCO_BODY_JOINT_NAMES"])
    return environment.mj_model, environment.mj_data, visible_joints


def _configure_native_joint_visualization(
    model: mujoco.MjModel, visible_joints: set[str], robot: str
) -> mujoco.MjvOption:
    model.vis.global_.offwidth = IMAGE_SIZE[0]
    model.vis.global_.offheight = IMAGE_SIZE[1]
    model.vis.global_.orthographic = 1
    model.vis.global_.fovy = CAMERA_SETTINGS[robot]["vertical_span"]
    model.vis.scale.jointlength = 0.82
    model.vis.scale.jointwidth = 0.13
    model.vis.rgba.joint[:] = (0.92, 0.12, 0.05, 1.0)
    model.vis.headlight.ambient[:] = (0.45, 0.45, 0.45)
    model.vis.headlight.diffuse[:] = (0.75, 0.75, 0.75)
    model.vis.headlight.specular[:] = (0.2, 0.2, 0.2)

    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE:
            model.geom_rgba[geom_id, 3] = 0.0

    # MuJoCo's joint groups select the 29/31 embodiment joints while hiding
    # the floating base and H2 loop's auxiliary ball/motor joints.
    model.jnt_group[:] = 1
    selected = 0
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        if name in visible_joints:
            model.jnt_group[joint_id] = 0
            selected += 1
    if selected != len(visible_joints):
        raise ValueError(
            f"Expected {len(visible_joints)} visible joints, found {selected}"
        )

    option = mujoco.MjvOption()
    option.jointgroup[:] = 0
    option.jointgroup[0] = 1
    option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1
    option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1
    return option


def _render(
    robot: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    option: mujoco.MjvOption,
) -> Image.Image:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, CAMERA_SETTINGS[robot]["lookat_z"])
    camera.distance = 3.0
    camera.azimuth = 180.0
    camera.elevation = 0.0

    renderer = mujoco.Renderer(
        model, height=IMAGE_SIZE[1], width=IMAGE_SIZE[0]
    )
    renderer.update_scene(data, camera, scene_option=option)
    rgb = renderer.render().copy()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera, scene_option=option)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    foreground = np.isfinite(depth) & (depth < float(depth.max()) - 1e-5)
    rgb[~foreground] = 255
    renderer.close()
    return Image.fromarray(rgb)


def render_robot(robot: str, output_dir: Path) -> Path:
    if robot == "g1":
        model, data, visible_joints = _load_g1()
    else:
        model, data, visible_joints = _load_h2_loop()

    option = _configure_native_joint_visualization(
        model, visible_joints, robot
    )
    image = _render(robot, model, data, option)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{robot}_joint_directions.png"
    image.save(output_path, optimize=True)
    print(
        f"Rendered {robot.upper()} with {len(visible_joints)} native MuJoCo "
        f"joint axes -> {output_path}"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("all", "g1", "h2"), default="all")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    robots = ("g1", "h2") if args.robot == "all" else (args.robot,)
    for robot in robots:
        render_robot(robot, args.output_dir.resolve())


if __name__ == "__main__":
    main()
