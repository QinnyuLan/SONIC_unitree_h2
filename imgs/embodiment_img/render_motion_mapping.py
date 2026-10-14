#!/usr/bin/env python3
"""Render a paper-ready SMPL-X/G1/H2 motion-mapping sequence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[2]
TELEOP_PYTHON = REPO_ROOT / ".venv_teleop/bin/python"
try:
    import smplx
    import torch
except ModuleNotFoundError:
    if TELEOP_PYTHON.exists() and Path(sys.executable).resolve() != TELEOP_PYTHON.resolve():
        os.execv(str(TELEOP_PYTHON), [str(TELEOP_PYTHON), *sys.argv])
    raise

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import yaml


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv  # noqa: E402


DEFAULT_MOTION_DIR = (
    REPO_ROOT
    / "gear_sonic_deploy/reference/example_h2/deploy/macarena_001__A545"
)
G1_MODEL_PATH = (
    REPO_ROOT
    / "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"
)
H2_LOOP_CONFIG_PATH = (
    REPO_ROOT
    / "gear_sonic/utils/mujoco_sim/wbc_configs/"
    "h2_native_sonic_unitree_ros_loop.yaml"
)
SMPLX_MODEL_ROOT = REPO_ROOT / "data/body_models"
SMPLX_MODEL_PATH = SMPLX_MODEL_ROOT / "smplx/SMPLX_NEUTRAL.npz"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

G1_JOINT_NAMES = (
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
)

SMPL22_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
SMPL22_FIT_WEIGHTS = (1, 1, 1, 0.1, 1.5, 1.5, 0.1, 1.5, 1.5, 0.1, 1, 1, 0.5, 0.2, 0.2, 1, 2, 2, 2.5, 2.5, 4, 4)

RAW_RENDER_SIZE = (600, 800)
CELL_SIZE = (410, 540)
COLUMN_GAP = 24
ROW_GAP = 42
CANVAS_MARGIN = 42
HUMAN_RGBA = (0.64, 0.50, 0.42, 1.0)


def _load_csv(path: Path, columns: int) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != columns:
        raise ValueError(f"Expected {columns} columns in {path}, got {values.shape[1]}")
    return values


def _load_motion(motion_dir: Path) -> dict[str, np.ndarray]:
    arrays = {
        "g1": _load_csv(motion_dir / "joint_pos.csv", 29),
        "h2": _load_csv(motion_dir / "h2_joint_pos.csv", 31),
        "root_quat": _load_csv(motion_dir / "body_quat.csv", 4),
        "smpl_joints": _load_csv(motion_dir / "smpl_joint.csv", 72).reshape(
            -1, 24, 3
        ),
    }
    frame_counts = {values.shape[0] for values in arrays.values()}
    if len(frame_counts) != 1:
        raise ValueError(f"Motion arrays have inconsistent frame counts: {frame_counts}")
    return arrays


def _load_smplx_betas(motion_dir: Path) -> np.ndarray:
    metadata_path = motion_dir / "h2_lora_reference.json"
    metadata = json.loads(metadata_path.read_text())
    source_path = Path(metadata["source_smpl_pkl"])
    if not source_path.is_absolute():
        source_path = REPO_ROOT / source_path
    with np.load(source_path, allow_pickle=True) as source:
        betas = np.asarray(source["identity_coeffs"], dtype=np.float32).reshape(-1)
    if betas.size < 10:
        betas = np.pad(betas, (0, 10 - betas.size))
    return betas[:10]


def _load_g1() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(G1_MODEL_PATH))
    data = mujoco.MjData(model)
    _configure_model_for_rendering(model)
    return model, data


def _load_h2_loop() -> DefaultEnv:
    with H2_LOOP_CONFIG_PATH.open() as config_file:
        config = yaml.safe_load(config_file)
    config["PRINT_SCENE_INFORMATION"] = False
    for key in ("ROBOT_SCENE", "MJCF_MESHDIR"):
        config[key] = str(REPO_ROOT / config[key])
    environment = DefaultEnv(config, onscreen=False)
    _configure_model_for_rendering(environment.mj_model)
    return environment


def _configure_model_for_rendering(model: mujoco.MjModel) -> None:
    model.vis.global_.offwidth = RAW_RENDER_SIZE[0]
    model.vis.global_.offheight = RAW_RENDER_SIZE[1]
    model.vis.global_.orthographic = 1
    model.vis.headlight.ambient[:] = (0.45, 0.45, 0.45)
    model.vis.headlight.diffuse[:] = (0.75, 0.75, 0.75)
    model.vis.headlight.specular[:] = (0.2, 0.2, 0.2)
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE:
            model.geom_rgba[geom_id, 3] = 0.0


def _set_root_pose(
    model: mujoco.MjModel, data: mujoco.MjData, quat_wxyz: np.ndarray
) -> None:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    data.qpos[:3] = model.qpos0[:3]
    data.qpos[:2] = 0.0
    data.qpos[3:7] = quat


def _set_g1_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_values: np.ndarray,
    root_quat: np.ndarray,
) -> None:
    data.qpos[:] = model.qpos0
    _set_root_pose(model, data, root_quat)
    for name, value in zip(G1_JOINT_NAMES, joint_values, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"G1 model is missing joint {name}")
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)


def _set_h2_loop_pose(
    environment: DefaultEnv,
    joint_values: np.ndarray,
    root_quat: np.ndarray,
) -> None:
    model = environment.mj_model
    data = environment.mj_data
    _set_root_pose(model, data, root_quat)

    closed_outputs = {
        int(position)
        for group in environment.closed_loop_transmissions
        for position in group["output_positions"]
    }
    direct_positions = np.asarray(
        [
            position
            for position in range(len(environment.body_joint_index))
            if position not in closed_outputs
        ],
        dtype=np.int64,
    )
    data.qpos[environment.body_qpos_index[direct_positions]] = joint_values[
        direct_positions
    ]
    mujoco.mj_forward(model, data)
    environment._project_closed_loop_default_pose(joint_values)


def _render_rgba(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    lookat_z: float,
    vertical_span: float,
) -> Image.Image:
    model.vis.global_.fovy = vertical_span
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, lookat_z)
    camera.distance = 3.0
    camera.azimuth = 90.0
    camera.elevation = 0.0

    renderer = mujoco.Renderer(
        model, height=RAW_RENDER_SIZE[1], width=RAW_RENDER_SIZE[0]
    )
    renderer.update_scene(data, camera)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    renderer.close()

    foreground = np.isfinite(depth) & (depth < float(depth.max()) - 1e-5)
    rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = foreground.astype(np.uint8) * 255
    return Image.fromarray(rgba, mode="RGBA")


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    return vector / norm


def _initial_smplx_root_orientations(targets_yup: np.ndarray) -> np.ndarray:
    orientations = []
    for joints in targets_yup:
        x_axis = _normalize(joints[16] - joints[17])
        y_axis = joints[12] - joints[0]
        y_axis = _normalize(y_axis - x_axis * np.dot(x_axis, y_axis))
        z_axis = _normalize(np.cross(x_axis, y_axis))
        y_axis = _normalize(np.cross(z_axis, x_axis))
        orientations.append(
            Rotation.from_matrix(np.column_stack((x_axis, y_axis, z_axis))).as_rotvec()
        )
    return np.asarray(orientations, dtype=np.float32)


def _fit_smplx_vertices(
    smpl_joints_zup: np.ndarray, betas: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not SMPLX_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing licensed SMPL-X model: {SMPLX_MODEL_PATH}")

    targets = np.stack(
        (
            smpl_joints_zup[:, :22, 0],
            smpl_joints_zup[:, :22, 2],
            -smpl_joints_zup[:, :22, 1],
        ),
        axis=-1,
    ).astype(np.float32)
    targets -= targets[:, :1]
    batch_size = targets.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body_model = smplx.create(
        str(SMPLX_MODEL_ROOT),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        batch_size=batch_size,
    ).to(device)
    body_model.eval()

    target_tensor = torch.as_tensor(targets, device=device)
    beta_tensor = torch.as_tensor(betas, device=device).view(1, 10)
    beta_tensor = beta_tensor.expand(batch_size, -1)
    global_orient = torch.nn.Parameter(
        torch.as_tensor(
            _initial_smplx_root_orientations(targets), device=device
        )
    )
    body_pose = torch.nn.Parameter(torch.zeros(batch_size, 63, device=device))
    weights = torch.as_tensor(SMPL22_FIT_WEIGHTS, device=device)
    parents = torch.as_tensor(SMPL22_PARENTS, device=device)
    optimizer = torch.optim.Adam((global_orient, body_pose), lr=0.03)

    for _ in range(800):
        optimizer.zero_grad()
        output = body_model(
            betas=beta_tensor,
            global_orient=global_orient,
            body_pose=body_pose,
            return_verts=False,
        )
        predicted = output.joints[:, :22]
        predicted = predicted - predicted[:, :1]
        position_loss = (
            ((predicted - target_tensor).square().sum(dim=-1) * weights).sum()
            / weights.sum()
            / batch_size
        )
        predicted_bones = predicted[:, 1:] - predicted[:, parents[1:]]
        target_bones = target_tensor[:, 1:] - target_tensor[:, parents[1:]]
        direction_loss = (
            1
            - torch.nn.functional.cosine_similarity(
                predicted_bones, target_bones, dim=-1
            )
        ).mean()
        loss = position_loss + 0.02 * direction_loss + 5e-5 * body_pose.square().mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        output = body_model(
            betas=beta_tensor,
            global_orient=global_orient,
            body_pose=body_pose,
            return_verts=True,
        )
        predicted = output.joints[:, :22]
        predicted = predicted - predicted[:, :1]
        mean_error_cm = 100 * torch.linalg.vector_norm(
            predicted - target_tensor, dim=-1
        ).mean(dim=-1)
        print(
            "SMPL-X IK mean joint error (cm): "
            + ", ".join(f"{value:.2f}" for value in mean_error_cm.cpu().tolist())
        )
        vertices = output.vertices.cpu().numpy().astype(np.float32)
    return vertices, np.asarray(body_model.faces, dtype=np.int32)


def _human_model(
    vertices_yup: np.ndarray, faces: np.ndarray
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    vertices = np.stack(
        (vertices_yup[:, 0], -vertices_yup[:, 2], vertices_yup[:, 1]), axis=-1
    )
    vertices[:, 2] -= float(vertices[:, 2].min())
    rgba = " ".join(str(value) for value in HUMAN_RGBA)
    with tempfile.TemporaryDirectory(prefix="smplx_render_") as temp_dir:
        mesh_path = Path(temp_dir) / "smplx.obj"
        with mesh_path.open("w") as mesh_file:
            np.savetxt(mesh_file, vertices, fmt="v %.7f %.7f %.7f")
            np.savetxt(mesh_file, faces + 1, fmt="f %d %d %d")
        xml = (
            f'<mujoco model="smplx_human"><compiler meshdir="{temp_dir}"/>'
            '<visual><global orthographic="true" '
            f'offwidth="{RAW_RENDER_SIZE[0]}" offheight="{RAW_RENDER_SIZE[1]}"/>'
            '</visual><asset><mesh name="smplx" file="smplx.obj"/></asset>'
            f'<worldbody><geom type="mesh" mesh="smplx" rgba="{rgba}" '
            'contype="0" conaffinity="0"/></worldbody></mujoco>'
        )
        model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    _configure_model_for_rendering(model)
    mujoco.mj_forward(model, data)
    return model, data


def _fit_to_cell(image: Image.Image) -> Image.Image:
    alpha_box = image.getchannel("A").getbbox()
    if alpha_box is None:
        raise ValueError("Rendered frame is empty")
    cropped = image.crop(alpha_box)
    max_width = CELL_SIZE[0] - 20
    max_height = CELL_SIZE[1] - 16
    scale = min(max_width / cropped.width, max_height / cropped.height)
    size = (
        max(1, round(cropped.width * scale)),
        max(1, round(cropped.height * scale)),
    )
    return cropped.resize(size, Image.Resampling.LANCZOS)


def _compose(rows: list[list[Image.Image]]) -> Image.Image:
    columns = len(rows[0])
    width = (
        2 * CANVAS_MARGIN
        + columns * CELL_SIZE[0]
        + (columns - 1) * COLUMN_GAP
    )
    height = (
        2 * CANVAS_MARGIN + len(rows) * CELL_SIZE[1] + (len(rows) - 1) * ROW_GAP
    )
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for row_index, row in enumerate(rows):
        if len(row) != columns:
            raise ValueError("All rows must contain the same number of frames")
        cell_y = CANVAS_MARGIN + row_index * (CELL_SIZE[1] + ROW_GAP)
        for column, image in enumerate(row):
            fitted = _fit_to_cell(image)
            cell_x = CANVAS_MARGIN + column * (CELL_SIZE[0] + COLUMN_GAP)
            x = cell_x + (CELL_SIZE[0] - fitted.width) // 2
            y = cell_y + CELL_SIZE[1] - fitted.height
            canvas.alpha_composite(fitted, (x, y))
    return canvas


def render_mapping(
    motion_dir: Path,
    frame_indices: list[int],
    output_dir: Path,
) -> tuple[Path, Path]:
    motion = _load_motion(motion_dir)
    frame_count = motion["g1"].shape[0]
    invalid = [frame for frame in frame_indices if not 0 <= frame < frame_count]
    if invalid:
        raise IndexError(f"Frames outside [0, {frame_count - 1}]: {invalid}")

    g1_model, g1_data = _load_g1()
    h2_environment = _load_h2_loop()
    human_vertices, human_faces = _fit_smplx_vertices(
        motion["smpl_joints"][frame_indices],
        _load_smplx_betas(motion_dir),
    )
    human_frames: list[Image.Image] = []
    g1_frames: list[Image.Image] = []
    h2_frames: list[Image.Image] = []

    for sample_index, frame in enumerate(frame_indices):
        root_quat = motion["root_quat"][frame]

        human_model, human_data = _human_model(
            human_vertices[sample_index], human_faces
        )
        human_frames.append(
            _render_rgba(
                human_model,
                human_data,
                lookat_z=0.90,
                vertical_span=1.90,
            )
        )

        _set_g1_pose(g1_model, g1_data, motion["g1"][frame], root_quat)
        g1_frames.append(
            _render_rgba(g1_model, g1_data, lookat_z=0.68, vertical_span=1.55)
        )

        _set_h2_loop_pose(
            h2_environment, motion["h2"][frame], root_quat
        )
        h2_frames.append(
            _render_rgba(
                h2_environment.mj_model,
                h2_environment.mj_data,
                lookat_z=1.02,
                vertical_span=2.20,
            )
        )

    composite = _compose([human_frames, g1_frames, h2_frames])
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = motion_dir.name
    transparent_path = output_dir / f"{stem}_mapping_transparent.png"
    white_path = output_dir / f"{stem}_mapping_white.png"
    composite.save(transparent_path, optimize=True, dpi=(300, 300))
    white = Image.new("RGB", composite.size, (255, 255, 255))
    white.paste(composite, mask=composite.getchannel("A"))
    white.save(white_path, optimize=True, dpi=(300, 300))

    print(f"Frames: {frame_indices}")
    print(f"Transparent: {transparent_path}")
    print(f"White: {white_path}")
    return transparent_path, white_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-dir", type=Path, default=DEFAULT_MOTION_DIR)
    parser.add_argument("--start-frame", type=int, default=125)
    parser.add_argument("--frame-step", type=int, default=125)
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.frame_step <= 0 or args.num_frames <= 0:
        parser.error("--frame-step and --num-frames must be positive")

    frames = [
        args.start_frame + index * args.frame_step
        for index in range(args.num_frames)
    ]
    render_mapping(
        args.motion_dir.resolve(),
        frames,
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
