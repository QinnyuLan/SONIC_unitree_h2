#!/usr/bin/env python3
"""Render a G1 release-policy rollout on a H2->G1 semantic reference motion.

This diagnostic isolates the H2LoRA action/robot side from the released G1
policy.  It converts one H2 MotionLib clip into a temporary G1-compatible
MotionLib clip by mapping the H2 headless 29-DoF physical joint trajectory into
G1 semantic joint space, then runs ``sonic_release/last.pt`` on G1 and renders
reference/tracking robots side by side.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import io
import json
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import joblib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.envs.env_utils import h2lora_alignment, joint_utils  # noqa: E402

from scripts.visualize_h2lora_ref_tracking import (  # noqa: E402
    _load_config_for_checkpoint,
    _make_app_args,
    _null_cfg_terms,
    _prefix_names,
    _sanitize_name,
    _set_robot_rgba,
    _setup_model,
    _trajectory_fps,
)


DEFAULT_CHECKPOINT = REPO_ROOT / "sonic_release/last.pt"
DEFAULT_H2_MOTION_FILE = Path("/mnt/data/seed/h2lora_motion_lib/seed_5m_50hz_schema6/robot")
DEFAULT_H2_SMPL_MOTION_FILE = Path("/mnt/data/seed/h2lora_motion_lib/seed_5m_50hz_schema6/smpl")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "visualizations/g1_release_h2adapted"
G1_MJCF = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"
G1_MESH_DIR = REPO_ROOT / "gear_sonic/data/assets/robot_description/meshes/g1"


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _prepare_output_dir(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).expanduser().resolve()
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        name = _sanitize_name(args.experiment_name or f"g1_release_h2adapted_{args.motion_key}")
        output_dir = output_root / f"{name}_{args.timestamp or _timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _parse_mjcf(path: Path) -> tuple[list[str], dict[str, str], dict[str, np.ndarray], list[str]]:
    root = ET.parse(path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF has no worldbody: {path}")

    body_names: list[str] = []
    joint_to_body: dict[str, str] = {}
    joint_axes: dict[str, np.ndarray] = {}
    joint_names: list[str] = []

    def parse_vec(text: str | None) -> np.ndarray:
        if text is None:
            return np.zeros(3, dtype=np.float32)
        return np.asarray([float(x) for x in text.split()], dtype=np.float32)

    def walk(body: ET.Element) -> None:
        body_name = body.get("name")
        if body_name is None:
            return
        body_names.append(body_name)
        for joint in body.findall("joint"):
            name = joint.get("name")
            if name is None or name == "floating_base_joint":
                continue
            joint_names.append(name)
            joint_to_body[name] = body_name
            joint_axes[name] = parse_vec(joint.get("axis"))
        for child in body.findall("body"):
            walk(child)

    for body in worldbody.findall("body"):
        walk(body)
    return body_names, joint_to_body, joint_axes, joint_names


def _find_motion_file(root: Path, motion_key: str) -> Path:
    matches = sorted(root.rglob(f"{motion_key}.pkl"))
    if not matches:
        raise FileNotFoundError(f"Motion key {motion_key!r} not found under {root}")
    if len(matches) > 1:
        raise ValueError(f"Motion key {motion_key!r} is ambiguous under {root}: {matches[:5]}")
    return matches[0]


def _unwrap_single(data: dict, fallback_key: str) -> tuple[str, dict]:
    if isinstance(data, dict) and len(data) == 1:
        key, entry = next(iter(data.items()))
        return str(key), entry
    return fallback_key, data


def _g1_pose_aa_from_semantic_dof(
    g1_dof_isaac: np.ndarray,
    root_quat_xyzw: np.ndarray,
) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    body_names, joint_to_body, joint_axes, _ = _parse_mjcf(G1_MJCF)
    body_index = {name: idx for idx, name in enumerate(body_names)}
    pose_aa = np.zeros((g1_dof_isaac.shape[0], len(body_names), 3), dtype=np.float32)
    pose_aa[:, 0, :] = Rotation.from_quat(root_quat_xyzw).as_rotvec().astype(np.float32)
    for joint_idx, joint_name in enumerate(joint_utils.G1_ISAACLab_ORDER):
        body_name = joint_to_body[joint_name]
        pose_aa[:, body_index[body_name], :] = (
            joint_axes[joint_name][None, :] * g1_dof_isaac[:, joint_idx : joint_idx + 1]
        )
    return pose_aa


def prepare_g1_motion_from_h2(args: argparse.Namespace, output_dir: Path) -> tuple[Path, Path]:
    """Create a temporary one-clip G1 MotionLib from H2 MotionLib data."""
    h2_root = Path(args.h2_motion_file).expanduser().resolve()
    smpl_root = Path(args.h2_smpl_motion_file).expanduser().resolve()
    h2_path = _find_motion_file(h2_root, args.motion_key)
    smpl_path = _find_motion_file(smpl_root, args.motion_key)
    motion_key, h2_entry = _unwrap_single(joblib.load(h2_path), args.motion_key)
    smpl_key, smpl_entry = _unwrap_single(joblib.load(smpl_path), args.motion_key)
    if motion_key != args.motion_key or smpl_key != args.motion_key:
        raise ValueError(f"Unexpected motion keys: robot={motion_key}, smpl={smpl_key}")

    h2_dof = np.asarray(h2_entry.get("dof", h2_entry.get("dof_pos")), dtype=np.float32)
    if h2_dof.shape[-1] != 31:
        raise ValueError(f"Expected raw H2 31-DoF dof in {h2_path}, got {h2_dof.shape}")
    h2_joint_names = list(h2_entry.get("joint_names", []))
    if not h2_joint_names:
        raise ValueError(f"H2 entry must include joint_names metadata: {h2_path}")
    h2_body_indices = [h2_joint_names.index(name) for name in joint_utils.H2LORA_BODY_JOINTS_IN_G1_ORDER]
    h2_body_physical = h2_dof[:, h2_body_indices]

    import torch

    g1_dof_isaac = (
        h2lora_alignment.h2_to_g1_semantic(torch.from_numpy(h2_body_physical))
        .numpy()
        .astype(np.float32)
    )
    root_quat_xyzw = np.asarray(h2_entry["root_rot"], dtype=np.float32)
    pose_aa = _g1_pose_aa_from_semantic_dof(g1_dof_isaac, root_quat_xyzw)

    g1_entry = {
        "root_trans_offset": np.asarray(h2_entry["root_trans_offset"], dtype=np.float32),
        "root_rot": root_quat_xyzw,
        "pose_aa": pose_aa,
        "dof": g1_dof_isaac,
        "fps": int(h2_entry.get("fps", 50)),
        "joint_names": list(joint_utils.G1_ISAACLab_ORDER),
        "joint_order": "g1_isaaclab_semantic_from_h2_headless",
        "source_h2_motion": str(h2_path),
        "source_h2_smpl": str(smpl_path),
        "conversion": "h2_physical_headless -> h2lora_alignment.h2_to_g1_semantic",
    }

    subdir = h2_path.parent.name
    robot_out = output_dir / "g1_adapted_motion" / "robot" / subdir / f"{args.motion_key}.pkl"
    smpl_out = output_dir / "g1_adapted_motion" / "smpl" / subdir / f"{args.motion_key}.pkl"
    robot_out.parent.mkdir(parents=True, exist_ok=True)
    smpl_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({args.motion_key: g1_entry}, robot_out, compress=True)
    joblib.dump({args.motion_key: smpl_entry}, smpl_out, compress=True)

    manifest = {
        "motion_key": args.motion_key,
        "h2_motion": str(h2_path),
        "h2_smpl": str(smpl_path),
        "robot_motion_file": str(robot_out.parent.parent),
        "smpl_motion_file": str(smpl_out.parent.parent),
        "frames": int(g1_dof_isaac.shape[0]),
        "fps": int(g1_entry["fps"]),
        "xy_displacement_m": float(
            np.linalg.norm(g1_entry["root_trans_offset"][-1, :2] - g1_entry["root_trans_offset"][0, :2])
        ),
        "g1_dof_abs_mean": float(np.mean(np.abs(g1_dof_isaac))),
        "g1_dof_std": float(np.std(g1_dof_isaac)),
    }
    (output_dir / "g1_adapted_motion_manifest.json").write_text(json.dumps(manifest, indent=2))
    return robot_out.parent.parent, smpl_out.parent.parent


def _g1_mujoco_joint_names() -> list[str]:
    return _parse_mjcf(G1_MJCF)[3]


def _isaac_to_g1_mujoco_dof_from_names(isaac_joint_names: list[str]) -> np.ndarray:
    missing = [name for name in _g1_mujoco_joint_names() if name not in isaac_joint_names]
    if missing:
        raise ValueError(f"G1 MuJoCo joints missing from IsaacLab asset: {missing}")
    return np.asarray([isaac_joint_names.index(name) for name in _g1_mujoco_joint_names()], dtype=np.int64)


def _load_config_for_g1(args: argparse.Namespace, robot_motion_file: Path, smpl_motion_file: Path):
    # Reuse the H2 visualization compatibility loader, then enforce G1 release
    # settings for this diagnostic.
    forwarded = copy.copy(args)
    forwarded.motion_file = str(robot_motion_file)
    forwarded.smpl_motion_file = str(smpl_motion_file)
    config = _load_config_for_checkpoint(forwarded)
    from omegaconf import open_dict

    with open_dict(config):
        config.manager_env.config.robot.type = "g1_model_12_dex"
        config.manager_env.commands.motion.use_h2lora_alignment = False
        config.manager_env.commands.motion.motion_lib_num_dof = 29
        config.manager_env.commands.motion.motion_lib_cfg.asset.assetFileName = "g1_29dof_rev_1_0.xml"
        config.manager_env.commands.motion.motion_lib_cfg.motion_file = str(robot_motion_file)
        config.manager_env.commands.motion.motion_lib_cfg.smpl_motion_file = str(smpl_motion_file)
        config.manager_env.commands.motion.motion_lib_cfg.filter_motion_keys = [args.motion_key]
        config.manager_env.commands.motion.filter_motion_keys = [args.motion_key]
        config.manager_env.commands.motion.start_from_first_frame = True
        config.manager_env.commands.motion.freeze_frame_aug = False
        if args.use_encoder is not None:
            probs = config.manager_env.commands.motion.encoder_sample_probs
            for encoder_name in list(probs.keys()):
                probs[encoder_name] = 1.0 if encoder_name == args.use_encoder else 0.0
        if args.disable_terminations:
            _null_cfg_terms(config.manager_env.terminations)
        if args.disable_events:
            _null_cfg_terms(config.manager_env.events)
    return config


def collect_rollout(args: argparse.Namespace, output_dir: Path, robot_motion_file: Path, smpl_motion_file: Path) -> Path:
    import torch
    from filelock import FileLock
    from omegaconf import OmegaConf

    from gear_sonic.utils.common import seeding

    config = _load_config_for_g1(args, robot_motion_file, smpl_motion_file)
    seeding(int(config.seed))
    print(f"[g1-h2adapted-viz] Launching IsaacLab app; output_dir={output_dir}", flush=True)

    from isaaclab.app import AppLauncher

    app_args = _make_app_args(config, AppLauncher)
    with FileLock("/tmp/isaaclab_app_launcher.lock"):
        app_launcher = AppLauncher(app_args)
    simulation_app = app_launcher.app
    print("[g1-h2adapted-viz] IsaacLab app launched", flush=True)

    from gear_sonic import train_agent_trl

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    env = None
    try:
        print("[g1-h2adapted-viz] Creating G1 manager env", flush=True)
        env = train_agent_trl.create_manager_env(config, device, app_args)
        if hasattr(env, "set_is_evaluating"):
            env.set_is_evaluating(True)
        if hasattr(env, "reinit_dr"):
            env.reinit_dr()

        print("[g1-h2adapted-viz] Loading release G1 policy checkpoint", flush=True)
        policy = _setup_model(config, env, device)
        obs_dict = env.reset_all()
        robot = env.env.scene["robot"]
        command = env.motion_command
        isaac_to_mujoco = _isaac_to_g1_mujoco_dof_from_names(robot.joint_names)

        frames: dict[str, list[np.ndarray | float | bool]] = {
            "ref_root_pos": [],
            "ref_root_quat": [],
            "ref_dof_mujoco": [],
            "tracking_root_pos": [],
            "tracking_root_quat": [],
            "tracking_dof_mujoco": [],
            "policy_action_mean": [],
            "policy_action_sample": [],
            "policy_action_used": [],
            "tokenizer_encoder_index": [],
            "reward": [],
            "done": [],
            "root_error": [],
            "joint_mae": [],
        }

        with torch.inference_mode():
            for _ in range(int(args.num_steps)):
                ref_root_pos = command.anchor_pos_w[0].detach().cpu().numpy().astype(np.float32)
                ref_root_quat = command.anchor_quat_w[0].detach().cpu().numpy().astype(np.float32)
                ref_dof_mujoco = command.joint_pos[0].detach().cpu().numpy().astype(np.float32)[isaac_to_mujoco]

                tracking_root_pos = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
                tracking_root_quat = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float32)
                tracking_dof_mujoco = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)[
                    isaac_to_mujoco
                ]

                frames["ref_root_pos"].append(ref_root_pos)
                frames["ref_root_quat"].append(ref_root_quat)
                frames["ref_dof_mujoco"].append(ref_dof_mujoco)
                frames["tracking_root_pos"].append(tracking_root_pos)
                frames["tracking_root_quat"].append(tracking_root_quat)
                frames["tracking_dof_mujoco"].append(tracking_dof_mujoco)
                frames["root_error"].append(float(np.linalg.norm(ref_root_pos - tracking_root_pos)))
                frames["joint_mae"].append(float(np.mean(np.abs(ref_dof_mujoco - tracking_dof_mujoco))))

                tokenizer = obs_dict.get("tokenizer")
                if tokenizer is not None:
                    frames["tokenizer_encoder_index"].append(
                        tokenizer[0, :3].detach().cpu().numpy().astype(np.float32)
                    )

                policy.init_rollout()
                policy_state = policy.rollout(obs_dict=obs_dict)
                action_mean = policy_state["action_mean"].detach()
                action_sample = policy_state["actions"].detach()
                action = action_sample if args.action_mode == "sample" else action_mean
                frames["policy_action_mean"].append(action_mean[0].detach().cpu().numpy().astype(np.float32))
                frames["policy_action_sample"].append(action_sample[0].detach().cpu().numpy().astype(np.float32))
                frames["policy_action_used"].append(action[0].detach().cpu().numpy().astype(np.float32))
                obs_dict, rew, dones, _ = env.step({"actions": action})
                frames["reward"].append(float(rew[0].detach().cpu().item()))
                frames["done"].append(bool(dones[0].detach().cpu().item()))

        traj_path = output_dir / "trajectory_g1_release_h2adapted.npz"
        np.savez_compressed(
            traj_path,
            **{key: np.asarray(value) for key, value in frames.items()},
            fps=np.array(float(args.fps), dtype=np.float32),
            checkpoint=np.array(str(Path(args.checkpoint).expanduser().resolve())),
            motion_file=np.array(str(robot_motion_file)),
            smpl_motion_file=np.array(str(smpl_motion_file)),
            motion_key=np.array(args.motion_key),
            robot_joint_names=np.asarray(robot.joint_names),
            isaac_to_mujoco_dof=np.asarray(isaac_to_mujoco, dtype=np.int64),
            config_yaml=np.array(OmegaConf.to_yaml(config, resolve=False)),
            label=np.array(args.label or ""),
        )
        (output_dir / "collect_meta.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
                    "motion_file": str(robot_motion_file),
                    "smpl_motion_file": str(smpl_motion_file),
                    "motion_key": args.motion_key,
                    "num_steps": int(args.num_steps),
                    "fps": float(args.fps),
                    "action_mode": args.action_mode,
                    "mean_root_error": float(np.mean(frames["root_error"])),
                    "mean_joint_mae": float(np.mean(frames["joint_mae"])),
                    "mean_policy_action_used_abs": float(np.mean(np.abs(frames["policy_action_used"]))),
                    "done_count": int(np.sum(frames["done"])),
                },
                indent=2,
            )
        )
        print(f"[g1-h2adapted-viz] Wrote trajectory {traj_path}", flush=True)
        if args.collect_only:
            print(f"COLLECTED_TRAJECTORY={traj_path}", flush=True)
            print(f"OUTPUT_DIR={output_dir}", flush=True)
            os._exit(0)
        return traj_path
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()
        simulation_app.close()


def _build_dual_g1_xml() -> str:
    source = ET.parse(G1_MJCF).getroot()
    scene = ET.Element("mujoco", {"model": "g1_release_h2adapted_ref_tracking"})
    ET.SubElement(scene, "compiler", {"angle": "radian"})
    ET.SubElement(scene, "option", {"timestep": "0.02", "gravity": "0 0 -9.81"})

    asset = ET.SubElement(scene, "asset")
    ET.SubElement(
        asset,
        "texture",
        {"type": "skybox", "builtin": "gradient", "rgb1": "0.78 0.86 0.94", "rgb2": "1 1 1", "width": "256", "height": "256"},
    )
    ET.SubElement(
        asset,
        "texture",
        {"name": "grid", "type": "2d", "builtin": "checker", "rgb1": "0.92 0.92 0.90", "rgb2": "0.78 0.80 0.82", "width": "512", "height": "512"},
    )
    ET.SubElement(asset, "material", {"name": "grid", "texture": "grid", "texrepeat": "4 4", "reflectance": "0.15"})
    for mesh in source.find("asset").findall("mesh"):
        mesh_copy = copy.deepcopy(mesh)
        mesh_copy.set("file", str(G1_MESH_DIR / mesh_copy.attrib["file"]))
        asset.append(mesh_copy)

    worldbody = ET.SubElement(scene, "worldbody")
    ET.SubElement(worldbody, "geom", {"name": "floor", "type": "plane", "size": "8 8 0.02", "material": "grid", "rgba": "0.82 0.84 0.86 1"})
    ET.SubElement(worldbody, "light", {"name": "key", "pos": "0 -3 5", "dir": "0 0 -1", "diffuse": "0.8 0.8 0.8"})
    ET.SubElement(worldbody, "light", {"name": "fill", "pos": "-3 3 4", "dir": "0 0 -1", "diffuse": "0.35 0.35 0.35"})

    robot_body = source.find("worldbody").find("body")
    if robot_body is None:
        raise RuntimeError(f"No robot body found in {G1_MJCF}")

    ref_body = copy.deepcopy(robot_body)
    _prefix_names(ref_body, "ref_")
    _set_robot_rgba(ref_body, "0.10 0.35 1.00 0.42")
    worldbody.append(ref_body)

    tracking_body = copy.deepcopy(robot_body)
    _prefix_names(tracking_body, "tracking_")
    _set_robot_rgba(tracking_body, "1.00 0.34 0.04 0.92")
    worldbody.append(tracking_body)
    return ET.tostring(scene, encoding="unicode")


def render_video(args: argparse.Namespace, output_dir: Path, traj_path: Path | None = None) -> Path:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio.v2 as imageio
    import mujoco

    try:
        import cv2
    except ImportError:
        cv2 = None

    traj_path = Path(traj_path or args.trajectory).expanduser().resolve()
    traj = {key: value for key, value in np.load(traj_path, allow_pickle=True).items()}
    label = args.label or str(np.asarray(traj.get("label", np.array(""))).item())

    model = mujoco.MjModel.from_xml_string(_build_dual_g1_xml())
    data = mujoco.MjData(model)
    if model.nq != 72:
        raise RuntimeError(f"Expected two 36-qpos G1 robots, got model.nq={model.nq}")

    width = int(args.width)
    height = int(args.height)
    trajectory_fps = _trajectory_fps(traj, fallback=float(args.fps))
    fps = float(args.fps) if float(args.fps) > 0 else trajectory_fps
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    model.vis.quality.shadowsize = 2048
    model.vis.quality.offsamples = 4
    model.vis.headlight.ambient = [0.55, 0.55, 0.55]
    model.vis.headlight.diffuse = [0.8, 0.8, 0.8]
    model.vis.headlight.specular = [0.1, 0.1, 0.1]

    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = float(args.camera_distance)
    cam.azimuth = float(args.camera_azimuth)
    cam.elevation = float(args.camera_elevation)

    ref_root_pos = traj["ref_root_pos"].astype(np.float64)
    tracking_root_pos = traj["tracking_root_pos"].astype(np.float64)
    ref_root_quat = traj["ref_root_quat"].astype(np.float64)
    tracking_root_quat = traj["tracking_root_quat"].astype(np.float64)
    ref_dof = traj["ref_dof_mujoco"].astype(np.float64)
    tracking_dof = traj["tracking_dof_mujoco"].astype(np.float64)

    origin = ref_root_pos[0].copy()
    origin[2] = 0.0
    nframes = min(len(ref_root_pos), int(args.max_render_frames or len(ref_root_pos)))
    video_path = output_dir / "g1_release_h2adapted_ref_tracking.mp4"
    preview_path = output_dir / "preview.png"
    writer = imageio.get_writer(
        video_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=8,
    )
    try:
        for idx in range(nframes):
            ref_pos = ref_root_pos[idx] - origin
            tracking_pos = tracking_root_pos[idx] - origin
            data.qpos[:] = 0.0
            data.qpos[0:3] = ref_pos
            data.qpos[3:7] = ref_root_quat[idx]
            data.qpos[7:36] = ref_dof[idx]
            data.qpos[36:39] = tracking_pos
            data.qpos[39:43] = tracking_root_quat[idx]
            data.qpos[43:72] = tracking_dof[idx]

            mujoco.mj_forward(model, data)
            lookat = 0.5 * (ref_pos + tracking_pos)
            lookat[2] = max(0.75, lookat[2])
            cam.lookat[:] = lookat
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()

            if cv2 is not None:
                root_err = float(np.linalg.norm(ref_root_pos[idx] - tracking_root_pos[idx]))
                joint_mae = float(np.mean(np.abs(ref_dof[idx] - tracking_dof[idx])))
                text = [
                    f"G1 release on H2->G1 ref {label}".strip(),
                    "blue: adapted G1 reference   orange: release policy tracking",
                    f"frame {idx + 1}/{nframes}  root_err {root_err:.3f} m  joint_mae {joint_mae:.3f} rad",
                ]
                y = 32
                for line in text:
                    cv2.putText(frame, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (24, 24, 24), 3, cv2.LINE_AA)
                    cv2.putText(frame, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 1, cv2.LINE_AA)
                    y += 30

            if idx == 0:
                imageio.imwrite(preview_path, frame)
            writer.append_data(frame)
    finally:
        writer.close()
        renderer.close()

    ref_xy = float(np.linalg.norm(ref_root_pos[nframes - 1, :2] - ref_root_pos[0, :2]))
    tracking_xy = float(np.linalg.norm(tracking_root_pos[nframes - 1, :2] - tracking_root_pos[0, :2]))
    metrics = {
        "trajectory": str(traj_path),
        "video": str(video_path),
        "preview": str(preview_path),
        "frames": int(nframes),
        "fps": fps,
        "trajectory_fps": trajectory_fps,
        "duration_s": float(nframes / fps) if fps > 0 else None,
        "trajectory_duration_s": float(nframes / trajectory_fps) if trajectory_fps > 0 else None,
        "ref_xy_displacement_m": ref_xy,
        "tracking_xy_displacement_m": tracking_xy,
        "mean_root_error_m": float(np.mean(np.linalg.norm(ref_root_pos[:nframes] - tracking_root_pos[:nframes], axis=-1))),
        "mean_joint_mae_rad": float(np.mean(np.abs(ref_dof[:nframes] - tracking_dof[:nframes]))),
        "max_root_error_m": float(np.max(np.linalg.norm(ref_root_pos[:nframes] - tracking_root_pos[:nframes], axis=-1))),
    }
    (output_dir / "render_meta.json").write_text(json.dumps(metrics, indent=2))
    return video_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--h2-motion-file", default=str(DEFAULT_H2_MOTION_FILE))
    parser.add_argument("--h2-smpl-motion-file", default=str(DEFAULT_H2_SMPL_MOTION_FILE))
    parser.add_argument("--motion-key", default="walk_forward_loop_002__A036")
    parser.add_argument("--num-steps", type=int, default=360)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-render-frames", type=int, default=0)
    parser.add_argument("--action-mode", choices=("mean", "sample"), default="mean")
    parser.add_argument(
        "--use-encoder",
        choices=("g1", "teleop", "smpl"),
        default="g1",
        help="Force the release universal-token encoder used for the diagnostic rollout.",
    )
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--keep-terminations", dest="disable_terminations", action="store_false")
    parser.add_argument("--keep-events", dest="disable_events", action="store_false")
    parser.set_defaults(disable_terminations=True, disable_events=True)
    args = parser.parse_args()
    if args.collect_only and args.render_only:
        raise ValueError("--collect-only and --render-only are mutually exclusive")
    if args.render_only and args.trajectory is None:
        raise ValueError("--render-only requires --trajectory")
    return args


def main() -> None:
    os.chdir(REPO_ROOT)
    args = parse_args()
    output_dir = _prepare_output_dir(args)
    if args.render_only:
        video_path = render_video(args, output_dir, Path(args.trajectory))
        print(f"RENDERED_VIDEO={video_path}")
        print(f"OUTPUT_DIR={output_dir}")
        return

    robot_motion_file, smpl_motion_file = prepare_g1_motion_from_h2(args, output_dir)
    print(f"PREPARED_G1_ROBOT_MOTION={robot_motion_file}")
    print(f"PREPARED_G1_SMPL_MOTION={smpl_motion_file}")
    if args.prepare_only:
        print(f"OUTPUT_DIR={output_dir}")
        return

    traj_path = collect_rollout(args, output_dir, robot_motion_file, smpl_motion_file)
    print(f"COLLECTED_TRAJECTORY={traj_path}")
    if not args.collect_only:
        video_path = render_video(args, output_dir, traj_path)
        print(f"RENDERED_VIDEO={video_path}")
    print(f"OUTPUT_DIR={output_dir}")


if __name__ == "__main__":
    main()
