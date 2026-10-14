#!/usr/bin/env python3
"""Collect and render H2LoRA reference/tracking videos.

The script intentionally has two separable stages:

1. ``--collect-only`` runs under Isaac Sim Python, loads a H2LoRA checkpoint,
   and saves a one-env rollout as ``trajectory_ref_tracking.npz``.
2. ``--render-only`` runs under the lightweight uv environment, composes two
   colored H2 MuJoCo robots, and writes an MP4.

Running without either flag does both stages in one process if the runtime has
both IsaacLab and MuJoCo available.
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
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "logs_rl/sonic-h2lora/manager/universal_token/all_modes/"
    "sonic_h2lora_h2lora-20260531_232745/last.pt"
)
DEFAULT_MOTION_FILE = Path("/mnt/data/seed/h2lora_motion_lib/idle_debug/robot")
DEFAULT_SMPL_MOTION_FILE = Path("/mnt/data/seed/h2lora_motion_lib/idle_debug/smpl")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "visualizations/h2lora"
H2_MJCF = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/h2.xml"
H2_MESH_DIR = REPO_ROOT / "gear_sonic/data/assets/robot_description/urdf/h2/meshes"
DEFAULT_ONNX_ENCODER = REPO_ROOT / "sonic_h2lora_release/exported/model_encoder.onnx"
DEFAULT_ONNX_DECODER = REPO_ROOT / "sonic_h2lora_release/exported/model_decoder_latest.onnx"
ONNX_ENCODER_MODE_IDS = {"g1": 0, "teleop": 1, "smpl": 2}
ONNX_ENCODER_TOKENIZER_FEATURES = [
    "encoder_index",
    "command_multi_future_nonflat",
    "motion_anchor_ori_b",
    "motion_anchor_ori_b_mf_nonflat",
    "command_multi_future_lower_body",
    "vr_3point_local_target",
    "vr_3point_local_orn_target",
    "smpl_joints_multi_future_local_nonflat",
    "smpl_root_ori_b_multi_future",
    "joint_pos_multi_future_wrist_for_smpl",
]


def _h2_mujoco_joint_names() -> list[str]:
    """Return H2 actuated joint names in MuJoCo qpos order."""
    root = ET.parse(H2_MJCF).getroot()
    names = []
    for joint in root.findall(".//joint"):
        name = joint.attrib["name"]
        if name != "floating_base_joint":
            names.append(name)
    return names


def _isaac_to_mujoco_dof_from_names(isaac_joint_names: list[str]) -> np.ndarray:
    """Build indices that gather an IsaacLab joint vector into MuJoCo joint order."""
    missing = [name for name in _h2_mujoco_joint_names() if name not in isaac_joint_names]
    if missing:
        raise ValueError(f"H2 MuJoCo joints missing from IsaacLab asset: {missing}")
    return np.asarray([isaac_joint_names.index(name) for name in _h2_mujoco_joint_names()], dtype=np.int64)


def _sanitize_name(name: str) -> str:
    name = name.strip().replace(os.sep, "__")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("_") or "h2lora_ref_tracking"


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _prepare_output_dir(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).expanduser().resolve()
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    experiment_name = args.experiment_name
    if not experiment_name and args.checkpoint is not None:
        checkpoint = Path(args.checkpoint)
        experiment_name = re.sub(r"-\d{8}_\d{6}$", "", checkpoint.parent.name)
    experiment_name = _sanitize_name(experiment_name or "h2lora_ref_tracking")
    stamp = args.timestamp or _timestamp()
    output_dir = output_root / f"{experiment_name}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _rewrite_training_config(raw: str) -> str:
    """Match the compatibility rewrites used by ``eval_agent_trl.py``."""
    replacements = {
        "groot.rl.trl.": "gear_sonic.trl.",
        "groot.rl.envs.": "gear_sonic.envs.",
        "groot.rl.utils.": "gear_sonic.utils.",
        "groot.rl.agents.modules.modules.": "gear_sonic.trl.modules.base_module.",
        "groot.rl.agents.": "gear_sonic.trl.",
        "groot/rl/data/": "gear_sonic/data/",
        "assets/bm/unitree_description/": "assets/robot_description/",
        "1215_bones_seed_filtered": "bones_seed_smpl",
    }
    for src, dst in replacements.items():
        raw = raw.replace(src, dst)
    return raw


def _null_cfg_terms(cfg_node) -> None:
    """Disable Hydra dataclass cfg terms while keeping the cfg object target."""
    from omegaconf import open_dict

    if cfg_node is None:
        return
    with open_dict(cfg_node):
        for key in list(cfg_node.keys()):
            if key != "_target_":
                cfg_node[key] = None


def _load_config_for_checkpoint(args: argparse.Namespace):
    from omegaconf import OmegaConf, open_dict

    from gear_sonic.utils import config_utils

    config_utils.register_rl_resolvers()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config_path = checkpoint.parent / "config.yaml"
    if not config_path.exists():
        config_path = checkpoint.parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Cannot find config.yaml next to checkpoint: {checkpoint}")

    os.environ.setdefault("H2LORA_MOTION_FILE", str(Path(args.motion_file).expanduser()))
    os.environ.setdefault("H2LORA_SMPL_MOTION_FILE", str(Path(args.smpl_motion_file).expanduser()))
    os.environ.setdefault("SWANLAB_MODE", "disabled")
    os.environ.setdefault("WANDB_MODE", "disabled")

    raw = _rewrite_training_config(config_path.read_text())
    config = OmegaConf.load(io.StringIO(raw))
    with open_dict(config):
        config.checkpoint = str(checkpoint)
        config.experiment_dir = str(checkpoint.parent)
        config.headless = True
        config.num_envs = 1
        config.use_wandb = False
        config.use_swanlab = False
        config.multi_gpu = False
        config.global_rank = 0
        config.output_dir = str(checkpoint.parent / "visualization_output")
        if "algo" in config and "trl" in config.algo:
            config.algo.trl.output_dir = str(checkpoint.parent)
        if "callbacks" in config:
            config.callbacks = {}

        motion_cfg = config.manager_env.commands.motion
        motion_cfg.start_from_first_frame = True
        motion_cfg.freeze_frame_aug = False
        if args.motion_key:
            motion_cfg.filter_motion_keys = [args.motion_key]
            if "motion_lib_cfg" in motion_cfg:
                motion_cfg.motion_lib_cfg.filter_motion_keys = [args.motion_key]
        if "motion_lib_cfg" in motion_cfg:
            motion_cfg.motion_lib_cfg.motion_file = str(Path(args.motion_file).expanduser())
            motion_cfg.motion_lib_cfg.smpl_motion_file = str(Path(args.smpl_motion_file).expanduser())
            motion_cfg.motion_lib_cfg.cache_all_motions_to_ddr = True
            motion_cfg.motion_lib_cfg.max_cpu_workers = min(
                int(motion_cfg.motion_lib_cfg.get("max_cpu_workers", 16)), 16
            )
        if args.use_encoder is not None:
            probs = motion_cfg.encoder_sample_probs
            for encoder_name in list(probs.keys()):
                probs[encoder_name] = 1.0 if encoder_name == args.use_encoder else 0.0

        if args.disable_terminations:
            _null_cfg_terms(config.manager_env.terminations)
        if args.disable_events:
            _null_cfg_terms(config.manager_env.events)

    return config


def _make_app_args(config, app_launcher_cls=None):
    if app_launcher_cls is not None:
        parser = argparse.ArgumentParser(add_help=False)
        app_launcher_cls.add_app_launcher_args(parser)
        args_cli = parser.parse_args([])
    else:
        args_cli = SimpleNamespace()

    args_cli.num_envs = int(config.num_envs)
    args_cli.seed = int(config.seed)
    args_cli.env_spacing = float(config.manager_env.config.env_spacing)
    args_cli.output_dir = str(config.output_dir)
    args_cli.enable_cameras = False
    args_cli.headless = True
    args_cli.multi_gpu = False
    args_cli.distributed = False
    args_cli.device = "cuda:0"
    args_cli.kit_args = (
        "--/log/level=error --/log/fileLogLevel=error "
        "--/log/outputStreamLevel=error "
        "--enable isaacsim.asset.importer.urdf --enable isaacsim.asset.importer.mjcf --no-window"
    )
    return args_cli


def _setup_model(config, env, device: str):
    import torch

    from gear_sonic.trl.utils import common as trl_utils_common
    from gear_sonic.trl.utils.lora import apply_sonic_h2lora
    from gear_sonic.utils import obs_utils

    module_dim_dict = getattr(config.algo.config, "module_dim", {})
    env.config["obs"]["obs_dims"]["actor_obs"] = env.env.observation_space["policy"].shape[-1]
    env.config["obs"]["obs_dims"]["critic_obs"] = env.env.observation_space["critic"].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["actor_obs"] = env.env.observation_space[
        "policy"
    ].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["critic_obs"] = env.env.observation_space[
        "critic"
    ].shape[-1]

    example_obs = env.reset(flatten_dict_obs=False)
    for key in env.env.observation_space:
        if key in {"policy", "critic"}:
            continue
        group_obs_dims, group_obs_names, group_obs_total_dim = obs_utils.get_group_term_obs_shape(
            example_obs, key
        )
        env.config["obs"]["group_obs_dims"][key] = group_obs_dims
        env.config["obs"]["group_obs_names"][key] = group_obs_names
        env.config["obs"]["obs_dims"][key] = group_obs_total_dim
        env.config["robot"]["algo_obs_dim_dict"][key] = group_obs_total_dim

    if config.manager_env.config.get("policy_action_dim", None) is not None:
        env.config["robot"]["actions_dim"] = config.manager_env.config.policy_action_dim
    elif config.manager_env.config.get("meta_action_dim", None) is not None:
        env.config["robot"]["actions_dim"] = config.manager_env.config.meta_action_dim
    elif env.config.get("meta_action_dim", None) is not None and env.config.get("meta_action_dim") > 0:
        env.config["robot"]["actions_dim"] = env.config.get("meta_action_dim")
    else:
        env.config["robot"]["actions_dim"] = env.env.action_space.shape[-1]

    policy = trl_utils_common.custom_instantiate(
        config.algo.config.actor,
        env_config=env.config,
        algo_config=config.algo.config,
        module_dim_dict=module_dim_dict,
        backbone_kwargs={},
        _resolve=False,
    ).to(device)

    value_model = trl_utils_common.custom_instantiate(
        config.algo.config.critic,
        env_config=env.config,
        algo_config=config.algo.config,
        module_dim_dict=module_dim_dict,
        backbone_kwargs={},
        _resolve=False,
    ).to(device)

    if config.algo.config.get("lora_adaptation", None) is not None:
        apply_sonic_h2lora(policy, value_model, config.algo.config.lora_adaptation)
        policy.to(device)
        value_model.to(device)

    from gear_sonic.train_agent_trl import _install_trl_checkpoint_compat

    _install_trl_checkpoint_compat()
    checkpoint = torch.load(config.checkpoint, map_location=device, weights_only=False)
    if "actor_model_state_dict" in checkpoint:
        state_dict = checkpoint["actor_model_state_dict"]
    elif "policy_state_dict" in checkpoint:
        state_dict = checkpoint["policy_state_dict"]
    else:
        raise KeyError("Checkpoint does not contain actor_model_state_dict or policy_state_dict")

    model_uses_std = "std" in policy.state_dict()
    checkpoint_has_std = "std" in state_dict
    checkpoint_has_log_std = "log_std" in state_dict
    if model_uses_std and checkpoint_has_log_std and not checkpoint_has_std:
        state_dict = dict(state_dict)
        state_dict["std"] = torch.exp(state_dict.pop("log_std"))
    elif not model_uses_std and checkpoint_has_std and not checkpoint_has_log_std:
        state_dict = dict(state_dict)
        state_dict["log_std"] = torch.log(state_dict.pop("std"))

    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    for param in policy.parameters():
        param.requires_grad = False
    return policy


def _ensure_env_obs_metadata(env) -> None:
    """Populate wrapper obs metadata without constructing the PyTorch policy."""
    from gear_sonic.utils import obs_utils

    example_obs = env.reset(flatten_dict_obs=False)
    for key in env.env.observation_space:
        if key in {"policy", "critic"}:
            continue
        group_obs_dims, group_obs_names, group_obs_total_dim = obs_utils.get_group_term_obs_shape(
            example_obs, key
        )
        env.config["obs"]["group_obs_dims"][key] = group_obs_dims
        env.config["obs"]["group_obs_names"][key] = group_obs_names
        env.config["obs"]["obs_dims"][key] = group_obs_total_dim
        env.config["robot"]["algo_obs_dim_dict"][key] = group_obs_total_dim


def _import_onnxruntime():
    try:
        import onnxruntime as ort

        return ort
    except ImportError:
        isaac_ort_site = Path(os.environ.get("ISAAC_ONNXRUNTIME_SITE", "/tmp/isaac_ort311"))
        if isaac_ort_site.exists() and str(isaac_ort_site) not in sys.path:
            sys.path.insert(0, str(isaac_ort_site))
        venv_site = (
            REPO_ROOT
            / ".venv-h2lora"
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        if venv_site.exists() and str(venv_site) not in sys.path:
            sys.path.insert(0, str(venv_site))
        import onnxruntime as ort

        return ort


def _setup_onnx_policy(args: argparse.Namespace):
    ort = _import_onnxruntime()
    providers = ["CPUExecutionProvider"]
    encoder_path = Path(args.onnx_encoder).expanduser().resolve()
    decoder_path = Path(args.onnx_decoder).expanduser().resolve()
    if not encoder_path.exists():
        raise FileNotFoundError(f"ONNX encoder does not exist: {encoder_path}")
    if not decoder_path.exists():
        raise FileNotFoundError(f"ONNX decoder does not exist: {decoder_path}")
    encoder = ort.InferenceSession(str(encoder_path), providers=providers)
    decoder = ort.InferenceSession(str(decoder_path), providers=providers)
    encoder_shape = list(encoder.get_inputs()[0].shape)
    decoder_shape = list(decoder.get_inputs()[0].shape)
    if encoder_shape[-1] != 1751:
        raise RuntimeError(f"Expected H2LoRA encoder input width 1751, got {encoder_shape}")
    if decoder_shape[-1] != 994:
        raise RuntimeError(f"Expected H2LoRA decoder input width 994, got {decoder_shape}")
    print(
        f"[h2lora-viz] Using ONNX policy encoder={encoder_path} decoder={decoder_path}",
        flush=True,
    )
    return encoder, decoder


def _split_tokenizer_obs(env, tokenizer_tensor):
    tokenizer_np = tokenizer_tensor.detach().cpu().numpy().astype(np.float32)
    names = list(env.config["obs"]["group_obs_names"]["tokenizer"])
    dims = env.config["obs"]["group_obs_dims"]["tokenizer"]
    out = {}
    offset = 0
    for name in names:
        shape = tuple(int(v) for v in dims[name])
        width = int(np.prod(shape))
        out[name] = tokenizer_np[:, offset : offset + width].reshape((tokenizer_np.shape[0],) + shape)
        offset += width
    if offset != tokenizer_np.shape[-1]:
        raise RuntimeError(
            f"Tokenizer split consumed {offset} dims, tensor has {tokenizer_np.shape[-1]}"
        )
    return out


def _onnx_encoder_input(env, obs_dict, encoder_name: str | None):
    tokenizer = obs_dict.get("tokenizer")
    if tokenizer is None:
        raise KeyError("obs_dict does not contain tokenizer observations")
    split = _split_tokenizer_obs(env, tokenizer)
    if encoder_name is not None:
        mode_id = ONNX_ENCODER_MODE_IDS[encoder_name]
        split["encoder_index"] = np.zeros_like(split["encoder_index"], dtype=np.float32)
        split["encoder_index"][:, mode_id] = 1.0
    else:
        mode_id = int(np.argmax(split["encoder_index"][0]))

    flat_terms = []
    for name in ONNX_ENCODER_TOKENIZER_FEATURES:
        if name not in split:
            raise KeyError(f"Tokenizer observation '{name}' is missing from IsaacLab obs")
        flat_terms.append(split[name].reshape(split[name].shape[0], -1).astype(np.float32))
    encoder_input = np.concatenate(
        [np.full((flat_terms[0].shape[0], 1), float(mode_id), dtype=np.float32), *flat_terms],
        axis=-1,
    )
    if encoder_input.shape[-1] != 1751:
        raise RuntimeError(f"Expected ONNX encoder input width 1751, got {encoder_input.shape}")
    return encoder_input


def _run_onnx_policy(env, obs_dict, encoder, decoder, encoder_name: str | None):
    encoder_input = _onnx_encoder_input(env, obs_dict, encoder_name)
    encoder_name_in = encoder.get_inputs()[0].name
    decoder_name_in = decoder.get_inputs()[0].name
    token_state = encoder.run(None, {encoder_name_in: encoder_input})[0].astype(np.float32)
    actor_obs = obs_dict["actor_obs"].detach().cpu().numpy().astype(np.float32)
    policy_input = np.concatenate([token_state, actor_obs], axis=-1).astype(np.float32)
    if policy_input.shape[-1] != 994:
        raise RuntimeError(f"Expected ONNX decoder input width 994, got {policy_input.shape}")
    action = decoder.run(None, {decoder_name_in: policy_input})[0].astype(np.float32)
    return action, encoder_input, token_state, policy_input, actor_obs


def _clip_onnx_policy_action(action: np.ndarray, clip_value: float) -> np.ndarray:
    if clip_value is None or clip_value <= 0:
        return action.astype(np.float32, copy=False)
    return np.clip(action, -float(clip_value), float(clip_value)).astype(np.float32, copy=False)


def _first_motion_key(env) -> str | None:
    command = env.motion_command
    motion_lib = getattr(command, "motion_lib", None)
    motion_keys = getattr(motion_lib, "motion_keys", None)
    if motion_keys is None:
        motion_keys = getattr(motion_lib, "_motion_keys", None)
    if motion_keys is None:
        return None
    try:
        motion_id = int(command.motion_ids[0].detach().cpu().item())
        return str(motion_keys[motion_id])
    except Exception:
        return None


def _load_external_policy_actions(path: str | None) -> np.ndarray:
    if path is None:
        raise ValueError("--external-action-trajectory is required for external-g1-actions")
    traj_path = Path(path).expanduser().resolve()
    if not traj_path.exists():
        raise FileNotFoundError(f"External action trajectory does not exist: {traj_path}")
    data = np.load(traj_path, allow_pickle=True)
    if "policy_action_used" not in data:
        raise KeyError(f"{traj_path} does not contain policy_action_used")
    actions = np.asarray(data["policy_action_used"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] != 29:
        raise ValueError(f"Expected external 29D policy_action_used, got {actions.shape}")
    return actions


def _broadcast_action_tensor(value, num_envs: int, num_joints: int, dtype, device, default: float):
    import torch

    if torch.is_tensor(value):
        tensor = value.to(device=device, dtype=dtype)
    else:
        tensor = torch.full((num_joints,), float(default), dtype=dtype, device=device)
    if tensor.ndim == 0:
        tensor = tensor.expand(num_joints)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0).expand(num_envs, -1)
    elif tensor.shape[0] == 1 and num_envs > 1:
        tensor = tensor.expand(num_envs, -1)
    if tensor.shape[-1] != num_joints:
        raise ValueError(f"Action tensor has incompatible shape {tuple(tensor.shape)}")
    return tensor


def _h2_ref_pd_sim_action(env, command, robot, body_indices, head_indices, head_defaults):
    """Convert H2 physical MotionLib joint targets into the simulator action space."""
    import torch

    num_envs = robot.data.joint_pos.shape[0]
    num_joints = robot.num_joints
    dtype = robot.data.joint_pos.dtype
    device = robot.data.joint_pos.device
    action_term = env.env.action_manager.get_term("joint_pos")
    scale = _broadcast_action_tensor(
        getattr(action_term, "_scale", None), num_envs, num_joints, dtype, device, default=1.0
    )
    offset = _broadcast_action_tensor(
        getattr(action_term, "_offset", None), num_envs, num_joints, dtype, device, default=0.0
    )
    safe_scale = torch.where(torch.abs(scale) > 1.0e-8, scale, torch.ones_like(scale))

    target_joint_pos = robot.data.default_joint_pos.clone()
    body_idx = torch.as_tensor(body_indices, dtype=torch.long, device=device)
    target_joint_pos[:, body_idx] = command.joint_pos_physical.to(device=device, dtype=dtype)
    if len(head_indices) > 0:
        head_idx = torch.as_tensor(head_indices, dtype=torch.long, device=device)
        head_default_tensor = torch.as_tensor(
            head_defaults, dtype=dtype, device=device
        ).unsqueeze(0)
        target_joint_pos[:, head_idx] = head_default_tensor
    return (target_joint_pos - offset) / safe_scale


def _step_sim_action_direct(env, sim_action):
    """Step the underlying IsaacLab env with full simulator actions."""
    import torch

    obs, rew, terminated, truncated, extras = env.env.step(sim_action)
    if hasattr(env, "_lock_h2lora_head_state"):
        env._lock_h2lora_head_state()  # noqa: SLF001
    dones = (terminated | truncated).to(dtype=torch.long)
    obs_dict = env.process_raw_obs(obs, flatten_dict_obs=True)
    return obs_dict, rew, dones, extras


def collect_rollout(args: argparse.Namespace, output_dir: Path) -> Path:
    """Collect a single-env H2LoRA rollout with IsaacLab."""
    import torch

    from filelock import FileLock
    from omegaconf import OmegaConf

    from gear_sonic.utils.common import seeding

    config = _load_config_for_checkpoint(args)
    seeding(int(config.seed))
    print(f"[h2lora-viz] Launching IsaacLab app; output_dir={output_dir}", flush=True)

    from isaaclab.app import AppLauncher

    app_args = _make_app_args(config, AppLauncher)
    with FileLock("/tmp/isaaclab_app_launcher.lock"):
        app_launcher = AppLauncher(app_args)
    simulation_app = app_launcher.app
    print("[h2lora-viz] IsaacLab app launched", flush=True)

    from gear_sonic import train_agent_trl
    from gear_sonic.envs.env_utils import joint_utils

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    env = None
    try:
        print("[h2lora-viz] Creating manager env", flush=True)
        env = train_agent_trl.create_manager_env(config, device, app_args)
        if hasattr(env, "set_is_evaluating"):
            env.set_is_evaluating(True)
        if hasattr(env, "reinit_dr"):
            env.reinit_dr()

        policy = None
        onnx_encoder = None
        onnx_decoder = None
        external_actions = None
        if args.action_source == "policy":
            print("[h2lora-viz] Loading policy checkpoint", flush=True)
            policy = _setup_model(config, env, device)
        elif args.action_source == "onnx-policy":
            _ensure_env_obs_metadata(env)
            onnx_encoder, onnx_decoder = _setup_onnx_policy(args)
        elif args.action_source == "external-g1-actions":
            external_actions = _load_external_policy_actions(args.external_action_trajectory)
            print(
                f"[h2lora-viz] Using external 29D G1 policy actions: "
                f"{args.external_action_trajectory} ({len(external_actions)} frames)",
                flush=True,
            )
        else:
            print("[h2lora-viz] Using H2 reference PD joint targets; policy is not loaded", flush=True)
        obs_dict = env.reset_all()
        print("[h2lora-viz] Collecting rollout frames", flush=True)

        robot = env.env.scene["robot"]
        command = env.motion_command
        body_indices = joint_utils.get_h2lora_body_joint_indices(robot).detach().cpu().numpy()
        head_indices = joint_utils.get_h2_head_joint_indices(robot).detach().cpu().numpy()
        head_defaults = np.array(
            [
                float(env.config.get("direct_joint_defaults", {}).get("head_pitch_joint", 0.0)),
                float(env.config.get("direct_joint_defaults", {}).get("head_yaw_joint", 0.0)),
            ],
            dtype=np.float32,
        )
        isaac_to_mujoco = _isaac_to_mujoco_dof_from_names(robot.joint_names)

        frames: dict[str, list[np.ndarray | float | bool]] = {
            "ref_root_pos": [],
            "ref_root_quat": [],
            "ref_dof_isaac": [],
            "ref_dof_mujoco": [],
            "tracking_root_pos": [],
            "tracking_root_quat": [],
            "tracking_dof_isaac": [],
            "tracking_dof_mujoco": [],
            "policy_action_mean": [],
            "policy_action_sample": [],
            "policy_action_used": [],
            "policy_action_raw": [],
            "sim_action_used": [],
            "tokenizer_encoder_index": [],
            "actor_obs": [],
            "tokenizer_flat": [],
            "onnx_encoder_input": [],
            "onnx_token_state": [],
            "onnx_policy_input": [],
            "reward": [],
            "done": [],
            "root_error": [],
            "joint_mae": [],
        }

        with torch.inference_mode():
            zero_policy_action = torch.zeros(1, 29, dtype=torch.float32, device=device)
            for step_idx in range(int(args.num_steps)):
                ref_root_pos = command.anchor_pos_w[0].detach().cpu().numpy().astype(np.float32)
                ref_root_quat = command.anchor_quat_w[0].detach().cpu().numpy().astype(np.float32)

                ref_full_isaac = np.zeros(robot.num_joints, dtype=np.float32)
                ref_full_isaac[body_indices] = command.joint_pos_physical[0].detach().cpu().numpy()
                if len(head_indices) == 2:
                    ref_full_isaac[head_indices] = head_defaults
                ref_dof_mujoco = ref_full_isaac[isaac_to_mujoco]

                tracking_root_pos = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
                tracking_root_quat = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float32)
                tracking_dof_isaac = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
                tracking_dof_mujoco = tracking_dof_isaac[isaac_to_mujoco]

                frames["ref_root_pos"].append(ref_root_pos)
                frames["ref_root_quat"].append(ref_root_quat)
                frames["ref_dof_isaac"].append(ref_full_isaac.astype(np.float32))
                frames["ref_dof_mujoco"].append(ref_dof_mujoco.astype(np.float32))
                frames["tracking_root_pos"].append(tracking_root_pos)
                frames["tracking_root_quat"].append(tracking_root_quat)
                frames["tracking_dof_isaac"].append(tracking_dof_isaac.astype(np.float32))
                frames["tracking_dof_mujoco"].append(tracking_dof_mujoco.astype(np.float32))
                frames["root_error"].append(float(np.linalg.norm(ref_root_pos - tracking_root_pos)))
                frames["joint_mae"].append(float(np.mean(np.abs(ref_dof_mujoco - tracking_dof_mujoco))))

                tokenizer = obs_dict.get("tokenizer")
                if tokenizer is not None:
                    frames["tokenizer_encoder_index"].append(
                        tokenizer[0, :3].detach().cpu().numpy().astype(np.float32)
                    )
                    frames["tokenizer_flat"].append(
                        tokenizer[0].detach().cpu().numpy().astype(np.float32)
                    )
                frames["actor_obs"].append(
                    obs_dict["actor_obs"][0].detach().cpu().numpy().astype(np.float32)
                )

                if args.action_source == "policy":
                    if policy is None:
                        raise RuntimeError("Policy action source selected but policy was not loaded")
                    policy.init_rollout()
                    policy_state = policy.rollout(obs_dict=obs_dict)
                    action_mean = policy_state["action_mean"].detach()
                    action_sample = policy_state["actions"].detach()
                    action = action_sample if args.action_mode == "sample" else action_mean
                    if hasattr(env, "_expand_h2lora_policy_action"):
                        sim_action = env._expand_h2lora_policy_action(action)
                    obs_dict, rew, dones, extras = env.step({"actions": action})
                    frames["onnx_encoder_input"].append(np.zeros(1751, dtype=np.float32))
                    frames["onnx_token_state"].append(np.zeros(64, dtype=np.float32))
                    frames["onnx_policy_input"].append(np.zeros(994, dtype=np.float32))
                elif args.action_source == "onnx-policy":
                    if onnx_encoder is None or onnx_decoder is None:
                        raise RuntimeError("ONNX action source selected but sessions were not loaded")
                    (
                        action_np,
                        encoder_input_np,
                        token_state_np,
                        policy_input_np,
                        _actor_obs_np,
                    ) = _run_onnx_policy(
                        env, obs_dict, onnx_encoder, onnx_decoder, args.use_encoder
                    )
                    action_raw = torch.as_tensor(action_np, dtype=torch.float32, device=device)
                    clipped_action_np = _clip_onnx_policy_action(action_np, args.onnx_action_clip)
                    action = torch.as_tensor(clipped_action_np, dtype=torch.float32, device=device)
                    action_mean = action
                    action_sample = action_raw
                    if hasattr(env, "_expand_h2lora_policy_action"):
                        sim_action = env._expand_h2lora_policy_action(action)
                    obs_dict, rew, dones, extras = env.step({"actions": action})
                    frames["onnx_encoder_input"].append(encoder_input_np[0].astype(np.float32))
                    frames["onnx_token_state"].append(token_state_np[0].astype(np.float32))
                    frames["onnx_policy_input"].append(policy_input_np[0].astype(np.float32))
                elif args.action_source == "external-g1-actions":
                    if external_actions is None:
                        raise RuntimeError("External action source selected but actions were not loaded")
                    ext_idx = min(step_idx, len(external_actions) - 1)
                    action = torch.as_tensor(
                        external_actions[ext_idx : ext_idx + 1], dtype=torch.float32, device=device
                    )
                    action_mean = action
                    action_sample = action
                    if hasattr(env, "_expand_h2lora_policy_action"):
                        sim_action = env._expand_h2lora_policy_action(action)
                    obs_dict, rew, dones, extras = env.step({"actions": action})
                    frames["onnx_encoder_input"].append(np.zeros(1751, dtype=np.float32))
                    frames["onnx_token_state"].append(np.zeros(64, dtype=np.float32))
                    frames["onnx_policy_input"].append(np.zeros(994, dtype=np.float32))
                elif args.action_source == "pd-ref":
                    action = zero_policy_action
                    action_mean = action
                    action_sample = action
                    sim_action = _h2_ref_pd_sim_action(
                        env, command, robot, body_indices, head_indices, head_defaults
                    )
                    obs_dict, rew, dones, extras = _step_sim_action_direct(env, sim_action)
                    frames["onnx_encoder_input"].append(np.zeros(1751, dtype=np.float32))
                    frames["onnx_token_state"].append(np.zeros(64, dtype=np.float32))
                    frames["onnx_policy_input"].append(np.zeros(994, dtype=np.float32))
                else:
                    raise ValueError(f"Unknown action_source: {args.action_source}")

                frames["policy_action_mean"].append(
                    action_mean[0].detach().cpu().numpy().astype(np.float32)
                )
                frames["policy_action_sample"].append(
                    action_sample[0].detach().cpu().numpy().astype(np.float32)
                )
                frames["policy_action_used"].append(
                    action[0].detach().cpu().numpy().astype(np.float32)
                )
                if args.action_source == "onnx-policy":
                    frames["policy_action_raw"].append(
                        action_sample[0].detach().cpu().numpy().astype(np.float32)
                    )
                else:
                    frames["policy_action_raw"].append(
                        action[0].detach().cpu().numpy().astype(np.float32)
                    )
                frames["sim_action_used"].append(
                    sim_action[0].detach().cpu().numpy().astype(np.float32)
                )
                frames["reward"].append(float(rew[0].detach().cpu().item()))
                frames["done"].append(bool(dones[0].detach().cpu().item()))

        traj_path = output_dir / "trajectory_ref_tracking.npz"
        np.savez_compressed(
            traj_path,
            **{key: np.asarray(value) for key, value in frames.items()},
            fps=np.array(float(args.fps), dtype=np.float32),
            checkpoint=np.array(str(Path(args.checkpoint).expanduser().resolve())),
            motion_file=np.array(str(Path(args.motion_file).expanduser().resolve())),
            smpl_motion_file=np.array(str(Path(args.smpl_motion_file).expanduser().resolve())),
            motion_key=np.array(args.motion_key or _first_motion_key(env) or ""),
            robot_joint_names=np.asarray(robot.joint_names),
            body_joint_indices=np.asarray(body_indices, dtype=np.int64),
            body_joint_names=np.asarray([robot.joint_names[int(i)] for i in body_indices]),
            head_joint_indices=np.asarray(head_indices, dtype=np.int64),
            isaac_to_mujoco_dof=np.asarray(isaac_to_mujoco, dtype=np.int64),
            config_yaml=np.array(OmegaConf.to_yaml(config, resolve=False)),
            disable_terminations=np.array(bool(args.disable_terminations)),
            disable_events=np.array(bool(args.disable_events)),
            action_source=np.array(args.action_source),
            onnx_encoder=np.array(str(Path(args.onnx_encoder).expanduser().resolve())),
            onnx_decoder=np.array(str(Path(args.onnx_decoder).expanduser().resolve())),
            external_action_trajectory=np.array(args.external_action_trajectory or ""),
            label=np.array(args.label or ""),
        )
        (output_dir / "collect_meta.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
                    "motion_file": str(Path(args.motion_file).expanduser().resolve()),
                    "smpl_motion_file": str(Path(args.smpl_motion_file).expanduser().resolve()),
                    "motion_key": args.motion_key or _first_motion_key(env),
                    "num_steps": int(args.num_steps),
                    "fps": float(args.fps),
                    "disable_terminations": bool(args.disable_terminations),
                    "disable_events": bool(args.disable_events),
                    "label": args.label or "",
                    "action_mode": args.action_mode,
                    "action_source": args.action_source,
                    "onnx_encoder": str(Path(args.onnx_encoder).expanduser().resolve()),
                    "onnx_decoder": str(Path(args.onnx_decoder).expanduser().resolve()),
                    "external_action_trajectory": args.external_action_trajectory or "",
                    "mean_root_error": float(np.mean(frames["root_error"])),
                    "mean_joint_mae": float(np.mean(frames["joint_mae"])),
                    "mean_policy_action_mean_abs": float(
                        np.mean(np.abs(frames["policy_action_mean"]))
                    ),
                    "mean_policy_action_used_abs": float(
                        np.mean(np.abs(frames["policy_action_used"]))
                    ),
                    "mean_sim_action_used_abs": float(np.mean(np.abs(frames["sim_action_used"]))),
                    "done_count": int(np.sum(frames["done"])),
                },
                indent=2,
            )
        )
        print(f"[h2lora-viz] Wrote trajectory {traj_path}", flush=True)
        if args.collect_only:
            print(f"COLLECTED_TRAJECTORY={traj_path}", flush=True)
            print(f"OUTPUT_DIR={output_dir}", flush=True)
            os._exit(0)
        return traj_path
    except BaseException as exc:
        import traceback

        print(f"[h2lora-viz] collect_rollout failed: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        if isinstance(exc, SystemExit) and exc.code == 0:
            raise SystemExit(1) from exc
        raise
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()
        simulation_app.close()


def _prefix_names(elem: ET.Element, prefix: str) -> None:
    if "name" in elem.attrib:
        elem.set("name", prefix + elem.attrib["name"])
    for child in list(elem):
        _prefix_names(child, prefix)


def _set_robot_rgba(elem: ET.Element, rgba: str) -> None:
    for child in elem.iter():
        if child.tag == "geom":
            child.set("rgba", rgba)
            child.set("contype", "0")
            child.set("conaffinity", "0")


def _build_dual_h2_xml() -> str:
    source = ET.parse(H2_MJCF).getroot()
    scene = ET.Element("mujoco", {"model": "h2lora_ref_tracking"})
    ET.SubElement(scene, "compiler", {"angle": "radian"})
    ET.SubElement(scene, "option", {"timestep": "0.02", "gravity": "0 0 -9.81"})

    asset = ET.SubElement(scene, "asset")
    ET.SubElement(asset, "texture", {"type": "skybox", "builtin": "gradient", "rgb1": "0.78 0.86 0.94", "rgb2": "1 1 1", "width": "256", "height": "256"})
    ET.SubElement(asset, "texture", {"name": "grid", "type": "2d", "builtin": "checker", "rgb1": "0.92 0.92 0.90", "rgb2": "0.78 0.80 0.82", "width": "512", "height": "512"})
    ET.SubElement(asset, "material", {"name": "grid", "texture": "grid", "texrepeat": "4 4", "reflectance": "0.15"})
    for mesh in source.find("asset").findall("mesh"):
        mesh_copy = copy.deepcopy(mesh)
        mesh_file = H2_MESH_DIR / mesh_copy.attrib["file"]
        mesh_copy.set("file", str(mesh_file))
        asset.append(mesh_copy)

    worldbody = ET.SubElement(scene, "worldbody")
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "8 8 0.02",
            "material": "grid",
            "rgba": "0.82 0.84 0.86 1",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {"name": "key", "pos": "0 -3 5", "dir": "0 0 -1", "diffuse": "0.8 0.8 0.8"},
    )
    ET.SubElement(
        worldbody,
        "light",
        {"name": "fill", "pos": "-3 3 4", "dir": "0 0 -1", "diffuse": "0.35 0.35 0.35"},
    )

    robot_body = source.find("worldbody").find("body")
    if robot_body is None:
        raise RuntimeError(f"No robot body found in {H2_MJCF}")

    ref_body = copy.deepcopy(robot_body)
    _prefix_names(ref_body, "ref_")
    _set_robot_rgba(ref_body, "0.10 0.35 1.00 0.42")
    worldbody.append(ref_body)

    tracking_body = copy.deepcopy(robot_body)
    _prefix_names(tracking_body, "tracking_")
    _set_robot_rgba(tracking_body, "1.00 0.34 0.04 0.92")
    worldbody.append(tracking_body)

    return ET.tostring(scene, encoding="unicode")


def _load_traj(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _trajectory_fps(traj: dict[str, np.ndarray], fallback: float) -> float:
    if "fps" not in traj:
        return float(fallback)
    try:
        value = float(np.asarray(traj["fps"]).item())
    except Exception:
        return float(fallback)
    return value if value > 0 else float(fallback)


def render_video(args: argparse.Namespace, output_dir: Path, traj_path: Path | None = None) -> Path:
    """Render the collected trajectory with two colored MuJoCo H2 robots."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio.v2 as imageio
    import mujoco

    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Rendering MP4 requires imageio-ffmpeg. Install it with "
            "`uv pip install --python .venv-h2lora/bin/python imageio-ffmpeg` "
            "or rerun scripts/setup_h2lora_uv.sh."
        ) from exc

    try:
        import cv2
    except ImportError:  # pragma: no cover - text overlay is optional.
        cv2 = None

    traj_path = Path(traj_path or args.trajectory).expanduser().resolve()
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file does not exist: {traj_path}")
    traj = _load_traj(traj_path)
    label = args.label or ""
    if not label and "label" in traj:
        label = str(np.asarray(traj["label"]).item())

    model = mujoco.MjModel.from_xml_string(_build_dual_h2_xml())
    data = mujoco.MjData(model)
    if model.nq != 76:
        raise RuntimeError(f"Expected two 38-qpos H2 robots, got model.nq={model.nq}")

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
    video_path = output_dir / "ref_tracking.mp4"
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
            qpos = data.qpos
            qpos[:] = 0.0
            qpos[0:3] = ref_pos
            qpos[3:7] = ref_root_quat[idx]
            qpos[7:38] = ref_dof[idx]
            qpos[38:41] = tracking_pos
            qpos[41:45] = tracking_root_quat[idx]
            qpos[45:76] = tracking_dof[idx]

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
                    f"H2LoRA ref/tracking {label}".strip(),
                    "blue: reference   orange: tracking",
                    f"frame {idx + 1}/{nframes}  root_err {root_err:.3f} m  joint_mae {joint_mae:.3f} rad",
                ]
                y = 32
                for line in text:
                    cv2.putText(
                        frame,
                        line,
                        (24, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (24, 24, 24),
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        line,
                        (24, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (245, 245, 245),
                        1,
                        cv2.LINE_AA,
                    )
                    y += 30

            if idx == 0:
                imageio.imwrite(preview_path, frame)
            writer.append_data(frame)
    finally:
        writer.close()
        renderer.close()

    metrics = {
        "trajectory": str(traj_path),
        "video": str(video_path),
        "preview": str(preview_path),
        "label": label,
        "frames": int(nframes),
        "fps": fps,
        "trajectory_fps": trajectory_fps,
        "duration_s": float(nframes / fps) if fps > 0 else None,
        "trajectory_duration_s": float(nframes / trajectory_fps) if trajectory_fps > 0 else None,
        "mean_root_error_m": float(np.mean(np.linalg.norm(ref_root_pos[:nframes] - tracking_root_pos[:nframes], axis=-1))),
        "mean_joint_mae_rad": float(np.mean(np.abs(ref_dof[:nframes] - tracking_dof[:nframes]))),
        "max_root_error_m": float(np.max(np.linalg.norm(ref_root_pos[:nframes] - tracking_root_pos[:nframes], axis=-1))),
        "max_joint_mae_rad": float(np.max(np.mean(np.abs(ref_dof[:nframes] - tracking_dof[:nframes]), axis=-1))),
    }
    (output_dir / "render_meta.json").write_text(json.dumps(metrics, indent=2))
    return video_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="H2LoRA checkpoint to visualize.")
    parser.add_argument("--motion-file", default=str(DEFAULT_MOTION_FILE), help="Robot MotionLib directory.")
    parser.add_argument("--smpl-motion-file", default=str(DEFAULT_SMPL_MOTION_FILE), help="SMPL MotionLib directory.")
    parser.add_argument("--motion-key", default="Idle_Left_001__A017", help="Optional motion key filter.")
    parser.add_argument("--num-steps", type=int, default=160, help="Rollout frames to collect.")
    parser.add_argument("--fps", type=float, default=50.0, help="Output video FPS. Use 0 to reuse trajectory fps.")
    parser.add_argument("--width", type=int, default=1280, help="Video width.")
    parser.add_argument("--height", type=int, default=720, help="Video height.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root directory for videos.")
    parser.add_argument("--output-dir", default=None, help="Exact output directory. Overrides naming rule.")
    parser.add_argument("--experiment-name", default=None, help="Folder prefix; sanitized before use.")
    parser.add_argument("--label", default=None, help="Optional stage label rendered into the video overlay.")
    parser.add_argument("--timestamp", default=None, help="Folder timestamp. Defaults to current local time.")
    parser.add_argument("--trajectory", default=None, help="Existing trajectory .npz for --render-only.")
    parser.add_argument("--collect-only", action="store_true", help="Only collect IsaacLab rollout.")
    parser.add_argument("--render-only", action="store_true", help="Only render an existing trajectory.")
    parser.add_argument("--max-render-frames", type=int, default=0, help="Optional frame cap for rendering.")
    parser.add_argument(
        "--action-mode",
        choices=("mean", "sample"),
        default="mean",
        help="Use deterministic policy mean or sampled training-style action during rollout.",
    )
    parser.add_argument(
        "--action-source",
        choices=("policy", "onnx-policy", "pd-ref", "external-g1-actions"),
        default="policy",
        help=(
            "Diagnostic action source: checkpoint policy, exported ONNX policy, "
            "direct H2 reference joint targets, or 29D G1 actions from another rollout."
        ),
    )
    parser.add_argument(
        "--onnx-encoder",
        default=str(DEFAULT_ONNX_ENCODER),
        help="Exported H2LoRA encoder ONNX used by --action-source onnx-policy.",
    )
    parser.add_argument(
        "--onnx-decoder",
        default=str(DEFAULT_ONNX_DECODER),
        help="Exported H2LoRA decoder ONNX used by --action-source onnx-policy.",
    )
    parser.add_argument(
        "--onnx-action-clip",
        type=float,
        default=6.0,
        help="Clip ONNX decoder raw actions before H2LoRA action adaptation; deploy defaults to 6.",
    )
    parser.add_argument(
        "--external-action-trajectory",
        default=None,
        help="Trajectory .npz containing policy_action_used for --action-source external-g1-actions.",
    )
    parser.add_argument(
        "--use-encoder",
        choices=("g1", "teleop", "smpl"),
        default=None,
        help="Force one universal-token encoder during visualization rollout.",
    )
    parser.add_argument("--camera-distance", type=float, default=3.3)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument(
        "--keep-terminations",
        dest="disable_terminations",
        action="store_false",
        help="Keep training early terminations during visualization rollout.",
    )
    parser.add_argument(
        "--keep-events",
        dest="disable_events",
        action="store_false",
        help="Keep domain/randomization events during visualization rollout.",
    )
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

    traj_path = None
    if not args.render_only:
        traj_path = collect_rollout(args, output_dir)
        print(f"COLLECTED_TRAJECTORY={traj_path}")
    if not args.collect_only:
        traj_path = Path(args.trajectory).expanduser().resolve() if args.render_only else traj_path
        video_path = render_video(args, output_dir, traj_path)
        print(f"RENDERED_VIDEO={video_path}")
        print(f"OUTPUT_DIR={output_dir}")
    else:
        print(f"OUTPUT_DIR={output_dir}")


if __name__ == "__main__":
    main()
