"""Validated policy inputs. G1 producers all use ten samples at 0.1 s spacing."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

SCHEMA = "sonic.h2.command.v1"
G1_OFFSETS = np.arange(10, dtype=np.float64) * 0.1
SMPL_OFFSETS = np.arange(10, dtype=np.float64) * 0.02


def array(value: Any, shape: tuple[int, ...], name: str) -> list:
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {a.shape}")
    return a.tolist()


def quaternions(value: Any, count: int, name: str = "root_quat") -> list:
    a = np.asarray(array(value, (count, 4), name))
    norm = np.linalg.norm(a, axis=-1, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm < 1e-6):
        raise ValueError(f"{name} contains a zero quaternion")
    return (a / norm).tolist()


def g1_reference(joint_pos, joint_vel, root_quat) -> dict:
    """Native G1 semantic joint order; root quaternion is world-frame wxyz."""
    return {"joint_pos": array(joint_pos, (10, 29), "joint_pos"),
            "joint_vel": array(joint_vel, (10, 29), "joint_vel"),
            "root_quat": quaternions(root_quat, 10)}


def smpl_reference(joints, root_quat, wrist_joints) -> dict:
    """Ten SMPL samples at 50 Hz; root-centered world-axis joints, wxyz root."""
    return {"joints": array(joints, (10, 24, 3), "joints"),
            "root_quat": quaternions(root_quat, 10),
            "wrist_joints": array(wrist_joints, (10, 6), "wrist_joints")}


def vr_reference(positions, orientations, lower_body: dict) -> dict:
    """Left wrist, right wrist, head in the calibrated robot-local frame.

    The producer supplies the lower-body G1 reference, from either a standing
    template or the independent planner service. The runtime never runs a planner.
    """
    return {"positions": array(positions, (3, 3), "positions"),
            "orientations": quaternions(orientations, 3, "orientations"),
            "lower_body": g1_reference(**lower_body)}


def token_command(tokens) -> dict:
    """A VLA or other external policy supplies exactly 64 finite latent values."""
    return {"tokens": array(tokens, (64,), "tokens")}


def planner_command(mode=0, movement=(0, 0, 0), facing=(1, 0, 0), speed=-1, height=-1):
    if isinstance(mode, bool) or not isinstance(mode, int) or not 0 <= mode <= 26:
        raise ValueError("mode must be an integer in [0, 26]")
    return {"mode": mode, "movement": array(movement, (3,), "movement"),
            "facing": array(facing, (3,), "facing"),
            "speed": array(speed, (), "speed"), "height": array(height, (), "height")}


def head_command(pitch, yaw) -> dict:
    values = array([pitch, yaw], (2,), "head")
    return {"pitch": values[0], "yaw": values[1]}


def hands_command(**hands) -> dict:
    if not hands or not set(hands) <= {"left", "right"}:
        raise ValueError("hands requires left and/or right")
    return {k: array(v, (7,), k) for k, v in hands.items()}


def envelope(kind: str, data: dict, *, source: str, session: str, seq: int,
             ttl_ms: int = 250, takeover: bool = False) -> dict:
    if kind not in {"g1", "smpl", "vr", "token", "stand", "stop", "head", "hands", "planner"}:
        raise ValueError(f"Unknown command kind: {kind}")
    if not source or not session or len(source) > 64 or len(session) > 64:
        raise ValueError("source and session must contain 1..64 characters")
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        raise ValueError("seq must be a nonnegative integer")
    if isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or not 50 <= ttl_ms <= 1000:
        raise ValueError("ttl_ms must be between 50 and 1000")
    validators = {"g1": g1_reference, "smpl": smpl_reference,
                  "vr": vr_reference, "token": token_command,
                  "head": head_command, "hands": hands_command, "planner": planner_command}
    if kind in validators:
        data = validators[kind](**data)
    elif data:
        raise ValueError(f"{kind} does not carry policy data")
    result = {"schema": SCHEMA, "kind": kind, "source": source, "session": session,
              "seq": seq, "sent_at_ms": time.time_ns() // 1_000_000,
              "ttl_ms": ttl_ms, "takeover": bool(takeover), "data": data}
    return result
