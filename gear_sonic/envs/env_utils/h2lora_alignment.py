"""Kinematic alignment utilities for SONIC G1-to-H2LoRA transfer."""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

from gear_sonic.envs.env_utils import joint_utils

ASSET_ROOT = Path("gear_sonic/data/assets/robot_description")
G1_MJCF = ASSET_ROOT / "mjcf/g1_29dof_rev_1_0.xml"
H2_MJCF = ASSET_ROOT / "mjcf/h2.xml"
H2_URDF = ASSET_ROOT / "urdf/h2/h2.urdf"


def _quat_to_matrix(quat: list[float]) -> torch.Tensor:
    w, x, y, z = quat
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        return torch.eye(3, dtype=torch.float64)
    w, x, y, z = (w / norm, x / norm, y / norm, z / norm)
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float64,
    )


def _rpy_to_matrix(rpy: list[float]) -> torch.Tensor:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = torch.tensor([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=torch.float64)
    ry = torch.tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=torch.float64)
    rz = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=torch.float64)
    return rz @ ry @ rx


def _parse_vec(text: str | None, n: int, default: float = 0.0) -> list[float]:
    if text is None:
        return [default] * n
    return [float(x) for x in text.split()]


def parse_mjcf_joint_axes(path: Path | str) -> dict[str, torch.Tensor]:
    """Parse world-frame joint axes from an MJCF at the zero pose."""
    path = Path(path)
    root = ET.parse(path).getroot()
    axes: dict[str, torch.Tensor] = {}

    def walk(body: ET.Element, parent_rot: torch.Tensor) -> None:
        body_rot = parent_rot @ _quat_to_matrix(_parse_vec(body.get("quat"), 4, default=0.0))
        if body.get("quat") is None:
            body_rot = parent_rot
        for joint in body.findall("joint"):
            name = joint.get("name")
            axis = torch.tensor(_parse_vec(joint.get("axis"), 3), dtype=torch.float64)
            axes[name] = body_rot @ axis
        for child in body.findall("body"):
            walk(child, body_rot)

    for body in root.find("worldbody").findall("body"):
        walk(body, torch.eye(3, dtype=torch.float64))
    return axes


def parse_urdf_joint_axes(path: Path | str, root_link: str = "pelvis") -> dict[str, torch.Tensor]:
    """Parse world-frame joint axes from a URDF at the zero pose."""
    path = Path(path)
    root = ET.parse(path).getroot()
    children: dict[str, list[ET.Element]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")
        children.setdefault(parent, []).append(joint)

    axes: dict[str, torch.Tensor] = {}

    def walk(link: str, parent_rot: torch.Tensor) -> None:
        for joint in children.get(link, []):
            origin = joint.find("origin")
            joint_rot = parent_rot @ _rpy_to_matrix(
                _parse_vec(origin.get("rpy") if origin is not None else None, 3)
            )
            if joint.get("type") != "fixed":
                axis_el = joint.find("axis")
                axis = torch.tensor(
                    _parse_vec(axis_el.get("xyz") if axis_el is not None else None, 3),
                    dtype=torch.float64,
                )
                axes[joint.get("name")] = joint_rot @ axis
            walk(joint.find("child").get("link"), joint_rot)

    walk(root_link, torch.eye(3, dtype=torch.float64))
    return axes


def _basis_for_pair(axes: dict[str, torch.Tensor], pair: tuple[str, str]) -> torch.Tensor:
    return torch.stack([axes[pair[0]], axes[pair[1]]], dim=1)


@lru_cache(maxsize=1)
def hip_alignment_matrices() -> dict[str, dict[str, torch.Tensor]]:
    """Return per-side 2x2 source<->target hip alignment matrices.

    The matrices implement the local linear projection described by Any2Any:
    source = pinv(B_source) @ B_target @ target and the inverse for actions.
    """
    g1_axes = parse_mjcf_joint_axes(G1_MJCF)
    h2_axes = parse_mjcf_joint_axes(H2_MJCF)
    matrices: dict[str, dict[str, torch.Tensor]] = {}
    for side, pair in joint_utils.HIP_JOINT_PAIRS.items():
        source_basis = _basis_for_pair(g1_axes, pair)
        target_basis = _basis_for_pair(h2_axes, pair)
        target_to_source = torch.linalg.pinv(source_basis) @ target_basis
        source_to_target = torch.linalg.pinv(target_basis) @ source_basis
        matrices[side] = {
            "target_to_source": target_to_source.float(),
            "source_to_target": source_to_target.float(),
        }
    return matrices


def _apply_pair_matrix(
    values: torch.Tensor,
    joint_names: list[str],
    side: str,
    matrix: torch.Tensor,
) -> torch.Tensor:
    pair = joint_utils.HIP_JOINT_PAIRS[side]
    idx = [joint_names.index(pair[0]), joint_names.index(pair[1])]
    out = values.clone()
    source = values[..., idx]
    out[..., idx] = torch.matmul(source, matrix.to(values.device, values.dtype).T)
    return out


def h2_to_g1_semantic(values: torch.Tensor) -> torch.Tensor:
    """Map H2 headless 29D physical values to G1 semantic 29D values."""
    out = values
    matrices = hip_alignment_matrices()
    for side in ("left", "right"):
        out = _apply_pair_matrix(
            out,
            joint_utils.H2LORA_BODY_JOINTS_IN_G1_ORDER,
            side,
            matrices[side]["target_to_source"],
        )
    return out


def g1_to_h2_physical(values: torch.Tensor) -> torch.Tensor:
    """Map G1 semantic 29D values to H2 headless physical 29D values."""
    out = values
    matrices = hip_alignment_matrices()
    for side in ("left", "right"):
        out = _apply_pair_matrix(
            out,
            joint_utils.H2LORA_BODY_JOINTS_IN_G1_ORDER,
            side,
            matrices[side]["source_to_target"],
        )
    return out


def assert_axis_audit() -> None:
    """Validate the known G1/H2 kinematic gaps used by H2LoRA."""
    g1 = parse_mjcf_joint_axes(G1_MJCF)
    h2_mjcf = parse_mjcf_joint_axes(H2_MJCF)
    h2_urdf = parse_urdf_joint_axes(H2_URDF)

    assert torch.allclose(g1["left_hip_pitch_joint"], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))
    assert torch.allclose(
        h2_mjcf["left_hip_pitch_joint"],
        h2_urdf["left_hip_pitch_joint"],
        atol=1e-5,
    )
    assert abs(float(h2_mjcf["left_hip_pitch_joint"][2])) > 0.49
    assert torch.allclose(h2_mjcf["head_pitch_joint"], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))
    assert torch.allclose(h2_mjcf["head_yaw_joint"], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
