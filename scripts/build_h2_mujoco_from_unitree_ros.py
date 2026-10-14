#!/usr/bin/env python3
"""Build a Gear-Sonic-compatible MJCF from Unitree ROS's H2 URDF."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_DIR = (
    REPO_ROOT / "gear_sonic/data/assets/robot_description/unitree_ros/h2_description"
)

JOINT_NAMES = [
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
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def format_values(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.15g}" for value in values)


def parse_vector(value: str | None) -> np.ndarray:
    return np.fromstring(value or "0 0 0", sep=" ", dtype=float)


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def origin_transform(element: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if element is None:
        return transform
    transform[:3, :3] = rpy_matrix(parse_vector(element.get("rpy")))
    transform[:3, 3] = parse_vector(element.get("xyz"))
    return transform


def inertia_matrix(inertia: ET.Element) -> np.ndarray:
    return np.array(
        [
            [float(inertia.get("ixx")), float(inertia.get("ixy")), float(inertia.get("ixz"))],
            [float(inertia.get("ixy")), float(inertia.get("iyy")), float(inertia.get("iyz"))],
            [float(inertia.get("ixz")), float(inertia.get("iyz")), float(inertia.get("izz"))],
        ],
        dtype=float,
    )


def collect_compound_inertials(
    urdf_root: ET.Element, body_names: set[str]
) -> dict[str, tuple[float, np.ndarray, np.ndarray]]:
    """Fold fixed-link inertials into the active MuJoCo body frame."""
    parent_joints: dict[str, ET.Element] = {}
    for joint in urdf_root.findall("joint"):
        child = joint.find("child")
        if child is not None:
            parent_joints[child.get("link")] = joint

    contributions: dict[str, list[tuple[float, np.ndarray, np.ndarray]]] = {
        name: [] for name in body_names
    }
    for link in urdf_root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue

        mass_element = inertial.find("mass")
        tensor_element = inertial.find("inertia")
        if mass_element is None or tensor_element is None:
            raise ValueError(f"Incomplete inertial on URDF link {link.get('name')}")

        source_name = link.get("name")
        destination_name = source_name
        destination_from_source = np.eye(4)
        while destination_name not in body_names:
            joint = parent_joints.get(destination_name)
            if joint is None:
                raise ValueError(
                    f"Cannot map URDF inertial link {source_name} to a MuJoCo body"
                )
            if joint.get("type") != "fixed":
                raise ValueError(
                    f"URDF inertial link {source_name} reaches an active body through "
                    f"non-fixed joint {joint.get('name')}"
                )
            destination_from_source = (
                origin_transform(joint.find("origin")) @ destination_from_source
            )
            destination_name = joint.find("parent").get("link")

        inertial_origin = origin_transform(inertial.find("origin"))
        mass = float(mass_element.get("value"))
        center_in_source = inertial_origin[:3, 3]
        rotation_in_source = inertial_origin[:3, :3]
        tensor_in_source = (
            rotation_in_source
            @ inertia_matrix(tensor_element)
            @ rotation_in_source.T
        )

        rotation = destination_from_source[:3, :3]
        center = (
            destination_from_source[:3, 3] + rotation @ center_in_source
        )
        tensor = rotation @ tensor_in_source @ rotation.T
        contributions[destination_name].append((mass, center, tensor))

    compounds: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    for body_name, parts in contributions.items():
        if not parts:
            raise ValueError(f"No URDF inertial mapped to MuJoCo body {body_name}")
        total_mass = sum(part[0] for part in parts)
        center = sum(mass * part_center for mass, part_center, _ in parts) / total_mass
        tensor = np.zeros((3, 3), dtype=float)
        for mass, part_center, part_tensor in parts:
            offset = part_center - center
            tensor += part_tensor + mass * (
                np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
            )
        if np.min(np.linalg.eigvalsh(tensor)) <= 0.0:
            raise ValueError(f"Non-positive compound inertia for {body_name}")
        compounds[body_name] = (total_mass, center, tensor)
    return compounds


def replace_compound_inertials(
    mjcf_root: ET.Element,
    compounds: dict[str, tuple[float, np.ndarray, np.ndarray]],
) -> None:
    body_elements = {
        body.get("name"): body for body in mjcf_root.findall(".//body[@name]")
    }
    if set(body_elements) != set(compounds):
        raise ValueError(
            "URDF/MJCF body mismatch while replacing inertials: "
            f"mjcf_only={sorted(set(body_elements) - set(compounds))}, "
            f"urdf_only={sorted(set(compounds) - set(body_elements))}"
        )

    for body_name, (mass, center, tensor) in compounds.items():
        body = body_elements[body_name]
        full_inertia = np.array(
            [
                tensor[0, 0],
                tensor[1, 1],
                tensor[2, 2],
                tensor[0, 1],
                tensor[0, 2],
                tensor[1, 2],
            ]
        )
        replacement = ET.Element(
            "inertial",
            {
                "pos": format_values(center),
                "mass": f"{mass:.15g}",
                "fullinertia": format_values(full_inertia),
            },
        )
        current = body.find("inertial")
        if current is None:
            body.insert(0, replacement)
        else:
            index = list(body).index(current)
            body.remove(current)
            body.insert(index, replacement)


def body_inertia_matrix(model: mujoco.MjModel, body_id: int) -> np.ndarray:
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.body_iquat[body_id])
    rotation = rotation.reshape(3, 3)
    return rotation @ np.diag(model.body_inertia[body_id]) @ rotation.T


def validate_compound_inertials(
    model: mujoco.MjModel,
    compounds: dict[str, tuple[float, np.ndarray, np.ndarray]],
) -> None:
    max_mass_error = 0.0
    max_center_error = 0.0
    max_tensor_error = 0.0
    for body_name, (mass, center, tensor) in compounds.items():
        body_id = model.body(body_name).id
        max_mass_error = max(max_mass_error, abs(float(model.body_mass[body_id]) - mass))
        max_center_error = max(
            max_center_error,
            float(np.linalg.norm(model.body_ipos[body_id] - center)),
        )
        max_tensor_error = max(
            max_tensor_error,
            float(np.max(np.abs(body_inertia_matrix(model, body_id) - tensor))),
        )
    if max_mass_error > 1e-9 or max_center_error > 1e-9 or max_tensor_error > 1e-8:
        raise ValueError(
            "Generated MJCF inertial validation failed: "
            f"mass={max_mass_error:.3g}, center={max_center_error:.3g}, "
            f"tensor={max_tensor_error:.3g}"
        )


def add_floating_base(urdf_root: ET.Element) -> None:
    if urdf_root.find("joint[@name='floating_base_joint']") is not None:
        return

    world = ET.Element("link", {"name": "world"})
    floating_joint = ET.Element(
        "joint", {"name": "floating_base_joint", "type": "floating"}
    )
    ET.SubElement(floating_joint, "parent", {"link": "world"})
    ET.SubElement(floating_joint, "child", {"link": "pelvis"})
    urdf_root.insert(0, floating_joint)
    urdf_root.insert(0, world)


def add_site(body: ET.Element, name: str) -> None:
    if body.find(f"site[@name='{name}']") is None:
        body.insert(0, ET.Element("site", {"name": name, "size": "0.01"}))


def build_mjcf(source_urdf: Path, output: Path) -> None:
    urdf_tree = ET.parse(source_urdf)
    urdf_root = urdf_tree.getroot()
    compiler = urdf_root.find("mujoco/compiler")
    if compiler is None:
        raise ValueError(f"Missing <mujoco><compiler> in {source_urdf}")

    # H2.urdf uses both meshdir="meshes" and filenames prefixed by "meshes/".
    # MuJoCo combines those paths, so compile from the asset root instead.
    compiler.set("meshdir", ".")
    add_floating_base(urdf_root)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", dir=source_urdf.parent, delete=False
    ) as temporary_urdf:
        urdf_tree.write(temporary_urdf, encoding="unicode")
        temporary_urdf_path = Path(temporary_urdf.name)

    temporary_mjcf_path = output.with_suffix(".compiled.xml")
    try:
        model = mujoco.MjModel.from_xml_path(str(temporary_urdf_path))
        compiled_joint_names = [model.joint(i).name for i in range(1, model.njnt)]
        if compiled_joint_names != JOINT_NAMES:
            raise ValueError(
                "Unexpected H2 joint order after URDF import:\n"
                f"expected={JOINT_NAMES}\nactual={compiled_joint_names}"
            )
        body_names = {model.body(i).name for i in range(1, model.nbody)}
        compound_inertials = collect_compound_inertials(urdf_root, body_names)
        mujoco.mj_saveLastXML(str(temporary_mjcf_path), model)
    finally:
        temporary_urdf_path.unlink(missing_ok=True)

    tree = ET.parse(temporary_mjcf_path)
    root = tree.getroot()
    root.set("model", "Unitree ROS H2")
    replace_compound_inertials(root, compound_inertials)

    mjcf_compiler = root.find("compiler")
    if mjcf_compiler is None:
        mjcf_compiler = ET.Element("compiler")
        root.insert(0, mjcf_compiler)
    mjcf_compiler.set("angle", "radian")
    mjcf_compiler.set("meshdir", ".")

    default = root.find("default")
    if default is None:
        default = ET.Element("default")
        root.insert(1, default)
    ET.SubElement(
        default,
        "joint",
        {"damping": "0.05", "armature": "0.01", "frictionloss": "0.2"},
    )

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(2, option)
    option.set("timestep", "0.001")

    pelvis = root.find("worldbody/body[@name='pelvis']")
    torso = root.find(".//body[@name='torso_link']")
    if pelvis is None or torso is None:
        raise ValueError("Compiled H2 model is missing pelvis or torso_link")
    pelvis.set("pos", "0 0 1.2")
    add_site(pelvis, "imu")
    add_site(torso, "secondary_imu")

    joint_elements = {
        joint.get("name"): joint for joint in root.findall(".//joint[@name]")
    }
    actuator = ET.SubElement(root, "actuator")
    for joint_name in JOINT_NAMES:
        joint = joint_elements[joint_name]
        force_range = joint.get("actuatorfrcrange")
        if force_range is None:
            raise ValueError(f"Missing effort limit for {joint_name}")
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": joint_name.removesuffix("_joint"),
                "joint": joint_name,
                "ctrlrange": force_range,
            },
        )

    sensor = ET.SubElement(root, "sensor")
    for joint_name in JOINT_NAMES:
        short_name = joint_name.removesuffix("_joint")
        ET.SubElement(
            sensor, "jointpos", {"name": f"{short_name}_pos", "joint": joint_name}
        )
    for joint_name in JOINT_NAMES:
        short_name = joint_name.removesuffix("_joint")
        ET.SubElement(
            sensor, "jointvel", {"name": f"{short_name}_vel", "joint": joint_name}
        )
    for joint_name in JOINT_NAMES:
        short_name = joint_name.removesuffix("_joint")
        ET.SubElement(
            sensor,
            "jointactuatorfrc",
            {"name": f"{short_name}_torque", "joint": joint_name},
        )

    ET.SubElement(
        sensor,
        "framequat",
        {"name": "imu_quat", "objtype": "site", "objname": "imu"},
    )
    ET.SubElement(sensor, "gyro", {"name": "imu_gyro", "site": "imu"})
    ET.SubElement(sensor, "accelerometer", {"name": "imu_acc", "site": "imu"})
    ET.SubElement(
        sensor,
        "framepos",
        {"name": "frame_pos", "objtype": "site", "objname": "imu"},
    )
    ET.SubElement(
        sensor,
        "framelinvel",
        {"name": "frame_vel", "objtype": "site", "objname": "imu"},
    )
    ET.SubElement(
        sensor,
        "framequat",
        {
            "name": "secondary_imu_quat",
            "objtype": "site",
            "objname": "secondary_imu",
        },
    )
    ET.SubElement(
        sensor,
        "gyro",
        {"name": "secondary_imu_gyro", "site": "secondary_imu", "noise": "5e-4", "cutoff": "34.9"},
    )
    ET.SubElement(
        sensor,
        "accelerometer",
        {"name": "secondary_imu_acc", "site": "secondary_imu", "noise": "1e-2", "cutoff": "157"},
    )

    ET.indent(tree, space="  ")
    tree.write(temporary_mjcf_path, encoding="unicode")

    validated_model = mujoco.MjModel.from_xml_path(str(temporary_mjcf_path))
    expected = (38, 37, 31)
    actual = (validated_model.nq, validated_model.nv, validated_model.nu)
    if actual != expected:
        raise ValueError(
            f"Unexpected compiled model dimensions: expected={expected}, actual={actual}"
        )
    validate_compound_inertials(validated_model, compound_inertials)

    temporary_mjcf_path.replace(output)
    print(f"Wrote {output}")
    print(
        f"Validated nq={validated_model.nq}, nv={validated_model.nv}, "
        f"nu={validated_model.nu}, nsensor={validated_model.nsensor}, "
        f"mass={float(np.sum(validated_model.body_mass)):.9f} kg"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_ASSET_DIR / "H2.urdf")
    parser.add_argument("--output", type=Path, default=DEFAULT_ASSET_DIR / "h2_mujoco.xml")
    args = parser.parse_args()
    build_mjcf(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
