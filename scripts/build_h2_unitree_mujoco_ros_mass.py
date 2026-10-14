#!/usr/bin/env python3
"""Align Unitree MuJoCo H2 inertials with the Unitree ROS H2 URDF."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from scripts.build_h2_mujoco_from_unitree_ros import collect_compound_inertials


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = REPO_ROOT / "gear_sonic/data/assets/robot_description"
DEFAULT_BASELINE = ASSET_ROOT / "unitree_mujoco/h2/h2_mujoco.xml"
DEFAULT_ROS_URDF = ASSET_ROOT / "unitree_ros/h2_description/H2.urdf"
DEFAULT_OUTPUT = ASSET_ROOT / "unitree_mujoco/h2_ros_mass_aligned/h2_mujoco.xml"


def format_values(values: np.ndarray | list[float]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def collect_ros_body_masses(
    urdf_root: ET.Element, mujoco_body_names: set[str]
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Aggregate URDF fixed-link masses into the corresponding MuJoCo body."""
    parent_joints: dict[str, tuple[str, str]] = {}
    for joint in urdf_root.findall("joint"):
        child = joint.find("child")
        parent = joint.find("parent")
        if child is None or parent is None:
            continue
        parent_joints[child.get("link")] = (parent.get("link"), joint.get("type"))

    target_masses = {name: 0.0 for name in mujoco_body_names}
    source_links = {name: [] for name in mujoco_body_names}
    for link in urdf_root.findall("link"):
        mass_element = link.find("inertial/mass")
        if mass_element is None:
            continue

        source_name = link.get("name")
        destination_name = source_name
        while destination_name not in mujoco_body_names:
            if destination_name not in parent_joints:
                raise ValueError(
                    f"Cannot map URDF inertial link {source_name} to a MuJoCo body"
                )
            parent_name, joint_type = parent_joints[destination_name]
            if joint_type != "fixed":
                raise ValueError(
                    f"URDF link {source_name} reaches {parent_name} through non-fixed "
                    f"joint type {joint_type}"
                )
            destination_name = parent_name

        target_masses[destination_name] += float(mass_element.get("value"))
        source_links[destination_name].append(source_name)

    missing = sorted(name for name, mass in target_masses.items() if mass <= 0.0)
    if missing:
        raise ValueError(f"No positive ROS mass found for MuJoCo bodies: {missing}")
    return target_masses, source_links


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


def compile_ros_urdf(ros_urdf_path: Path) -> mujoco.MjModel:
    tree = ET.parse(ros_urdf_path)
    root = tree.getroot()
    compiler = root.find("mujoco/compiler")
    if compiler is None:
        raise ValueError(f"Missing <mujoco><compiler> in {ros_urdf_path}")

    # H2.urdf prefixes mesh filenames with meshes/, so compile from its parent.
    compiler.set("meshdir", ".")
    add_floating_base(root)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", dir=ros_urdf_path.parent, delete=False
    ) as temporary_urdf:
        tree.write(temporary_urdf, encoding="unicode")
        temporary_urdf_path = Path(temporary_urdf.name)
    try:
        model = mujoco.MjModel.from_xml_path(str(temporary_urdf_path))
    finally:
        temporary_urdf_path.unlink(missing_ok=True)

    body_names = {model.body(i).name for i in range(1, model.nbody)}
    compounds = collect_compound_inertials(root, body_names)
    for body_name, (mass, center, tensor) in compounds.items():
        body_id = model.body(body_name).id
        principal, rotation = np.linalg.eigh(tensor)
        order = np.argsort(principal)[::-1]
        principal = principal[order]
        rotation = rotation[:, order]
        if np.linalg.det(rotation) < 0.0:
            rotation[:, -1] *= -1.0
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
        model.body_mass[body_id] = mass
        model.body_ipos[body_id] = center
        model.body_inertia[body_id] = principal
        model.body_iquat[body_id] = quaternion
    return model


def replace_inertial(
    body: ET.Element,
    target_model: mujoco.MjModel,
    target_body_id: int,
    position_override: np.ndarray | None = None,
    orientation_override: np.ndarray | None = None,
    principal_inertia_override: np.ndarray | None = None,
) -> None:
    position = (
        target_model.body_ipos[target_body_id]
        if position_override is None
        else position_override
    )
    principal_inertia = (
        target_model.body_inertia[target_body_id]
        if principal_inertia_override is None
        else principal_inertia_override
    )
    orientation = (
        target_model.body_iquat[target_body_id]
        if orientation_override is None
        else orientation_override
    )
    inertial = ET.Element(
        "inertial",
        {
            "pos": format_values(position),
            "quat": format_values(orientation),
            "mass": f"{float(target_model.body_mass[target_body_id]):.12g}",
            "diaginertia": format_values(principal_inertia),
        },
    )
    existing = body.find("inertial")
    if existing is None:
        body.insert(0, inertial)
        return
    index = list(body).index(existing)
    body.remove(existing)
    body.insert(index, inertial)


def build_model(
    baseline_path: Path,
    ros_urdf_path: Path,
    output_path: Path,
    report_path: Path,
    pelvis_com_source: str = "ros",
    pelvis_inertial_orientation_source: str = "ros",
    pelvis_principal_inertia_source: str = "ros",
) -> None:
    baseline_model = mujoco.MjModel.from_xml_path(str(baseline_path))
    body_names = {baseline_model.body(i).name for i in range(1, baseline_model.nbody)}

    urdf_root = ET.parse(ros_urdf_path).getroot()
    target_masses, source_links = collect_ros_body_masses(urdf_root, body_names)
    target_model = compile_ros_urdf(ros_urdf_path)
    target_body_names = {target_model.body(i).name for i in range(1, target_model.nbody)}
    if target_body_names != body_names:
        raise ValueError(
            "Unitree MuJoCo/ROS body names differ: "
            f"mujoco_only={sorted(body_names - target_body_names)}, "
            f"ros_only={sorted(target_body_names - body_names)}"
    )
    for body_name, aggregated_mass in target_masses.items():
        compiled_mass = float(target_model.body_mass[target_model.body(body_name).id])
        if abs(compiled_mass - aggregated_mass) > 1e-9:
            raise ValueError(
                f"ROS fixed-link mass aggregation disagrees for {body_name}: "
                f"manual={aggregated_mass}, compiled={compiled_mass}"
            )

    tree = ET.parse(baseline_path)
    root = tree.getroot()
    if (
        pelvis_com_source == "unitree_mujoco"
        and pelvis_inertial_orientation_source == "unitree_mujoco"
        and pelvis_principal_inertia_source == "unitree_mujoco"
    ):
        root.set(
            "model", "Unitree H2 - ROS inertial, official pelvis frame/inertia"
        )
    elif pelvis_com_source == "unitree_mujoco":
        root.set("model", "Unitree H2 - ROS inertial, official pelvis COM")
    else:
        root.set("model", "Unitree H2 - ROS inertial aligned")
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError(f"Missing MJCF compiler in {baseline_path}")
    compiler.set("meshdir", "../h2/meshes")

    body_elements = {
        body.get("name"): body for body in root.findall(".//body[@name]")
    }
    if set(body_elements) != body_names:
        raise ValueError(
            "MJCF XML/compiled body names differ: "
            f"xml_only={sorted(set(body_elements) - body_names)}, "
            f"compiled_only={sorted(body_names - set(body_elements))}"
        )

    report_rows = []
    expected_ipos = target_model.body_ipos.copy()
    expected_iquat = target_model.body_iquat.copy()
    expected_inertia = target_model.body_inertia.copy()
    for body_id in range(1, baseline_model.nbody):
        body_name = baseline_model.body(body_id).name
        old_mass = float(baseline_model.body_mass[body_id])
        target_body_id = target_model.body(body_name).id
        target_mass = float(target_model.body_mass[target_body_id])
        body = body_elements[body_name]
        position_override = None
        if body_name == "pelvis" and pelvis_com_source == "unitree_mujoco":
            position_override = baseline_model.body_ipos[body_id]
            expected_ipos[target_body_id] = position_override
        orientation_override = None
        if (
            body_name == "pelvis"
            and pelvis_inertial_orientation_source == "unitree_mujoco"
        ):
            orientation_override = baseline_model.body_iquat[body_id]
            expected_iquat[target_body_id] = orientation_override
        principal_inertia_override = None
        if (
            body_name == "pelvis"
            and pelvis_principal_inertia_source == "unitree_mujoco"
        ):
            principal_inertia_override = baseline_model.body_inertia[body_id]
            expected_inertia[target_body_id] = principal_inertia_override
        replace_inertial(
            body,
            target_model,
            target_body_id,
            position_override,
            orientation_override,
            principal_inertia_override,
        )

        report_rows.append(
            {
                "body": body_name,
                "unitree_mujoco_mass_kg": f"{old_mass:.12g}",
                "unitree_ros_mass_kg": f"{target_mass:.12g}",
                "mass_scale": f"{target_mass / old_mass:.12g}",
                "unitree_mujoco_com_xyz": format_values(
                    baseline_model.body_ipos[body_id]
                ),
                "unitree_ros_com_xyz": format_values(
                    target_model.body_ipos[target_body_id]
                ),
                "aligned_com_xyz": format_values(expected_ipos[target_body_id]),
                "unitree_mujoco_inertial_quat": format_values(
                    baseline_model.body_iquat[body_id]
                ),
                "unitree_ros_inertial_quat": format_values(
                    target_model.body_iquat[target_body_id]
                ),
                "aligned_inertial_quat": format_values(
                    expected_iquat[target_body_id]
                ),
                "unitree_mujoco_principal_inertia": format_values(
                    baseline_model.body_inertia[body_id]
                ),
                "unitree_ros_principal_inertia": format_values(
                    target_model.body_inertia[target_body_id]
                ),
                "aligned_principal_inertia": format_values(
                    expected_inertia[target_body_id]
                ),
                "ros_source_links": "+".join(source_links[body_name]),
            }
        )

    ET.indent(tree, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", dir=output_path.parent, delete=False
    ) as temporary_output:
        tree.write(temporary_output, encoding="unicode")
        temporary_output_path = Path(temporary_output.name)

    try:
        aligned_model = mujoco.MjModel.from_xml_path(str(temporary_output_path))
        expected_values = {
            "body_mass": target_model.body_mass,
            "body_ipos": expected_ipos,
            "body_inertia": expected_inertia,
        }
        for field, expected in expected_values.items():
            if not np.allclose(
                getattr(aligned_model, field),
                expected,
                rtol=0.0,
                atol=1e-9,
            ):
                max_error = float(
                    np.max(
                        np.abs(getattr(aligned_model, field) - expected)
                    )
                )
                raise ValueError(
                    f"ROS inertial alignment failed for {field}: max error={max_error}"
                )
        quaternion_alignment = np.abs(
            np.sum(aligned_model.body_iquat * expected_iquat, axis=1)
        )
        if not np.allclose(quaternion_alignment, 1.0, rtol=0.0, atol=1e-9):
            raise ValueError("ROS inertial alignment failed for body_iquat")

        exact_fields = (
            "body_parentid",
            "jnt_type",
            "jnt_bodyid",
            "geom_type",
            "geom_bodyid",
            "geom_contype",
            "geom_conaffinity",
            "geom_condim",
            "actuator_trnid",
            "actuator_dyntype",
            "actuator_gaintype",
            "actuator_biastype",
            "sensor_type",
            "sensor_objtype",
            "sensor_objid",
            "sensor_dim",
        )
        for field in exact_fields:
            if not np.array_equal(getattr(baseline_model, field), getattr(aligned_model, field)):
                raise ValueError(f"Mass alignment unexpectedly changed {field}")

        unchanged_float_fields = (
            "body_pos",
            "body_quat",
            "jnt_pos",
            "jnt_axis",
            "jnt_range",
            "geom_pos",
            "geom_quat",
            "geom_size",
            "geom_friction",
            "geom_solref",
            "geom_solimp",
            "geom_margin",
            "geom_gap",
            "actuator_ctrlrange",
            "actuator_forcerange",
            "actuator_gear",
            "sensor_cutoff",
            "sensor_noise",
            "mesh_vert",
        )
        for field in unchanged_float_fields:
            if not np.array_equal(getattr(baseline_model, field), getattr(aligned_model, field)):
                raise ValueError(f"Mass alignment unexpectedly changed {field}")

        if baseline_model.nu != aligned_model.nu or baseline_model.nsensor != aligned_model.nsensor:
            raise ValueError("Inertial alignment changed actuator or sensor counts")

        temporary_output_path.replace(output_path)
    finally:
        temporary_output_path.unlink(missing_ok=True)

    with report_path.open("w", newline="") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    print(
        f"Aligned mass, COM, and inertia for {len(body_names)} bodies; total mass "
        f"{baseline_model.body_mass.sum():.9f} -> {aligned_model.body_mass.sum():.9f} kg"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--ros-urdf", type=Path, default=DEFAULT_ROS_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_OUTPUT.parent / "mass_alignment.csv"
    )
    parser.add_argument(
        "--pelvis-com-source",
        choices=("ros", "unitree_mujoco"),
        default="ros",
        help="Select only the pelvis COM source; all other inertials come from ROS.",
    )
    parser.add_argument(
        "--pelvis-inertial-orientation-source",
        choices=("ros", "unitree_mujoco"),
        default="ros",
        help="Select only the pelvis principal-axis orientation source.",
    )
    parser.add_argument(
        "--pelvis-principal-inertia-source",
        choices=("ros", "unitree_mujoco"),
        default="ros",
        help="Select only the pelvis principal inertia values source.",
    )
    args = parser.parse_args()
    build_model(
        args.baseline.resolve(),
        args.ros_urdf.resolve(),
        args.output.resolve(),
        args.report.resolve(),
        args.pelvis_com_source,
        args.pelvis_inertial_orientation_source,
        args.pelvis_principal_inertia_source,
    )


if __name__ == "__main__":
    main()
