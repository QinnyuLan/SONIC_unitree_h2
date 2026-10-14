"""MuJoCo simulation environment and loop for the released Unitree H2.

DefaultEnv owns the MuJoCo model/data, computes PD torques from Unitree SDK
commands, steps physics, and publishes observations back via the SDK bridge.
BaseSimulator wraps DefaultEnv with rate-limiting and viewer/image update loops.
"""

import os
import pathlib
from pathlib import Path
import pickle
import tempfile
from threading import Lock, Thread
import time
from typing import Dict
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from sim.contacts import check_contact
from sim.robot import Robot
from sim.realtime import RealtimePacer
from sim.physics import get_subtree_body_names
from sim.dds import (
    H2_JOINT_NAMES_LOCAL_MUJOCO,
    ElasticBand,
    UnitreeSdk2Bridge,
)

from sim.config import ROOT


class DefaultEnv:
    """Base environment class that handles simulation environment setup and step"""

    def __init__(
        self,
        config: Dict[str, any],
        env_name: str = "default",
        camera_configs: Dict[str, any] = {},
        onscreen: bool = False,
        offscreen: bool = False,
        enable_image_publish: bool = False,
    ):
        self.config = config
        self.env_name = env_name
        self.robot = Robot(self.config)
        self.num_body_dof = self.robot.NUM_JOINTS
        self.num_hand_dof = self.robot.NUM_HAND_JOINTS
        self.sim_dt = self.config["SIMULATE_DT"]
        self.obs = None
        self.configured_torque_limit = np.array(self.robot.MOTOR_EFFORT_LIMIT_LIST, dtype=float)
        self.torques = np.zeros(self.num_body_dof + self.num_hand_dof * 2)
        self.torque_limit = self.configured_torque_limit.copy()
        self.body_torque_limit = self.configured_torque_limit[: self.num_body_dof].copy()
        self.camera_configs = camera_configs

        if not camera_configs and offscreen and enable_image_publish:
            self.camera_configs = {
                "ego_view": {"height": 480, "width": 640, "mjcf_name": "head_camera"},
            }

        self.reward_lock = Lock()
        self.unitree_bridge = None
        self.elastic_band = None
        self.onscreen = onscreen
        self.body_joint_names: list[str] = []

        self.init_scene()
        self.last_reward = 0

        self.offscreen = offscreen
        if self.offscreen:
            self.init_renderers()
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.image_publish_process = None
        self.diagnostics = None
        if config.get("DIAGNOSTICS_DIR"):
            from sim.diagnostics.squat import SquatDiagnostics
            self.diagnostics = SquatDiagnostics(self, config["DIAGNOSTICS_DIR"],
                                                config.get("DIAGNOSTICS_HZ", 200))

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        from sim.rendering.image_publish import ImagePublishProcess

        if len(self.camera_configs) == 0:
            print(
                "Warning: No camera configs provided, image publishing subprocess will not be started"
            )
            return
        start_method = self.config.get("MP_START_METHOD", "spawn")
        self.image_publish_process = ImagePublishProcess(
            camera_configs=self.camera_configs,
            image_dt=self.image_dt,
            zmq_port=camera_port,
            start_method=start_method,
            verbose=self.config.get("verbose", False),
        )
        self.image_publish_process.start_process()

    def _get_dof_indices_by_class(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            joint_class_map = {}
            for joint_element in root.findall(".//joint[@class]"):
                joint_name = joint_element.get("name")
                joint_class = joint_element.get("class")
                if joint_name and joint_class:
                    joint_id = mujoco.mj_name2id(
                        self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                    )
                    if joint_id != -1:
                        dof_adr = self.mj_model.jnt_dofadr[joint_id]
                        if joint_class not in joint_class_map:
                            joint_class_map[joint_class] = []
                        joint_class_map[joint_class].append(dof_adr)
        finally:
            os.remove(temp_xml_path)

        return joint_class_map

    def _get_default_dof_properties(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            default_dof_properties = {}
            for default_element in root.findall(".//default/default[@class]"):
                class_name = default_element.get("class")
                joint_element = default_element.find("joint")
                if class_name and joint_element is not None:
                    properties = {}
                    if "damping" in joint_element.attrib:
                        properties["damping"] = float(joint_element.get("damping"))
                    if "armature" in joint_element.attrib:
                        properties["armature"] = float(joint_element.get("armature"))
                    if "frictionloss" in joint_element.attrib:
                        properties["frictionloss"] = float(joint_element.get("frictionloss"))

                    if properties:
                        default_dof_properties[class_name] = properties
        finally:
            os.remove(temp_xml_path)

        return default_dof_properties

    def _build_runtime_scene_xml(self, xml_path: pathlib.Path) -> str:
        """Create a temporary MuJoCo scene without modifying robot asset files."""
        meshdir = self.config.get("MJCF_MESHDIR")
        add_scene = self.config.get("ADD_DEFAULT_MUJOCO_SCENE", False)
        add_h2_foot_contacts = self.config.get("ADD_H2_OFFICIAL_FOOT_CONTACT_GEOMS", False)
        ankle_roll_backlash = self.config.get("H2_ANKLE_ROLL_BACKLASH_DEG")
        ankle_pitch_backlash = self.config.get("H2_ANKLE_PITCH_BACKLASH_DEG")
        waist_backlash = self.config.get("H2_WAIST_BACKLASH_DEG")
        paper_style = self.config.get("VISUAL_STYLE") == "paper"
        if (
            not meshdir
            and not add_scene
            and not add_h2_foot_contacts
            and not ankle_roll_backlash
            and not ankle_pitch_backlash
            and not waist_backlash
            and not paper_style
        ):
            return str(xml_path)

        tree = ET.parse(xml_path)
        root = tree.getroot()

        if paper_style:
            from sim.rendering.presentation import expand_presentation_includes
            expand_presentation_includes(root, xml_path)

        if meshdir:
            meshdir_path = pathlib.Path(meshdir)
            if not meshdir_path.is_absolute():
                meshdir_path = pathlib.Path(ROOT) / meshdir_path
            compiler = root.find("compiler")
            if compiler is None:
                compiler = ET.Element("compiler")
                root.insert(0, compiler)
            compiler.set("meshdir", str(meshdir_path.resolve()))

        if add_scene:
            if root.find("statistic") is None:
                ET.SubElement(root, "statistic", {"center": "0 0 0.55", "extent": "2.0"})

            if root.find("visual") is None:
                visual = ET.SubElement(root, "visual")
                ET.SubElement(
                    visual,
                    "headlight",
                    {
                        "diffuse": "0.6 0.6 0.6",
                        "ambient": "0.3 0.3 0.3",
                        "specular": "0 0 0",
                    },
                )
                ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
                ET.SubElement(visual, "global", {"azimuth": "120", "elevation": "-20"})

            asset = root.find("asset")
            if asset is None:
                asset = ET.SubElement(root, "asset")
            if asset.find("texture[@name='groundplane']") is None:
                ET.SubElement(
                    asset,
                    "texture",
                    {
                        "type": "2d",
                        "name": "groundplane",
                        "builtin": "checker",
                        "mark": "edge",
                        "rgb1": "0.2 0.3 0.4",
                        "rgb2": "0.1 0.2 0.3",
                        "markrgb": "0.8 0.8 0.8",
                        "width": "300",
                        "height": "300",
                    },
                )
            if asset.find("material[@name='groundplane']") is None:
                ET.SubElement(
                    asset,
                    "material",
                    {
                        "name": "groundplane",
                        "texture": "groundplane",
                        "texuniform": "true",
                        "texrepeat": "5 5",
                        "reflectance": "0.2",
                    },
                )

            worldbody = root.find("worldbody")
            if worldbody is None:
                worldbody = ET.SubElement(root, "worldbody")
            if worldbody.find("geom[@name='floor']") is None:
                ET.SubElement(
                    worldbody,
                    "geom",
                    {
                        "name": "floor",
                        "size": "0 0 0.05",
                        "type": "plane",
                        "material": "groundplane",
                    },
                )
            if worldbody.find("light[@name='scene_light']") is None:
                ET.SubElement(
                    worldbody,
                    "light",
                    {
                        "name": "scene_light",
                        "pos": "0 0 1.5",
                        "dir": "0 0 -1",
                        "directional": "true",
                    },
                )
            if worldbody.find("camera[@name='global_view']") is None:
                ET.SubElement(
                    worldbody,
                    "camera",
                    {
                        "name": "global_view",
                        "pos": "1.6 -2.0 1.3",
                        "xyaxes": "0.78 0.63 0 -0.27 0.34 0.90",
                    },
                )

        if add_h2_foot_contacts:
            self._patch_h2_official_foot_contact_geoms(root)
        if ankle_roll_backlash:
            self._patch_h2_ankle_roll_backlash(root, ankle_roll_backlash)
        if ankle_pitch_backlash:
            self._patch_h2_ankle_pitch_backlash(root, ankle_pitch_backlash)
        if waist_backlash:
            self._patch_h2_waist_backlash(root, waist_backlash)

        if paper_style:
            from sim.rendering.presentation import apply_paper_style
            apply_paper_style(root)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".xml") as f:
            tree.write(f, encoding="unicode")
            return f.name

    def _patch_h2_ankle_roll_backlash(
        self, root: ET.Element, backlash_degrees: dict[str, float]
    ) -> None:
        """Insert passive limited joints after the H2 ankle-roll motor shafts."""
        configured_widths = []
        for side in ("left", "right"):
            half_width_deg = float(backlash_degrees.get(side, 0.0))
            if half_width_deg <= 0.0:
                raise ValueError(
                    f"H2 ankle-roll backlash for {side} must be positive, "
                    f"got {half_width_deg}"
                )

            roll_body = root.find(f".//body[@name='{side}_ankle_roll_link']")
            if roll_body is None:
                raise ValueError(
                    f"H2 ankle-roll backlash could not find {side}_ankle_roll_link"
                )
            parent = next(
                (candidate for candidate in root.iter("body") if roll_body in candidate),
                None,
            )
            if parent is None:
                raise ValueError(
                    f"H2 ankle-roll backlash could not find parent of {side}_ankle_roll_link"
                )

            motor_joint = roll_body.find(f"joint[@name='{side}_ankle_roll_joint']")
            if motor_joint is None:
                raise ValueError(
                    f"H2 ankle-roll backlash could not find {side}_ankle_roll_joint"
                )

            body_index = list(parent).index(roll_body)
            original_pos = roll_body.get("pos", "0 0 0")
            original_quat = roll_body.get("quat")
            carrier_attributes = {
                "name": f"{side}_ankle_roll_motor_carrier",
                "pos": original_pos,
            }
            if original_quat is not None:
                carrier_attributes["quat"] = original_quat
            carrier = ET.Element("body", carrier_attributes)
            ET.SubElement(
                carrier,
                "inertial",
                {
                    "pos": "0 0 0",
                    "mass": "1e-6",
                    "diaginertia": "1e-9 1e-9 1e-9",
                },
            )

            roll_body.remove(motor_joint)
            carrier.append(motor_joint)
            roll_body.set("pos", "0 0 0")
            roll_body.attrib.pop("quat", None)
            half_width_rad = np.deg2rad(half_width_deg)
            roll_body.insert(
                0,
                ET.Element(
                    "joint",
                    {
                        "name": f"{side}_ankle_roll_backlash_joint",
                        "axis": "1 0 0",
                        "limited": "true",
                        "range": f"{-half_width_rad:.12g} {half_width_rad:.12g}",
                        "damping": "0",
                        "frictionloss": "0",
                        "armature": "0",
                        "solreflimit": "0.002 1",
                    },
                ),
            )
            parent.remove(roll_body)
            carrier.append(roll_body)
            parent.insert(body_index, carrier)
            configured_widths.append(f"{side}=+/-{half_width_deg:g} deg")
        print(f"[MuJoCo] H2 ankle-roll backlash: {', '.join(configured_widths)}")

    def _patch_h2_ankle_pitch_backlash(
        self, root: ET.Element, backlash_degrees: dict[str, float]
    ) -> None:
        """Insert passive pitch play downstream of the H2 ankle closed loop."""
        configured_widths = []
        for side, configured_half_width in backlash_degrees.items():
            if side not in {"left", "right"}:
                raise ValueError(f"Unknown H2 ankle-pitch backlash side: {side}")
            half_width_deg = float(configured_half_width)
            if half_width_deg <= 0.0:
                raise ValueError(
                    f"H2 ankle-pitch backlash for {side} must be positive, "
                    f"got {half_width_deg}"
                )

            pitch_body = root.find(f".//body[@name='{side}_ankle_pitch_link']")
            if pitch_body is None:
                raise ValueError(
                    f"H2 ankle-pitch backlash could not find {side}_ankle_pitch_link"
                )
            parent = next(
                (candidate for candidate in root.iter("body") if pitch_body in candidate),
                None,
            )
            if parent is None:
                raise ValueError(
                    f"H2 ankle-pitch backlash could not find parent of "
                    f"{side}_ankle_pitch_link"
                )

            output_joint = pitch_body.find(f"joint[@name='{side}_ankle_pitch_joint']")
            if output_joint is None:
                raise ValueError(
                    f"H2 ankle-pitch backlash could not find {side}_ankle_pitch_joint"
                )
            equality = root.find(f".//equality/connect[@name='{side}_ankle_connect']")
            if equality is None or equality.get("body1") != f"{side}_ankle_pitch_link":
                raise ValueError(
                    f"H2 ankle-pitch backlash requires the official "
                    f"{side}_ankle_connect structure"
                )

            body_index = list(parent).index(pitch_body)
            original_pos = pitch_body.get("pos", "0 0 0")
            original_quat = pitch_body.get("quat")
            carrier_name = f"{side}_ankle_pitch_output_carrier"
            carrier_attributes = {"name": carrier_name, "pos": original_pos}
            if original_quat is not None:
                carrier_attributes["quat"] = original_quat
            carrier = ET.Element("body", carrier_attributes)
            ET.SubElement(
                carrier,
                "inertial",
                {
                    "pos": "0 0 0",
                    "mass": "1e-6",
                    "diaginertia": "1e-9 1e-9 1e-9",
                },
            )

            pitch_body.remove(output_joint)
            carrier.append(output_joint)
            pitch_body.set("pos", "0 0 0")
            pitch_body.attrib.pop("quat", None)
            half_width_rad = np.deg2rad(half_width_deg)
            pitch_body.insert(
                0,
                ET.Element(
                    "joint",
                    {
                        "name": f"{side}_ankle_pitch_backlash_joint",
                        "axis": "0 1 0",
                        "limited": "true",
                        "range": f"{-half_width_rad:.12g} {half_width_rad:.12g}",
                        "damping": "0",
                        "frictionloss": "0",
                        "armature": "0",
                        "solreflimit": "0.002 1",
                    },
                ),
            )
            parent.remove(pitch_body)
            carrier.append(pitch_body)
            parent.insert(body_index, carrier)

            # Keep Unitree's rod constraint on the original pitch output. The
            # downstream physical foot is free to traverse the added play.
            equality.set("body1", carrier_name)
            configured_widths.append(f"{side}=+/-{half_width_deg:g} deg")
        print(f"[MuJoCo] H2 ankle-pitch backlash: {', '.join(configured_widths)}")

    def _patch_h2_waist_backlash(
        self, root: ET.Element, backlash_degrees: dict[str, float]
    ) -> None:
        """Add passive roll/pitch play after the H2 waist closed-loop output."""
        expected_axes = {"roll": "1 0 0", "pitch": "0 1 0"}
        unknown_axes = set(backlash_degrees) - set(expected_axes)
        if unknown_axes:
            raise ValueError(f"Unknown H2 waist backlash axes: {sorted(unknown_axes)}")
        missing_axes = set(expected_axes) - set(backlash_degrees)
        if missing_axes:
            raise ValueError(f"Missing H2 waist backlash axes: {sorted(missing_axes)}")

        torso_body = root.find(".//body[@name='torso_link']")
        if torso_body is None:
            raise ValueError("H2 waist backlash could not find torso_link")
        waist_roll_body = next(
            (candidate for candidate in root.iter("body") if torso_body in candidate),
            None,
        )
        if waist_roll_body is None or waist_roll_body.get("name") != "waist_roll_link":
            raise ValueError(
                "H2 waist backlash requires torso_link under waist_roll_link"
            )
        pitch_joint = torso_body.find("joint[@name='waist_pitch_joint']")
        if pitch_joint is None:
            raise ValueError("H2 waist backlash could not find waist_pitch_joint")

        motor_bodies = []
        for name in ("torso_constraint_R_link", "torso_constraint_L_link"):
            motor_body = torso_body.find(f"body[@name='{name}']")
            if motor_body is None:
                raise ValueError(f"H2 waist backlash could not find {name}")
            motor_bodies.append(motor_body)

        torso_index = list(waist_roll_body).index(torso_body)
        original_pos = torso_body.get("pos", "0 0 0")
        original_quat = torso_body.get("quat")
        carrier_attributes = {
            "name": "waist_roll_pitch_output_carrier",
            "pos": original_pos,
        }
        if original_quat is not None:
            carrier_attributes["quat"] = original_quat
        carrier = ET.Element("body", carrier_attributes)
        ET.SubElement(
            carrier,
            "inertial",
            {
                "pos": "0 0 0",
                "mass": "1e-6",
                "diaginertia": "1e-9 1e-9 1e-9",
            },
        )

        torso_body.remove(pitch_joint)
        carrier.append(pitch_joint)
        for motor_body in motor_bodies:
            torso_body.remove(motor_body)
            carrier.append(motor_body)

        torso_body.set("pos", "0 0 0")
        torso_body.attrib.pop("quat", None)
        for insert_index, (axis_name, axis) in enumerate(expected_axes.items()):
            half_width_deg = float(backlash_degrees[axis_name])
            if half_width_deg <= 0.0:
                raise ValueError(
                    f"H2 waist {axis_name} backlash must be positive, "
                    f"got {half_width_deg}"
                )
            half_width_rad = np.deg2rad(half_width_deg)
            torso_body.insert(
                insert_index,
                ET.Element(
                    "joint",
                    {
                        "name": f"waist_{axis_name}_backlash_joint",
                        "axis": axis,
                        "limited": "true",
                        "range": f"{-half_width_rad:.12g} {half_width_rad:.12g}",
                        "damping": "0",
                        "frictionloss": "0",
                        "armature": "0",
                        "solreflimit": "0.002 1",
                    },
                ),
            )

        waist_roll_body.remove(torso_body)
        carrier.append(torso_body)
        waist_roll_body.insert(torso_index, carrier)
        print(
            "[MuJoCo] H2 waist backlash: "
            f"roll=+/-{float(backlash_degrees['roll']):g} deg, "
            f"pitch=+/-{float(backlash_degrees['pitch']):g} deg"
        )

    def _patch_h2_official_foot_contact_geoms(self, root: ET.Element) -> None:
        """Use Unitree's H2 sole contact geoms for the runtime MuJoCo scene.

        The H2 MJCF used by training assets can contain only ankle-pitch mesh
        collision.  In MuJoCo that leaves the default bent-knee H2LoRA stand
        with the COM in front of the effective support polygon.  Unitree's
        official h2_mujoco.xml disables ankle mesh contact and adds small
        sphere/box sole geoms; mirror that contact model at runtime.
        """

        def find_body(body_name: str) -> ET.Element | None:
            return root.find(f".//body[@name='{body_name}']")

        def disable_mesh_contacts(body: ET.Element) -> None:
            for geom in body.findall("geom"):
                if geom.get("type") == "mesh":
                    geom.set("contype", "0")
                    geom.set("conaffinity", "0")
                    geom.set("group", geom.get("group", "1"))

        def add_geom_if_missing(body: ET.Element, name: str, attrs: dict[str, str]) -> None:
            if body.find(f"geom[@name='{name}']") is not None:
                return
            ET.SubElement(body, "geom", {"name": name, **attrs})

        foot_specs = {
            "left_ankle_pitch_link": {
                "prefix": "left",
                "points": [
                    (-0.075, 0.066, -0.05),
                    (-0.075, 0.007, -0.05),
                    (0.100, 0.080, -0.048),
                    (0.100, -0.007, -0.048),
                    (0.150, 0.066, -0.05),
                    (0.150, 0.007, -0.05),
                ],
                "box_pos": (0.035, 0.0365, -0.04),
            },
            "right_ankle_pitch_link": {
                "prefix": "right",
                "points": [
                    (-0.075, -0.066, -0.05),
                    (-0.075, -0.007, -0.05),
                    (0.100, -0.080, -0.048),
                    (0.100, 0.007, -0.048),
                    (0.150, -0.066, -0.05),
                    (0.150, -0.007, -0.05),
                ],
                "box_pos": (0.035, -0.0365, -0.04),
            },
        }

        for body_name, spec in foot_specs.items():
            body = find_body(body_name)
            if body is None:
                raise ValueError(f"H2 foot contact patch could not find body: {body_name}")
            disable_mesh_contacts(body)

            prefix = spec["prefix"]
            for idx, pos in enumerate(spec["points"]):
                add_geom_if_missing(
                    body,
                    f"h2lora_{prefix}_sole_point_{idx}",
                    {
                        "size": "0.008",
                        "pos": f"{pos[0]:.6g} {pos[1]:.6g} {pos[2]:.6g}",
                        "rgba": "0.4 0.4 0.4 1",
                    },
                )
            box_pos = spec["box_pos"]
            add_geom_if_missing(
                body,
                f"h2lora_{prefix}_sole_box",
                {
                    "size": "0.115 0.04 0.004",
                    "pos": f"{box_pos[0]:.6g} {box_pos[1]:.6g} {box_pos[2]:.6g}",
                    "type": "box",
                    "rgba": "0.4 0.4 0.4 1",
                },
            )

    def init_scene(self):
        """Initialize the default robot scene"""
        configured_xml_path = pathlib.Path(self.config["ROBOT_SCENE"])
        if not configured_xml_path.is_absolute():
            configured_xml_path = pathlib.Path(ROOT) / configured_xml_path
        xml_path = str(configured_xml_path)
        runtime_xml_path = self._build_runtime_scene_xml(configured_xml_path)
        try:
            self.mj_model = mujoco.MjModel.from_xml_path(runtime_xml_path)
        finally:
            if runtime_xml_path != xml_path:
                os.remove(runtime_xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        self._apply_mujoco_runtime_options()
        self.torso_index = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.root_body = "pelvis"
        self.root_body_id = self.mj_model.body(self.root_body).id

        self.joint_class_map = self._get_dof_indices_by_class()

        self.perform_sysid_search = self.config.get("perform_sysid_search", False)

        # Check for static root link (fixed base)
        self.use_floating_root_link = "floating_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]
        self.use_constrained_root_link = "constrained_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]

        # MuJoCo qpos/qvel arrays start with root DOFs before joint DOFs:
        # floating base has 7 qpos (pos + quat) and 6 qvel (lin + ang velocity)
        if self.use_floating_root_link:
            self.qpos_offset = 7
            self.qvel_offset = 6
        else:
            if self.use_constrained_root_link:
                self.qpos_offset = 1
                self.qvel_offset = 1
            else:
                raise ValueError(
                    "No root link found --"
                    "The absolute static root will make the simulation unstable."
                )

        # Enable the elastic band
        if self.config["ENABLE_ELASTIC_BAND"] and self.use_floating_root_link:
            self.elastic_band = ElasticBand(
                point=self.config.get("ELASTIC_BAND_POINT"),
                length=self.config.get("ELASTIC_BAND_LENGTH", 0.0),
                kp_pos=self.config.get("ELASTIC_BAND_KP_POS", 10000.0),
                kd_pos=self.config.get("ELASTIC_BAND_KD_POS", 1000.0),
                kp_ang=self.config.get("ELASTIC_BAND_KP_ANG", 1000.0),
                kd_ang=self.config.get("ELASTIC_BAND_KD_ANG", 10.0),
            )
            if "g1" in self.config["ROBOT_TYPE"]:
                if self.config["enable_waist"]:
                    self.band_attached_link = self.mj_model.body("pelvis").id
                else:
                    self.band_attached_link = self.mj_model.body("torso_link").id
            elif "h1" in self.config["ROBOT_TYPE"] or "h2" in self.config["ROBOT_TYPE"]:
                self.band_attached_link = self.mj_model.body("torso_link").id
            else:
                self.band_attached_link = self.mj_model.body("base_link").id

            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model,
                    self.mj_data,
                    key_callback=self.elastic_band.MujuocoKeyCallback,
                    show_left_ui=False,
                    show_right_ui=False,
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None
        else:
            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None

        if self.viewer:
            self.viewer.cam.azimuth = 120
            self.viewer.cam.elevation = -30
            if "h2" in self.config["ROBOT_TYPE"]:
                self.viewer.cam.distance = 3.0
                self.viewer.cam.lookat = np.array([0, 0, 0.9])
            else:
                self.viewer.cam.distance = 2.0
                self.viewer.cam.lookat = np.array([0, 0, 0.5])
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.mj_model.body("pelvis").id
            if self.config.get("VISUAL_STYLE") == "paper":
                from sim.rendering.presentation import configure_presentation_camera
                configure_presentation_camera(self.viewer.cam, self.mj_model, self.config)

        self.body_joint_index = []
        self.left_hand_index = []
        self.right_hand_index = []
        configured_body_joint_names = self.config.get("MUJOCO_BODY_JOINT_NAMES")
        if configured_body_joint_names:
            self.body_joint_index = self._joint_ids_from_names(
                configured_body_joint_names, "MUJOCO_BODY_JOINT_NAMES"
            )
        else:
            for i in range(self.mj_model.njnt):
                name = self.mj_model.joint(i).name
                body_name_fragments = [
                    "hip",
                    "knee",
                    "ankle",
                    "waist",
                    "shoulder",
                    "elbow",
                    "wrist",
                ]
                if "h2" in self.config["ROBOT_TYPE"]:
                    body_name_fragments.append("head")
                if any(part_name in name for part_name in body_name_fragments):
                    self.body_joint_index.append(i)

        for i in range(self.mj_model.njnt):
            name = self.mj_model.joint(i).name
            if "left_hand" in name:
                self.left_hand_index.append(i)
            elif "right_hand" in name:
                self.right_hand_index.append(i)

        assert len(self.body_joint_index) == self.robot.NUM_JOINTS
        assert len(self.left_hand_index) == self.robot.NUM_HAND_JOINTS
        assert len(self.right_hand_index) == self.robot.NUM_HAND_JOINTS

        self.body_joint_index = np.asarray(self.body_joint_index, dtype=np.int64)
        self.left_hand_index = np.asarray(self.left_hand_index, dtype=np.int64)
        self.right_hand_index = np.asarray(self.right_hand_index, dtype=np.int64)
        self.body_joint_names = [
            self.mj_model.joint(int(joint_id)).name for joint_id in self.body_joint_index
        ]
        logical_joint_names = self.config.get(
            "MUJOCO_BODY_JOINT_LOGICAL_NAMES", self.body_joint_names
        )
        if len(logical_joint_names) != len(self.body_joint_names):
            raise ValueError(
                "MUJOCO_BODY_JOINT_LOGICAL_NAMES must match the controlled body "
                f"joint count: {len(logical_joint_names)} vs {len(self.body_joint_names)}"
            )
        self.body_joint_logical_names = list(logical_joint_names)
        self.config["MUJOCO_BODY_JOINT_NAMES"] = self.body_joint_names
        self.config["MUJOCO_BODY_JOINT_LOGICAL_NAMES"] = self.body_joint_logical_names

        configured_actuator_joint_names = self.config.get(
            "MUJOCO_BODY_ACTUATOR_JOINT_NAMES", self.body_joint_names
        )
        self.body_actuator_joint_index = np.asarray(
            self._joint_ids_from_names(
                configured_actuator_joint_names,
                "MUJOCO_BODY_ACTUATOR_JOINT_NAMES",
            ),
            dtype=np.int64,
        )
        if len(self.body_actuator_joint_index) != len(self.body_joint_index):
            raise ValueError(
                "MUJOCO_BODY_ACTUATOR_JOINT_NAMES must match the controlled body "
                f"joint count: {len(self.body_actuator_joint_index)} vs "
                f"{len(self.body_joint_index)}"
            )

        self.body_qpos_index = self.mj_model.jnt_qposadr[self.body_joint_index]
        self.body_dof_index = self.mj_model.jnt_dofadr[self.body_joint_index]
        self.left_hand_qpos_index = self.mj_model.jnt_qposadr[self.left_hand_index]
        self.left_hand_dof_index = self.mj_model.jnt_dofadr[self.left_hand_index]
        self.right_hand_qpos_index = self.mj_model.jnt_qposadr[self.right_hand_index]
        self.right_hand_dof_index = self.mj_model.jnt_dofadr[self.right_hand_index]

        self._configure_closed_loop_transmissions()
        self._configure_actuator_maps_and_limits()

        if self.config.get("OVERRIDE_MUJOCO_JOINT_FORCE_LIMITS", False):
            self._override_mujoco_joint_force_limits()

        if self.config.get("OVERRIDE_MUJOCO_DOF_PASSIVE", False):
            self._override_mujoco_dof_passive()

        if self.config.get("APPLY_DEFAULT_POSE_ON_RESET", False):
            self._apply_default_pose()

        self.ankle_response = None
        from sim.ankle_response import AnkleResponse, model_response_config
        response_config = model_response_config(self.mj_model, self.config)
        if response_config:
            self.ankle_response = AnkleResponse(self.mj_model, response_config)
        self._configure_motor_response()

    def _configure_motor_response(self):
        from sim.motor_response import (
            H2_MOTOR_GROUP_TRANSMISSIONS, MotorResponse,
        )

        self.motor_response = None
        self.motor_response_groups = {}
        response_config = self.config.get("H2_MOTOR_RESPONSE", {})
        if not isinstance(response_config, dict):
            raise ValueError("H2_MOTOR_RESPONSE must be a group parameter dictionary")
        if not response_config:
            return
        transmissions = {group["name"]: group for group in self.closed_loop_transmissions}
        self.motor_response_groups = {
            name: transmissions[transmission]["motor_positions"].tolist()
            for name, transmission in H2_MOTOR_GROUP_TRANSMISSIONS.items()
            if transmission in transmissions
        }
        self.motor_response = MotorResponse(
            self.body_torque_limit, self.motor_response_groups, response_config
        )

    def _joint_ids_from_names(self, joint_names, key: str) -> list[int]:
        joint_ids = []
        missing = []
        duplicates = []
        seen = set()
        for name in joint_names:
            if name in seen:
                duplicates.append(name)
                continue
            seen.add(name)
            joint_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                missing.append(name)
            else:
                joint_ids.append(joint_id)
        if missing or duplicates:
            raise ValueError(
                f"Invalid {key}: missing={missing}, duplicates={duplicates}"
            )
        return joint_ids

    def _build_actuator_index_for_joints(
        self, joint_indices: np.ndarray, label: str
    ) -> np.ndarray:
        """Map MuJoCo joint ids to actuator ids using model transmissions."""
        joint_to_actuator: dict[int, int] = {}
        for actuator_id in range(self.mj_model.nu):
            joint_id = int(self.mj_model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                continue
            if joint_id in joint_to_actuator:
                first_name = self.mj_model.actuator(joint_to_actuator[joint_id]).name
                second_name = self.mj_model.actuator(actuator_id).name
                joint_name = self.mj_model.joint(joint_id).name
                raise ValueError(
                    f"Multiple actuators found for {label} joint {joint_name}: "
                    f"{first_name}, {second_name}"
                )
            joint_to_actuator[joint_id] = actuator_id

        actuator_indices = []
        missing_joint_names = []
        for joint_id in np.asarray(joint_indices, dtype=np.int64):
            actuator_id = joint_to_actuator.get(int(joint_id))
            if actuator_id is None:
                missing_joint_names.append(self.mj_model.joint(int(joint_id)).name)
                continue
            actuator_indices.append(actuator_id)

        if missing_joint_names:
            raise ValueError(
                f"Could not find MuJoCo actuators for {label} joints: "
                f"{missing_joint_names}. Sim2sim requires joint actuators for all motors."
            )

        return np.asarray(actuator_indices, dtype=np.int64)

    def _configure_closed_loop_transmissions(self) -> None:
        """Resolve optional output-joint to motor-joint closed-loop mappings."""
        self.closed_loop_transmissions = []
        self._closed_loop_jacobian_cache_time = None
        self._closed_loop_jacobian_cache = {}
        self._closed_loop_equality_layout = None

        body_name_to_index = {
            name: index for index, name in enumerate(self.body_joint_names)
        }
        actuator_joint_names = [
            self.mj_model.joint(int(joint_id)).name
            for joint_id in self.body_actuator_joint_index
        ]
        actuator_name_to_index = {
            name: index for index, name in enumerate(actuator_joint_names)
        }

        for group_index, spec in enumerate(
            self.config.get("MUJOCO_CLOSED_LOOP_TRANSMISSIONS", [])
        ):
            label = spec.get("name", f"closed_loop_{group_index}")
            output_names = list(spec.get("output_joints", []))
            motor_names = list(spec.get("motor_joints", []))
            auxiliary_names = list(spec.get("auxiliary_joints", []))
            equality_names = list(spec.get("equalities", []))
            if not output_names or len(output_names) != len(motor_names):
                raise ValueError(
                    f"{label}: output_joints and motor_joints must be non-empty "
                    "and have the same length"
                )

            missing_outputs = [
                name for name in output_names if name not in body_name_to_index
            ]
            missing_motors = [
                name for name in motor_names if name not in actuator_name_to_index
            ]
            if missing_outputs or missing_motors:
                raise ValueError(
                    f"{label}: missing output joints {missing_outputs} or motor "
                    f"joints {missing_motors}"
                )

            output_joint_ids = self._joint_ids_from_names(
                output_names, f"{label}.output_joints"
            )
            motor_joint_ids = self._joint_ids_from_names(
                motor_names, f"{label}.motor_joints"
            )
            auxiliary_joint_ids = self._joint_ids_from_names(
                auxiliary_names, f"{label}.auxiliary_joints"
            )
            equality_ids = []
            missing_equalities = []
            for name in equality_names:
                equality_id = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, name
                )
                if equality_id < 0:
                    missing_equalities.append(name)
                else:
                    equality_ids.append(equality_id)
            if missing_equalities or not equality_ids:
                raise ValueError(
                    f"{label}: missing equality constraints {missing_equalities}"
                )

            auxiliary_dofs = []
            for joint_id in auxiliary_joint_ids:
                joint_type = int(self.mj_model.jnt_type[joint_id])
                dof_count = (
                    6
                    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE)
                    else 3
                    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL)
                    else 1
                )
                dof_start = int(self.mj_model.jnt_dofadr[joint_id])
                auxiliary_dofs.extend(range(dof_start, dof_start + dof_count))

            output_dofs = self.mj_model.jnt_dofadr[output_joint_ids]
            motor_dofs = self.mj_model.jnt_dofadr[motor_joint_ids]
            dependent_output_dofs = np.asarray(
                [dof for dof in output_dofs if dof not in set(motor_dofs)],
                dtype=np.int64,
            )

            self.closed_loop_transmissions.append(
                {
                    "name": label,
                    "output_positions": np.asarray(
                        [body_name_to_index[name] for name in output_names],
                        dtype=np.int64,
                    ),
                    "motor_positions": np.asarray(
                        [actuator_name_to_index[name] for name in motor_names],
                        dtype=np.int64,
                    ),
                    "output_qpos": self.mj_model.jnt_qposadr[output_joint_ids],
                    "output_dofs": output_dofs,
                    "dependent_output_dofs": dependent_output_dofs,
                    "motor_qpos": self.mj_model.jnt_qposadr[motor_joint_ids],
                    "motor_dofs": motor_dofs,
                    "auxiliary_dofs": np.asarray(auxiliary_dofs, dtype=np.int64),
                    "equality_ids": set(equality_ids),
                }
            )

    def _constraint_jacobian_block(
        self, row_indices: list[int], dof_indices: np.ndarray, *, dense=None
    ) -> np.ndarray:
        """Extract solver values natively, including during same-time pose projection."""
        if dense is None:
            dense = self._dense_constraint_jacobian()
        return dense[np.ix_(row_indices, dof_indices)]

    def _dense_constraint_jacobian(self):
        d = self.mj_data
        if mujoco.mj_isSparse(self.mj_model):
            dense = np.empty((d.nefc, self.mj_model.nv))
            mujoco.mju_sparse2dense(dense, d.efc_J, d.efc_J_rownnz,
                                   d.efc_J_rowadr, d.efc_J_colind)
        else:
            dense = d.efc_J.reshape(d.nefc, self.mj_model.nv)
        return dense

    def _closed_loop_output_to_motor_jacobian(self, group: dict) -> np.ndarray:
        if self._closed_loop_jacobian_cache_time != self.mj_data.time:
            self._closed_loop_jacobian_cache_time = self.mj_data.time
            self._closed_loop_jacobian_cache = {}
            self._closed_loop_constraint_jacobian = self._dense_constraint_jacobian()
            # Equality row membership usually stays fixed while values change.
            # Check its exact layout once per step, then reuse group row indices.
            rows = np.flatnonzero(self.mj_data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
            ids = self.mj_data.efc_id[rows]
            layout = (rows.tobytes(), ids.tobytes())
            if layout != getattr(self, '_closed_loop_equality_layout', None):
                self._closed_loop_equality_layout = layout
                for transmission in self.closed_loop_transmissions:
                    transmission['equality_rows'] = rows[np.isin(ids, list(transmission['equality_ids']))]
        cached = self._closed_loop_jacobian_cache.get(group["name"])
        if cached is not None:
            return cached

        rows = group['equality_rows']
        dependent_dofs = np.concatenate(
            (group["dependent_output_dofs"], group["auxiliary_dofs"])
        )
        dense = self._closed_loop_constraint_jacobian
        dependent_block = self._constraint_jacobian_block(rows, dependent_dofs, dense=dense)
        motor_block = self._constraint_jacobian_block(rows, group["motor_dofs"], dense=dense)
        dependent_velocity = -np.linalg.pinv(dependent_block) @ motor_block
        output_count = len(group["output_dofs"])
        jacobian = np.zeros(
            (output_count, len(group["motor_dofs"])), dtype=float
        )
        motor_dof_to_column = {
            int(dof): column for column, dof in enumerate(group["motor_dofs"])
        }
        dependent_dof_to_row = {
            int(dof): row
            for row, dof in enumerate(group["dependent_output_dofs"])
        }
        for output_row, output_dof in enumerate(group["output_dofs"]):
            output_dof = int(output_dof)
            motor_column = motor_dof_to_column.get(output_dof)
            if motor_column is not None:
                jacobian[output_row, motor_column] = 1.0
            else:
                jacobian[output_row] = dependent_velocity[
                    dependent_dof_to_row[output_dof]
                ]

        residual = np.linalg.norm(
            dependent_block @ dependent_velocity + motor_block
        )
        if residual > 1e-7 or np.linalg.matrix_rank(jacobian) < output_count:
            raise ValueError(
                f"{group['name']}: invalid closed-loop transmission Jacobian; "
                f"residual={residual:.3g}, jacobian={jacobian}"
            )
        self._closed_loop_jacobian_cache[group["name"]] = jacobian
        return jacobian

    def _map_body_torques_to_actuators(self, body_torques: np.ndarray) -> np.ndarray:
        motor_torques = body_torques.copy()
        for group in self.closed_loop_transmissions:
            jacobian = self._closed_loop_output_to_motor_jacobian(group)
            motor_torques[group["motor_positions"]] = (
                jacobian.T @ body_torques[group["output_positions"]]
            )
        return motor_torques

    def _map_actuator_torques_to_body(self, motor_torques: np.ndarray) -> np.ndarray:
        body_torques = motor_torques.copy()
        for group in self.closed_loop_transmissions:
            jacobian = self._closed_loop_output_to_motor_jacobian(group)
            effort = motor_torques[group["motor_positions"]]
            # Config requires square mappings and the Jacobian builder checks rank.
            # This inverse reports tau_est only; the applied forward torque map is unchanged.
            equivalent = (effort / jacobian[0, 0] if jacobian.shape == (1, 1)
                          else np.linalg.solve(jacobian.T, effort))
            body_torques[group["output_positions"]] = equivalent
        return body_torques

    def _h2_local_config_values_to_sim_order(self, values, key: str) -> np.ndarray:
        """Convert H2 local/training-order config arrays to actual MuJoCo joint order."""
        array = np.asarray(values, dtype=float)
        if "h2" not in self.config["ROBOT_TYPE"] or array.shape != (self.num_body_dof,):
            return array.copy()
        if len(self.body_joint_names) != self.num_body_dof:
            raise ValueError(
                f"Cannot reorder {key}: body_joint_names has "
                f"{len(self.body_joint_names)} entries, expected {self.num_body_dof}."
            )

        name_to_local = {
            name: index for index, name in enumerate(H2_JOINT_NAMES_LOCAL_MUJOCO)
        }
        missing = [
            name for name in self.body_joint_logical_names if name not in name_to_local
        ]
        if missing:
            raise ValueError(
                f"Cannot reorder H2 config array {key}; MuJoCo joints not in "
                f"local H2 order: {missing}"
            )
        return np.asarray(
            [array[name_to_local[name]] for name in self.body_joint_logical_names]
        )

    def _configure_actuator_maps_and_limits(self):
        """Build explicit joint->actuator maps and actuator-ordered torque limits."""
        self.body_actuator_index = self._build_actuator_index_for_joints(
            self.body_actuator_joint_index, "body"
        )
        self.left_hand_actuator_index = self._build_actuator_index_for_joints(
            self.left_hand_index, "left hand"
        )
        self.right_hand_actuator_index = self._build_actuator_index_for_joints(
            self.right_hand_index, "right hand"
        )

        self.torques = np.zeros(self.mj_model.nu, dtype=float)
        self.torque_limit = np.full(self.mj_model.nu, np.inf, dtype=float)

        limits = np.asarray(self.configured_torque_limit, dtype=float)
        required = self.num_body_dof + self.num_hand_dof * 2
        if limits.size < required:
            raise ValueError(
                "motor_effort_limit_list is shorter than controlled motors: "
                f"{limits.size} vs {required}"
            )

        limit_order = self.config.get("MOTOR_EFFORT_LIMIT_ORDER", "controlled_joint")
        if limit_order == "actuator":
            if limits.size != self.mj_model.nu:
                raise ValueError(
                    "Actuator-ordered motor_effort_limit_list must match MuJoCo nu: "
                    f"{limits.size} vs {self.mj_model.nu}"
                )
            self.torque_limit[:] = limits
            self.body_torque_limit = self.torque_limit[
                self.body_actuator_index
            ].copy()
        elif limit_order == "controlled_joint":
            self.body_torque_limit = self._h2_local_config_values_to_sim_order(
                limits[: self.num_body_dof], "motor_effort_limit_list"
            )
            self.torque_limit[self.body_actuator_index] = self.body_torque_limit

            if self.num_hand_dof > 0:
                hand_start = self.num_body_dof
                left_limits = limits[hand_start : hand_start + self.num_hand_dof]
                right_limits = limits[
                    hand_start
                    + self.num_hand_dof : hand_start
                    + self.num_hand_dof * 2
                ]
                self.torque_limit[self.left_hand_actuator_index] = left_limits
                self.torque_limit[self.right_hand_actuator_index] = right_limits
        else:
            raise ValueError(
                "MOTOR_EFFORT_LIMIT_ORDER must be 'actuator' or "
                f"'controlled_joint', got {limit_order!r}"
            )

        if self.config.get("PRINT_SCENE_INFORMATION", False):
            actuator_names = [
                self.mj_model.actuator(int(i)).name for i in self.body_actuator_index
            ]
            print(
                "[MuJoCo] Body joint actuator map: "
                f"{len(self.body_actuator_index)} joints -> {actuator_names}"
            )

    def _apply_mujoco_runtime_options(self):
        """Apply optional MuJoCo solver/integrator knobs from the runtime config."""
        integrator = self.config.get("MUJOCO_INTEGRATOR")
        if integrator:
            integrator_map = {
                "Euler": mujoco.mjtIntegrator.mjINT_EULER,
                "euler": mujoco.mjtIntegrator.mjINT_EULER,
                "RK4": mujoco.mjtIntegrator.mjINT_RK4,
                "rk4": mujoco.mjtIntegrator.mjINT_RK4,
                "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
                "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
            }
            if integrator not in integrator_map:
                raise ValueError(f"Unsupported MUJOCO_INTEGRATOR: {integrator}")
            self.mj_model.opt.integrator = integrator_map[integrator]

        solver = self.config.get("MUJOCO_SOLVER")
        if solver:
            solver_map = {
                "PGS": mujoco.mjtSolver.mjSOL_PGS,
                "pgs": mujoco.mjtSolver.mjSOL_PGS,
                "CG": mujoco.mjtSolver.mjSOL_CG,
                "cg": mujoco.mjtSolver.mjSOL_CG,
                "Newton": mujoco.mjtSolver.mjSOL_NEWTON,
                "newton": mujoco.mjtSolver.mjSOL_NEWTON,
            }
            if solver not in solver_map:
                raise ValueError(f"Unsupported MUJOCO_SOLVER: {solver}")
            self.mj_model.opt.solver = solver_map[solver]

        if "MUJOCO_ITERATIONS" in self.config:
            self.mj_model.opt.iterations = int(self.config["MUJOCO_ITERATIONS"])
        if "MUJOCO_LS_ITERATIONS" in self.config:
            self.mj_model.opt.ls_iterations = int(self.config["MUJOCO_LS_ITERATIONS"])

        floor_friction = self.config.get("MUJOCO_FLOOR_FRICTION")
        if floor_friction is not None:
            friction = np.asarray(floor_friction, dtype=float)
            if friction.shape != (3,) or not np.isfinite(friction).all() or np.any(friction < 0.0):
                raise ValueError(
                    "MUJOCO_FLOOR_FRICTION must contain three finite, non-negative values"
                )
            floor_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
            )
            if floor_id < 0:
                raise ValueError(
                    "MUJOCO_FLOOR_FRICTION is configured but the scene has no 'floor' geom"
                )
            self.mj_model.geom_friction[floor_id] = friction
            if self.config.get("PRINT_SCENE_INFORMATION", False):
                print(f"[MuJoCo] Floor friction override: {friction.tolist()}")

    def _override_mujoco_joint_force_limits(self):
        """Align MuJoCo joint force clamps with the runtime motor effort config."""
        motor_effort_limits = self.body_torque_limit
        if motor_effort_limits.shape != (len(self.body_joint_index),):
            raise ValueError(
                "MOTOR_EFFORT_LIMIT_LIST length must match body joints when "
                "OVERRIDE_MUJOCO_JOINT_FORCE_LIMITS is enabled: "
                f"{motor_effort_limits.size} vs {len(self.body_joint_index)}"
            )

        for joint_id, actuator_id, limit in zip(
            self.body_actuator_joint_index,
            self.body_actuator_index,
            motor_effort_limits,
        ):
            self.mj_model.jnt_actfrclimited[joint_id] = True
            self.mj_model.jnt_actfrcrange[joint_id] = np.array([-limit, limit], dtype=float)
            self.mj_model.actuator_ctrllimited[actuator_id] = True
            self.mj_model.actuator_ctrlrange[actuator_id] = np.array([-limit, limit], dtype=float)

        if self.config.get("PRINT_SCENE_INFORMATION", False):
            print(
                "[MuJoCo] Overrode joint actuator force ranges from "
                "motor_effort_limit_list for "
                f"{len(self.body_joint_index)} body joints."
            )

    def _expand_dof_property(self, key: str, default: float) -> np.ndarray:
        value = self.config.get(key, default)
        if isinstance(value, (int, float)):
            return np.full(len(self.body_joint_index), float(value), dtype=float)
        array = np.array(value, dtype=float)
        if array.shape != (len(self.body_joint_index),):
            raise ValueError(
                f"{key} must be a scalar or contain {len(self.body_joint_index)} values, "
                f"got shape {array.shape}"
            )
        return self._h2_local_config_values_to_sim_order(array, key)

    def _override_mujoco_dof_passive(self):
        """Patch passive joint parameters in-memory for converted robot MJCFs."""
        armature = self._expand_dof_property("MUJOCO_DOF_ARMATURE", 0.0)
        damping = self._expand_dof_property("MUJOCO_DOF_DAMPING", 0.0)
        frictionloss = self._expand_dof_property("MUJOCO_DOF_FRICTIONLOSS", 0.0)

        dof_indices = self.mj_model.jnt_dofadr[self.body_joint_index]
        self.mj_model.dof_armature[dof_indices] = armature
        self.mj_model.dof_damping[dof_indices] = damping
        self.mj_model.dof_frictionloss[dof_indices] = frictionloss

        if self.config.get("PRINT_SCENE_INFORMATION", False):
            print(
                "[MuJoCo] Overrode passive dof params for "
                f"{len(dof_indices)} body joints."
            )

    def _project_closed_loop_default_pose(self, default_angles: np.ndarray) -> None:
        """Solve equality constraints while placing output joints at defaults."""
        if not self.closed_loop_transmissions:
            return

        max_iterations = int(
            self.config.get("MUJOCO_CLOSED_LOOP_INIT_MAX_ITERATIONS", 80)
        )
        tolerance = float(
            self.config.get("MUJOCO_CLOSED_LOOP_INIT_TOLERANCE", 1e-10)
        )
        equality_type = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)

        for group in self.closed_loop_transmissions:
            target = default_angles[group["output_positions"]]
            columns = np.concatenate(
                (
                    group["motor_dofs"],
                    group["dependent_output_dofs"],
                    group["auxiliary_dofs"],
                )
            )
            for _ in range(max_iterations):
                mujoco.mj_forward(self.mj_model, self.mj_data)
                rows = [
                    row_index
                    for row_index in range(self.mj_data.nefc)
                    if int(self.mj_data.efc_type[row_index]) == equality_type
                    and int(self.mj_data.efc_id[row_index])
                    in group["equality_ids"]
                ]
                constraint_error = np.asarray(self.mj_data.efc_pos[rows])
                output_error = target - self.mj_data.qpos[group["output_qpos"]]
                if max(
                    np.linalg.norm(constraint_error), np.linalg.norm(output_error)
                ) < tolerance:
                    break

                constraint_block = self._constraint_jacobian_block(rows, columns)
                output_block = np.zeros(
                    (len(group["output_dofs"]), len(columns)), dtype=float
                )
                column_positions = {
                    int(dof_index): column
                    for column, dof_index in enumerate(columns)
                }
                for output_row, dof_index in enumerate(group["output_dofs"]):
                    output_block[output_row, column_positions[int(dof_index)]] = 1.0

                delta = np.linalg.lstsq(
                    np.vstack((constraint_block, output_block)),
                    np.concatenate((-constraint_error, output_error)),
                    rcond=None,
                )[0]
                qvel_delta = np.zeros(self.mj_model.nv, dtype=float)
                qvel_delta[columns] = 0.7 * delta
                mujoco.mj_integratePos(
                    self.mj_model, self.mj_data.qpos, qvel_delta, 1.0
                )
                mujoco.mj_normalizeQuat(self.mj_model, self.mj_data.qpos)
            else:
                raise ValueError(
                    f"{group['name']}: default-pose projection did not converge"
                )

        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _apply_default_pose(self):
        if self.use_floating_root_link:
            base_pos = np.array(
                self.config.get("DEFAULT_BASE_POS", self.mj_data.qpos[:3]), dtype=float
            )
            base_z_override = os.environ.get("GEAR_SONIC_MUJOCO_DEFAULT_BASE_Z")
            if base_z_override:
                base_pos[2] = float(base_z_override)
            if base_pos.shape != (3,):
                raise ValueError(f"DEFAULT_BASE_POS must contain 3 values, got {base_pos}")
            self.mj_data.qpos[:3] = base_pos

            base_quat = np.array(
                self.config.get("DEFAULT_BASE_QUAT", [1.0, 0.0, 0.0, 0.0]), dtype=float
            )
            if base_quat.shape != (4,):
                raise ValueError(f"DEFAULT_BASE_QUAT must contain 4 values, got {base_quat}")
            quat_norm = np.linalg.norm(base_quat)
            if quat_norm == 0:
                raise ValueError("DEFAULT_BASE_QUAT must be non-zero")
            self.mj_data.qpos[3:7] = base_quat / quat_norm

        default_angles = np.array(self.config.get("DEFAULT_DOF_ANGLES", []), dtype=float)
        if default_angles.size:
            if default_angles.shape != (len(self.body_joint_index),):
                raise ValueError(
                    "DEFAULT_DOF_ANGLES length must match body joints: "
                    f"{default_angles.size} vs {len(self.body_joint_index)}"
                )
            default_angles = self._h2_local_config_values_to_sim_order(
                default_angles, "DEFAULT_DOF_ANGLES"
            )
            closed_output_positions = {
                int(position)
                for group in self.closed_loop_transmissions
                for position in group["output_positions"]
            }
            direct_positions = np.asarray(
                [
                    position
                    for position in range(len(self.body_joint_index))
                    if position not in closed_output_positions
                ],
                dtype=np.int64,
            )
            self.mj_data.qpos[self.body_qpos_index[direct_positions]] = default_angles[
                direct_positions
            ]
            mujoco.mj_forward(self.mj_model, self.mj_data)
            self._project_closed_loop_default_pose(default_angles)

        self.mj_data.qvel[:] = 0
        self.mj_data.ctrl[:] = 0
        self.mj_data.xfrc_applied[:] = 0
        mujoco.mj_forward(self.mj_model, self.mj_data)

        if self.config.get("CALIBRATE_BASE_Z_TO_H2_SOLE_CONTACT", False):
            self._calibrate_base_z_to_h2_sole_contact()

    def _geom_bottom_z(self, geom_id: int) -> float:
        geom_type = self.mj_model.geom_type[geom_id]
        center_z = float(self.mj_data.geom_xpos[geom_id, 2])
        size = self.mj_model.geom_size[geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return center_z - float(size[0])
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            rotation = self.mj_data.geom_xmat[geom_id].reshape(3, 3)
            z_half_extent = float(np.sum(np.abs(rotation[2, :]) * size[:3]))
            return center_z - z_half_extent
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(self.mj_model.geom_dataid[geom_id])
            vertex_start = int(self.mj_model.mesh_vertadr[mesh_id])
            vertex_count = int(self.mj_model.mesh_vertnum[mesh_id])
            vertices = self.mj_model.mesh_vert[
                vertex_start : vertex_start + vertex_count
            ]
            # mesh_vert is already expressed in MuJoCo's compiled mesh frame.
            # geom_xmat/geom_xpos include the mesh asset reference transform.
            geom_rotation = self.mj_data.geom_xmat[geom_id].reshape(3, 3)
            world_z = (
                vertices @ geom_rotation[2, :] + self.mj_data.geom_xpos[geom_id, 2]
            )
            return float(np.min(world_z))
        return center_z

    def _calibrate_base_z_to_h2_sole_contact(self, *, zero_velocity: bool = True) -> None:
        sole_geom_ids = [
            geom_id
            for geom_id in range(self.mj_model.ngeom)
            if (self.mj_model.geom(geom_id).name or "").startswith("h2lora_")
        ]
        if not sole_geom_ids:
            ankle_pitch_body_ids = {
                int(mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name))
                for name in ("left_ankle_pitch_link", "right_ankle_pitch_link")
            }
            ankle_pitch_body_ids.discard(-1)
            sole_geom_ids = [
                geom_id
                for geom_id in range(self.mj_model.ngeom)
                if int(self.mj_model.geom_bodyid[geom_id]) in ankle_pitch_body_ids
                and (
                    int(self.mj_model.geom_contype[geom_id]) != 0
                    or int(self.mj_model.geom_conaffinity[geom_id]) != 0
                )
                and int(self.mj_model.geom_type[geom_id])
                in {
                    int(mujoco.mjtGeom.mjGEOM_SPHERE),
                    int(mujoco.mjtGeom.mjGEOM_BOX),
                    int(mujoco.mjtGeom.mjGEOM_MESH),
                }
            ]
        if not sole_geom_ids:
            raise ValueError(
                "CALIBRATE_BASE_Z_TO_H2_SOLE_CONTACT requires H2 sole collision geoms "
                "on ankle_pitch bodies. Enable ADD_H2_OFFICIAL_FOOT_CONTACT_GEOMS or "
                "use an official Unitree H2 MJCF with sole contact geoms."
            )

        target_clearance = float(self.config.get("H2_SOLE_CONTACT_CLEARANCE", 0.002))
        min_bottom = min(self._geom_bottom_z(geom_id) for geom_id in sole_geom_ids)
        self.mj_data.qpos[2] += target_clearance - min_bottom
        if zero_velocity:
            self.mj_data.qvel[:] = 0
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def init_renderers(self):
        self.renderers = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = mujoco.Renderer(
                self.mj_model, height=camera_config["height"], width=camera_config["width"]
            )
            self.renderers[camera_name] = renderer

    def compute_body_torques(self) -> np.ndarray:
        # PD control: tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
        body_torques = np.zeros(self.num_body_dof)
        diagnostics = getattr(self, "diagnostics", None)
        capture_commands = diagnostics is not None and diagnostics.step % diagnostics.stride == 0
        if capture_commands:
            self.diagnostic_body_commands = np.full((5, self.num_body_dof), np.nan)
        if (
            self.unitree_bridge is not None
            and self.unitree_bridge.low_cmd
            and getattr(
                self.unitree_bridge,
                "body_low_cmd_available",
                self.unitree_bridge.body_low_cmd_is_fresh,
            )()
        ):
            motor_to_sim = getattr(
                self.unitree_bridge,
                "motor_to_sim_joint_index",
                np.arange(self.unitree_bridge.num_body_motor, dtype=np.int64),
            )
            # One packet snapshot and array arithmetic preserve the scalar PD
            # expression while avoiding 31 repeated NumPy scalar operations.
            packet = self.unitree_bridge.low_cmd
            count = self.unitree_bridge.num_body_motor
            if count == 0:
                return body_torques
            commands = np.asarray([(c.q, c.dq, c.kp, c.kd, c.tau)
                                   for c in packet.motor_cmd[:count]], dtype=float).T
            q_des, dq_des, kp, kd, tau = commands
            if capture_commands:
                self.diagnostic_body_commands[:, motor_to_sim] = commands
            if self.unitree_bridge.use_sensor:
                q = self.mj_data.sensordata[:count]
                dq = self.mj_data.sensordata[count:2 * count]
            else:
                q = self.mj_data.qpos[self.body_qpos_index[motor_to_sim]]
                dq = self.mj_data.qvel[self.body_dof_index[motor_to_sim]]
            body_torques[motor_to_sim] = tau + kp * (q_des - q) + kd * (dq_des - dq)
        return body_torques

    def get_head_pose(self) -> np.ndarray:
        root_pos = self.mj_data.body("torso_link").xpos.copy()
        # Reorder quaternion from MuJoCo [w,x,y,z] to scipy [x,y,z,w]
        root_quat = self.mj_data.body("torso_link").xquat.copy()[[1, 2, 3, 0]]
        head_pos = root_pos + Rotation.from_quat(root_quat).apply(np.array([0.0, 0.0, -0.044]))
        return np.concatenate((head_pos, root_quat))

    def get_root_vel(self) -> np.ndarray:
        return self.mj_data.qvel[:6]

    def compute_hand_torques(self) -> np.ndarray:
        left_hand_torques = np.zeros(self.num_hand_dof)
        right_hand_torques = np.zeros(self.num_hand_dof)
        if (
            self.unitree_bridge is not None
            and self.unitree_bridge.low_cmd
            and getattr(
                self.unitree_bridge,
                "body_low_cmd_available",
                self.unitree_bridge.body_low_cmd_is_fresh,
            )()
        ):
            for i in range(self.unitree_bridge.num_hand_motor):
                left_hand_torques[i] = (
                    self.unitree_bridge.left_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.left_hand_qpos_index[i]]
                    )
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.left_hand_dof_index[i]]
                    )
                )
                right_hand_torques[i] = (
                    self.unitree_bridge.right_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.right_hand_qpos_index[i]]
                    )
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.right_hand_dof_index[i]]
                    )
                )
        return np.concatenate((left_hand_torques, right_hand_torques))

    def compute_body_qpos(self) -> np.ndarray:
        body_qpos = np.zeros(self.num_body_dof)
        if (
            self.unitree_bridge is not None
            and self.unitree_bridge.low_cmd
            and getattr(
                self.unitree_bridge,
                "body_low_cmd_available",
                self.unitree_bridge.body_low_cmd_is_fresh,
            )()
        ):
            motor_to_sim = getattr(
                self.unitree_bridge,
                "motor_to_sim_joint_index",
                np.arange(self.unitree_bridge.num_body_motor, dtype=np.int64),
            )
            for motor_i in range(self.unitree_bridge.num_body_motor):
                body_qpos[int(motor_to_sim[motor_i])] = self.unitree_bridge.low_cmd.motor_cmd[motor_i].q
        return body_qpos

    def compute_hand_qpos(self) -> np.ndarray:
        hand_qpos = np.zeros(self.num_hand_dof * 2)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                hand_qpos[i] = self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                hand_qpos[i + self.num_hand_dof] = self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
        return hand_qpos

    def _body_velocity_in_link_frame(self, body_id: int) -> np.ndarray:
        """Return body velocity as [linear, angular] in the body's link frame."""
        velocity_world = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.mj_model,
            self.mj_data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity_world,
            0,
        )
        world_to_link = self.mj_data.xmat[body_id].reshape(3, 3).T
        # mj_objectVelocity returns [angular, linear].  Its local BODY mode uses
        # the inertial principal-axis frame (ximat), not the link frame (xmat).
        return np.concatenate(
            (
                world_to_link @ velocity_world[3:6],
                world_to_link @ velocity_world[0:3],
            )
        )

    def prepare_obs(self) -> Dict[str, any]:
        obs = {}
        if self.use_floating_root_link:
            obs["floating_base_pose"] = self.mj_data.qpos[:7]
            obs["floating_base_vel"] = self.mj_data.qvel[:6]
            obs["floating_base_acc"] = self.mj_data.qacc[:6]
            # IsaacLab mdp.base_ang_vel observes root_ang_vel_b.
            obs["floating_base_vel_local"] = self._body_velocity_in_link_frame(
                self.root_body_id
            )
        else:
            obs["floating_base_pose"] = np.zeros(7)
            obs["floating_base_vel"] = np.zeros(6)
            obs["floating_base_acc"] = np.zeros(6)
            obs["floating_base_vel_local"] = np.zeros(6)

        obs["secondary_imu_quat"] = self.mj_data.xquat[self.torso_index]

        torso_link = self.mj_model.body("torso_link").id
        obs["secondary_imu_vel"] = self._body_velocity_in_link_frame(torso_link)

        obs["body_q"] = self.mj_data.qpos[self.body_qpos_index]
        obs["body_dq"] = self.mj_data.qvel[self.body_dof_index]
        obs["body_ddq"] = self.mj_data.qacc[self.body_dof_index]
        obs["body_tau_est"] = self._map_actuator_torques_to_body(
            self.mj_data.actuator_force[self.body_actuator_index]
        )
        if self.num_hand_dof > 0:
            obs["left_hand_q"] = self.mj_data.qpos[self.left_hand_qpos_index]
            obs["left_hand_dq"] = self.mj_data.qvel[self.left_hand_dof_index]
            obs["left_hand_ddq"] = self.mj_data.qacc[self.left_hand_dof_index]
            obs["left_hand_tau_est"] = self.mj_data.actuator_force[self.left_hand_actuator_index]
            obs["right_hand_q"] = self.mj_data.qpos[self.right_hand_qpos_index]
            obs["right_hand_dq"] = self.mj_data.qvel[self.right_hand_dof_index]
            obs["right_hand_ddq"] = self.mj_data.qacc[self.right_hand_dof_index]
            obs["right_hand_tau_est"] = self.mj_data.actuator_force[
                self.right_hand_actuator_index
            ]
        obs["time"] = self.mj_data.time
        return obs

    def sim_step(self):
        self.obs = self.prepare_obs()
        self.unitree_bridge.PublishLowState(self.obs)
        if self.unitree_bridge.joystick:
            self.unitree_bridge.PublishWirelessController()

        if self.elastic_band:
            if self.elastic_band.enable and self.use_floating_root_link:
                pose = np.concatenate(
                    [
                        self.mj_data.xpos[self.band_attached_link],
                        self.mj_data.xquat[self.band_attached_link],
                        np.zeros(6),
                    ]
                )
                mujoco.mj_objectVelocity(
                    self.mj_model,
                    self.mj_data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.band_attached_link,
                    pose[7:13],
                    0,
                )
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
                elastic_force = self.elastic_band.Advance(pose)
                self.mj_data.xfrc_applied[self.band_attached_link] = elastic_force
            else:
                self.mj_data.xfrc_applied[self.band_attached_link] = np.zeros(6)
        body_pd_requested = self.compute_body_torques()
        body_torques_raw = self._map_body_torques_to_actuators(body_pd_requested)
        body_torques_response = body_torques_raw
        if self.motor_response is not None:
            body_torques_response = self.motor_response.apply(body_torques_raw, self.sim_dt)
        hand_torques = self.compute_hand_torques()
        self.torques.fill(0.0)
        self.torques[self.body_actuator_index] = body_torques_response
        if self.num_hand_dof > 0:
            self.torques[self.left_hand_actuator_index] = hand_torques[: self.num_hand_dof]
            self.torques[self.right_hand_actuator_index] = hand_torques[self.num_hand_dof :]

        self.torques = np.clip(self.torques, -self.torque_limit, self.torque_limit)
        if self.config["FREE_BASE"]:
            # Prepend 6 zeros for the floating-base root DOF actuators
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else:
            self.mj_data.ctrl = self.torques
        if self.ankle_response is not None:
            self.ankle_response.apply(self.mj_model, self.mj_data)
        if self.diagnostics is not None:
            self.diagnostics.begin(body_pd_requested, body_torques_raw, body_torques_response)
        mujoco.mj_step(self.mj_model, self.mj_data)
        if self.diagnostics is not None:
            self.diagnostics.end()

        self.check_fall()

    def apply_perturbation(self, key):
        perturbation_x_body = 0.0
        perturbation_y_body = 0.0
        if key == "up":
            perturbation_x_body = 1.0
        elif key == "down":
            perturbation_x_body = -1.0
        elif key == "left":
            perturbation_y_body = 1.0
        elif key == "right":
            perturbation_y_body = -1.0

        vel_body = np.array([perturbation_x_body, perturbation_y_body, 0.0])
        vel_world = np.zeros(3)
        base_quat = self.mj_data.qpos[3:7]
        mujoco.mju_rotVecQuat(vel_world, vel_body, base_quat)

        self.mj_data.qvel[0] += vel_world[0]
        self.mj_data.qvel[1] += vel_world[1]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def update_viewer(self):
        if self.viewer is not None:
            self.viewer.sync(state_only=self.config.get("VIEWER_STATE_ONLY", False))

    def update_viewer_camera(self):
        if self.viewer is not None:
            if self.viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    def update_reward(self):
        with self.reward_lock:
            self.last_reward = 0

    def get_reward(self):
        with self.reward_lock:
            return self.last_reward

    def set_unitree_bridge(self, unitree_bridge):
        self.unitree_bridge = unitree_bridge

    def get_privileged_obs(self):
        return {}

    def update_render_caches(self):
        render_caches = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = self.renderers[camera_name]
            if "params" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["params"])
            elif "mjcf_name" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["mjcf_name"])
            else:
                renderer.update_scene(self.mj_data, camera=camera_name)
            render_caches[camera_name + "_image"] = renderer.render()

        if self.image_publish_process is not None:
            self.image_publish_process.update_shared_memory(render_caches)

        return render_caches

    def handle_keyboard_button(self, key):
        if self.elastic_band:
            self.elastic_band.handle_keyboard_button(key)

        if key == "backspace":
            self.reset()
        if key == "v":
            self.update_viewer_camera()
        if key in ["up", "down", "left", "right"]:
            self.apply_perturbation(key)

    def check_fall(self):
        self.fall = False
        if self.mj_data.qpos[2] < 0.2:
            self.fall = True
            print(f"Warning: Robot has fallen, height: {self.mj_data.qpos[2]:.3f} m")

        if self.fall:
            self.reset()

    def check_self_collision(self):
        robot_bodies = get_subtree_body_names(self.mj_model, self.mj_model.body(self.root_body).id)
        self_collision, contact_bodies = check_contact(
            self.mj_model, self.mj_data, robot_bodies, robot_bodies, return_all_contact_bodies=True
        )
        if self_collision:
            print(f"Warning: Self-collision detected: {contact_bodies}")
        return self_collision

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self._closed_loop_jacobian_cache_time = None
        if self.ankle_response is not None:
            self.ankle_response.reset()
        if getattr(self, "motor_response", None) is not None:
            self.motor_response.reset()
        if self.config.get("APPLY_DEFAULT_POSE_ON_RESET", False):
            self._apply_default_pose()
        else:
            mujoco.mj_forward(self.mj_model, self.mj_data)
        if getattr(self, "diagnostics", None) is not None:
            self.diagnostics.reset()


class BaseSimulator:
    """Base simulator class that handles initialization and running of simulations"""

    def __init__(
        self, config: Dict[str, any], env_name: str = "default", redis_client=None, **kwargs
    ):
        self.config = config
        self.env_name = env_name
        self.redis_client = redis_client
        if self.redis_client is not None:
            self.redis_client.set("push_left_hand", "false")
            self.redis_client.set("push_right_hand", "false")
            self.redis_client.set("push_torso", "false")

        # Create rate objects
        self.sim_dt = self.config["SIMULATE_DT"]
        self.reward_dt = self.config.get("REWARD_DT", 0.02)
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.viewer_dt = self.config.get("VIEWER_DT", 0.02)
        self._running = True

        self.robot = Robot(self.config)

        # Create the environment
        if env_name == "default":
            self.sim_env = DefaultEnv(config, env_name, **kwargs)
        else:
            raise ValueError(
                f"Invalid environment name: {env_name}. "
                f"Only 'default' is supported in this minimal build."
            )

        try:
            if self.config.get("INTERFACE", None):
                ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.config["DOMAIN_ID"])
        except Exception as e:
            print(f"Note: Channel factory initialization attempt: {e}")

        self.init_unitree_bridge()
        self.sim_env.set_unitree_bridge(self.unitree_bridge)

        self.init_subscriber()
        self.init_publisher()

        self.sim_thread = None

    def start_as_thread(self):
        self.sim_thread = Thread(target=self.start)
        self.sim_thread.start()

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        self.sim_env.start_image_publish_subprocess(start_method, camera_port)

    def init_subscriber(self):
        pass

    def init_publisher(self):
        pass

    def init_unitree_bridge(self):
        self.unitree_bridge = UnitreeSdk2Bridge(self.config)
        if self.config["USE_JOYSTICK"]:
            self.unitree_bridge.SetupJoystick(
                device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
            )

    def start(self):
        """Run fixed physics steps against cumulative wall-clock deadlines."""
        sim_cnt = 0
        ts = time.monotonic()
        pacer = None
        work_sum = viewer_sum = other_sum = 0.0
        last_report = last_print = 0.0

        def timing():
            values = pacer.snapshot()
            count = max(1, pacer.steps)
            values.update(step_work_ms_mean=work_sum * 1000 / count,
                          viewer_ms_mean=viewer_sum * 1000 / count,
                          other_ms_mean=other_sum * 1000 / count)
            if self.sim_env.diagnostics is not None:
                self.sim_env.diagnostics.record_timing(values)
            return values

        try:
            # Exclude first viewer/render initialization from the pacing epoch.
            if self.sim_env.viewer is not None:
                self.sim_env.viewer.sync()
                # Passive rendering is asynchronous; let its first frames upload
                # assets before starting the externally clocked physics loop.
                time.sleep(0.25)
            # No physics step or publication: initialize lazy linear algebra and
            # DDS codecs before charging work to the realtime clock.
            self.sim_env.prepare_obs()
            self.unitree_bridge.WarmupSerialization()
            self.sim_env.update_viewer()
            self.sim_env.update_render_caches()
            pacer = RealtimePacer(self.sim_dt, self.config.get("REALTIME_MAX_LAG_MS", 20) / 1000)
            print(f"[Realtime] Absolute deadlines; physics dt={self.sim_dt * 1000:g} ms; "
                  f"maximum lag={pacer.max_lag_s * 1000:g} ms (overrun stops simulation)", flush=True)
            while self._running and (
                (self.sim_env.viewer and self.sim_env.viewer.is_running())
                or (self.sim_env.viewer is None)
            ):
                step_start = time.monotonic()

                self.sim_env.sim_step()
                step_end = time.monotonic()
                work_sum += step_end - step_start
                now = step_end
                if now - ts > 1 / 10.0 and self.redis_client is not None:
                    head_pose = self.sim_env.get_head_pose()
                    self.redis_client.set("head_pos", pickle.dumps(head_pose[:3]))
                    self.redis_client.set("head_quat", pickle.dumps(head_pose[3:]))
                    ts = now

                viewer_start = time.monotonic()
                if sim_cnt % max(1, round(self.viewer_dt / self.sim_dt)) == 0:
                    self.sim_env.update_viewer()
                viewer_end = time.monotonic()
                viewer_sum += viewer_end - viewer_start

                if sim_cnt % max(1, round(self.reward_dt / self.sim_dt)) == 0:
                    self.sim_env.update_reward()

                if sim_cnt % max(1, round(self.image_dt / self.sim_dt)) == 0:
                    self.sim_env.update_render_caches()

                other_sum += (viewer_start - step_end) + (time.monotonic() - viewer_end)
                pacer.finish_step()
                elapsed = pacer.now - pacer.start
                if elapsed - last_report >= 1.0:
                    values = timing()
                    last_report = elapsed
                    if elapsed - last_print >= 5.0:
                        print(f"[Realtime] RTF={values['realtime_factor']:.4f}, "
                              f"lag={values['lag_ms']:.2f} ms, "
                              f"max lag={values['max_lag_ms']:.2f} ms", flush=True)
                        last_print = elapsed

                sim_cnt += 1
        except KeyboardInterrupt:
            print("Simulator interrupted by user.")
        except Exception as exc:
            if pacer is not None and pacer.failure is None:
                pacer.failure = f"simulation exception: {exc}"
            raise
        finally:
            if pacer is not None:
                values = timing()
                diagnostics = self.sim_env.diagnostics
                if diagnostics is not None:
                    diagnostics.log.metadata['realtime'] = dict(
                        scheduler="absolute_deadline_v1", max_allowed_lag_ms=pacer.max_lag_s * 1000,
                        valid=pacer.failure is None, failure=pacer.failure, **values)
                    if pacer.failure:
                        diagnostics.close(incomplete=True)
                        self.sim_env.diagnostics = None
            self.close()

    def __del__(self):
        self.close()

    def reset(self):
        self.sim_env.reset()

    def close(self):
        self._running = False
        try:
            if getattr(self.sim_env, "diagnostics", None) is not None:
                self.sim_env.diagnostics.close()
                self.sim_env.diagnostics = None
            if self.sim_env.image_publish_process is not None:
                self.sim_env.image_publish_process.stop()
            if self.sim_env.viewer is not None:
                self.sim_env.viewer.close()
                self.sim_env.viewer = None
        except Exception as e:
            print(f"Warning during close: {e}")

    def get_privileged_obs(self):
        return self.sim_env.get_privileged_obs()

    def handle_keyboard_button(self, key):
        self.sim_env.handle_keyboard_button(key)
