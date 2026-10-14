from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import (
    apply_local_pitch_bias_wxyz,
)
from scripts.build_h2_mujoco_from_unitree_ros import (
    add_floating_base,
    collect_compound_inertials,
    validate_compound_inertials,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = (
    REPO_ROOT
    / "gear_sonic/data/assets/robot_description/unitree_ros/h2_description"
)
UNITREE_MUJOCO_DIR = (
    REPO_ROOT / "gear_sonic/data/assets/robot_description/unitree_mujoco"
)
CONFIG_PATH = (
    REPO_ROOT
    / "gear_sonic/utils/mujoco_sim/wbc_configs/h2_lora_sonic_unitree_ros.yaml"
)
IMU_BIAS_CONFIG_PATH = (
    REPO_ROOT
    / "gear_sonic/utils/mujoco_sim/wbc_configs/h2_lora_sonic_unitree_ros_imu_pitch_bias.yaml"
)


class H2UnitreeRosMujocoTest(unittest.TestCase):
    def _make_ros_env(self):
        with CONFIG_PATH.open() as config_file:
            config = yaml.safe_load(config_file)
        config["PRINT_SCENE_INFORMATION"] = False
        return DefaultEnv(config, onscreen=False)

    def test_generated_inertials_match_urdf_fixed_link_composition(self):
        urdf_root = ET.parse(ASSET_DIR / "H2.urdf").getroot()
        add_floating_base(urdf_root)
        model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "h2_mujoco.xml"))
        body_names = {model.body(i).name for i in range(1, model.nbody)}
        compounds = collect_compound_inertials(urdf_root, body_names)

        validate_compound_inertials(model, compounds)
        self.assertAlmostEqual(float(np.sum(model.body_mass)), 75.5874563, places=9)
        np.testing.assert_allclose(
            model.body_ipos[model.body("left_wrist_yaw_link").id],
            [0.080194972828, -0.004803505872, 0.003397670044],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            model.body_ipos[model.body("right_wrist_yaw_link").id],
            [0.080552592521, 0.004797897237, 0.003381652250],
            atol=1e-10,
        )

    def test_ros_mass_variants_use_correct_fixed_hand_inertials(self):
        aligned = mujoco.MjModel.from_xml_path(
            str(UNITREE_MUJOCO_DIR / "h2_ros_mass_aligned/h2_mujoco.xml")
        )
        debug = mujoco.MjModel.from_xml_path(
            str(UNITREE_MUJOCO_DIR / "h2_ros_mass_debug/h2_mujoco.xml")
        )
        for model in (aligned, debug):
            np.testing.assert_allclose(
                model.body_ipos[model.body("left_wrist_yaw_link").id],
                [0.080194972828, -0.004803505872, 0.003397670044],
                atol=1e-10,
            )
            np.testing.assert_allclose(
                model.body_ipos[model.body("right_wrist_yaw_link").id],
                [0.080552592521, 0.004797897237, 0.003381652250],
                atol=1e-10,
            )

        np.testing.assert_array_equal(aligned.body_mass, debug.body_mass)
        np.testing.assert_array_equal(aligned.body_ipos, debug.body_ipos)
        np.testing.assert_array_equal(aligned.body_inertia, debug.body_inertia)
        differing_quaternions = np.flatnonzero(
            np.max(np.abs(aligned.body_iquat - debug.body_iquat), axis=1) > 1e-12
        )
        np.testing.assert_array_equal(differing_quaternions, [aligned.body("pelvis").id])

    def test_mesh_sole_calibration_matches_mujoco_collision_frame(self):
        env = self._make_ros_env()

        ankle_body_ids = {
            env.mj_model.body("left_ankle_pitch_link").id,
            env.mj_model.body("right_ankle_pitch_link").id,
        }
        sole_geom_ids = [
            geom_id
            for geom_id in range(env.mj_model.ngeom)
            if int(env.mj_model.geom_bodyid[geom_id]) in ankle_body_ids
            and int(env.mj_model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
            and int(env.mj_model.geom_contype[geom_id]) != 0
        ]
        self.assertEqual(len(sole_geom_ids), 2)
        np.testing.assert_allclose(
            [env._geom_bottom_z(geom_id) for geom_id in sole_geom_ids],
            [0.002, 0.002],
            atol=1e-9,
        )

        env.mj_data.qpos[2] -= 0.003
        mujoco.mj_forward(env.mj_model, env.mj_data)
        floor_contacts = [
            contact
            for contact in env.mj_data.contact
            if "floor"
            in {
                env.mj_model.geom(contact.geom1).name,
                env.mj_model.geom(contact.geom2).name,
            }
        ]
        self.assertEqual(len(floor_contacts), 2)
        np.testing.assert_allclose(
            [contact.dist for contact in floor_contacts],
            [-0.001, -0.001],
            atol=1e-9,
        )

    def test_base_velocity_is_reported_in_link_not_inertial_frame(self):
        env = self._make_ros_env()
        env.mj_data.qpos[:7] = [
            0.0,
            0.0,
            1.0,
            0.965925826,
            0.258819045,
            0.0,
            0.0,
        ]
        expected_angular_velocity = np.array([0.4, 0.5, 0.6])
        env.mj_data.qvel[:6] = [0.1, 0.2, 0.3, *expected_angular_velocity]
        mujoco.mj_forward(env.mj_model, env.mj_data)

        obs = env.prepare_obs()
        np.testing.assert_allclose(
            obs["floating_base_vel_local"][3:6],
            expected_angular_velocity,
            atol=1e-8,
        )

        pelvis_id = env.mj_model.body("pelvis").id
        self.assertGreater(abs(float(env.mj_model.body_iquat[pelvis_id, 1])), 0.5)

    def test_imu_bias_profile_only_changes_reported_pelvis_pitch(self):
        with CONFIG_PATH.open() as config_file:
            baseline = yaml.safe_load(config_file)
        with IMU_BIAS_CONFIG_PATH.open() as config_file:
            biased = yaml.safe_load(config_file)

        self.assertEqual(biased.pop("PELVIS_IMU_PITCH_BIAS_DEG"), 2.7)
        self.assertEqual(biased, baseline)

        true_pitch_rad = np.deg2rad(-3.0)
        true_quat = np.array(
            [np.cos(true_pitch_rad / 2.0), 0.0, np.sin(true_pitch_rad / 2.0), 0.0]
        )
        reported_quat = apply_local_pitch_bias_wxyz(
            true_quat, np.deg2rad(2.7)
        )
        reported_pitch_rad = np.arcsin(
            2.0
            * (
                reported_quat[0] * reported_quat[2]
                - reported_quat[3] * reported_quat[1]
            )
        )
        self.assertAlmostEqual(np.rad2deg(reported_pitch_rad), -0.3, places=12)


if __name__ == "__main__":
    unittest.main()
