import hashlib
from pathlib import Path
import unittest

import mujoco
import numpy as np
import yaml

from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import (
    apply_local_pitch_bias_wxyz,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_PATH = (
    REPO_ROOT
    / "gear_sonic/data/assets/robot_description/unitree_ros/h2_description/H2_loop.xml"
)
BASE_CONFIG_PATH = (
    REPO_ROOT
    / "gear_sonic/utils/mujoco_sim/wbc_configs/h2_native_sonic_unitree_ros_loop.yaml"
)


class H2AnkleRollBacklashTest(unittest.TestCase):
    def _make_env(self) -> DefaultEnv:
        config = SimLoopConfig(
            wbc_version="h2_native_unitree_ros_loop_ankle_backlash",
            enable_onscreen=False,
        ).load_wbc_yaml()
        config["PRINT_SCENE_INFORMATION"] = False
        return DefaultEnv(config, onscreen=False)

    def test_official_asset_and_base_config_remain_unmodified(self):
        self.assertEqual(
            hashlib.sha256(ASSET_PATH.read_bytes()).hexdigest(),
            "399284179c4b7067b82cc5dc910b832936f35c0d7981dc8954897251f731e458",
        )
        with BASE_CONFIG_PATH.open() as config_file:
            base_config = yaml.safe_load(config_file)
        self.assertNotIn("H2_ANKLE_ROLL_BACKLASH_DEG", base_config)
        self.assertNotIn("H2_ANKLE_PITCH_BACKLASH_DEG", base_config)
        self.assertNotIn("H2_WAIST_BACKLASH_DEG", base_config)
        self.assertNotIn("PELVIS_IMU_PITCH_BIAS_DEG", base_config)

    def test_backlash_profile_adds_five_passive_limited_joints(self):
        env = self._make_env()
        self.assertEqual((env.mj_model.njnt, env.mj_model.nu, env.mj_model.neq), (49, 31, 6))
        self.assertEqual(len(env.body_joint_index), 31)

        expected_half_widths = {"left": 0.6, "right": 0.8}
        actuator_joint_ids = set(env.mj_model.actuator_trnid[:, 0].tolist())
        for side, half_width_deg in expected_half_widths.items():
            joint = env.mj_model.joint(f"{side}_ankle_roll_backlash_joint")
            np.testing.assert_allclose(
                env.mj_model.jnt_range[joint.id],
                np.deg2rad([-half_width_deg, half_width_deg]),
                atol=1e-12,
            )
            self.assertTrue(bool(env.mj_model.jnt_limited[joint.id]))
            self.assertNotIn(joint.id, actuator_joint_ids)
            self.assertEqual(
                env.mj_model.body(f"{side}_ankle_roll_link").parentid,
                env.mj_model.body(f"{side}_ankle_roll_motor_carrier").id,
            )

            motor_joint = env.mj_model.joint(f"{side}_ankle_roll_joint")
            self.assertIn(motor_joint.id, actuator_joint_ids)
            self.assertEqual(
                motor_joint.bodyid,
                env.mj_model.body(f"{side}_ankle_roll_motor_carrier").id,
            )

        pitch_backlash = env.mj_model.joint("left_ankle_pitch_backlash_joint")
        np.testing.assert_allclose(
            env.mj_model.jnt_range[pitch_backlash.id],
            np.deg2rad([-0.2, 0.2]),
            atol=1e-12,
        )
        self.assertTrue(bool(env.mj_model.jnt_limited[pitch_backlash.id]))
        self.assertNotIn(pitch_backlash.id, actuator_joint_ids)
        pitch_carrier = env.mj_model.body("left_ankle_pitch_output_carrier")
        self.assertEqual(
            env.mj_model.body("left_ankle_pitch_link").parentid,
            pitch_carrier.id,
        )
        self.assertEqual(
            env.mj_model.joint("left_ankle_pitch_joint").bodyid,
            pitch_carrier.id,
        )
        self.assertEqual(
            env.mj_model.equality("left_ankle_connect").obj1id,
            pitch_carrier.id,
        )

        waist_carrier = env.mj_model.body("waist_roll_pitch_output_carrier")
        self.assertEqual(env.mj_model.body("torso_link").parentid, waist_carrier.id)
        self.assertEqual(
            env.mj_model.joint("waist_pitch_joint").bodyid,
            waist_carrier.id,
        )
        for motor_body_name in (
            "torso_constraint_R_link",
            "torso_constraint_L_link",
        ):
            self.assertEqual(
                env.mj_model.body(motor_body_name).parentid,
                waist_carrier.id,
            )
        for axis_name in ("roll", "pitch"):
            joint = env.mj_model.joint(f"waist_{axis_name}_backlash_joint")
            np.testing.assert_allclose(
                env.mj_model.jnt_range[joint.id],
                np.deg2rad([-1.0, 1.0]),
                atol=1e-12,
            )
            self.assertTrue(bool(env.mj_model.jnt_limited[joint.id]))
            self.assertNotIn(joint.id, actuator_joint_ids)

    def test_backlash_joint_moves_physical_foot_without_changing_motor_angle(self):
        env = self._make_env()
        for side, half_width_deg in (("left", 0.6), ("right", 0.8)):
            motor_joint = env.mj_model.joint(f"{side}_ankle_roll_joint")
            backlash_joint = env.mj_model.joint(f"{side}_ankle_roll_backlash_joint")
            motor_qpos = int(motor_joint.qposadr[0])
            backlash_qpos = int(backlash_joint.qposadr[0])
            initial_motor_angle = float(env.mj_data.qpos[motor_qpos])

            env.mj_data.qpos[backlash_qpos] = np.deg2rad(half_width_deg)
            mujoco.mj_forward(env.mj_model, env.mj_data)

            self.assertEqual(float(env.mj_data.qpos[motor_qpos]), initial_motor_angle)
            self.assertAlmostEqual(
                float(env.mj_data.qpos[backlash_qpos]),
                np.deg2rad(half_width_deg),
                places=12,
            )
            carrier_rotation = env.mj_data.xmat[
                env.mj_model.body(f"{side}_ankle_roll_motor_carrier").id
            ].reshape(3, 3)
            foot_rotation = env.mj_data.xmat[
                env.mj_model.body(f"{side}_ankle_roll_link").id
            ].reshape(3, 3)
            relative_rotation = carrier_rotation.T @ foot_rotation
            expected_rotation = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, np.cos(np.deg2rad(half_width_deg)), -np.sin(np.deg2rad(half_width_deg))],
                    [0.0, np.sin(np.deg2rad(half_width_deg)), np.cos(np.deg2rad(half_width_deg))],
                ]
            )
            np.testing.assert_allclose(relative_rotation, expected_rotation, atol=1e-12)

        for group in env.closed_loop_transmissions:
            jacobian = env._closed_loop_output_to_motor_jacobian(group)
            self.assertEqual(np.linalg.matrix_rank(jacobian), jacobian.shape[0])

    def test_left_pitch_backlash_moves_foot_downstream_of_closed_loop_output(self):
        env = self._make_env()
        output_joint = env.mj_model.joint("left_ankle_pitch_joint")
        backlash_joint = env.mj_model.joint("left_ankle_pitch_backlash_joint")
        output_qpos = int(output_joint.qposadr[0])
        backlash_qpos = int(backlash_joint.qposadr[0])
        initial_output_angle = float(env.mj_data.qpos[output_qpos])

        env.mj_data.qpos[backlash_qpos] = np.deg2rad(0.2)
        mujoco.mj_forward(env.mj_model, env.mj_data)

        self.assertEqual(float(env.mj_data.qpos[output_qpos]), initial_output_angle)
        carrier_rotation = env.mj_data.xmat[
            env.mj_model.body("left_ankle_pitch_output_carrier").id
        ].reshape(3, 3)
        foot_rotation = env.mj_data.xmat[
            env.mj_model.body("left_ankle_pitch_link").id
        ].reshape(3, 3)
        relative_rotation = carrier_rotation.T @ foot_rotation
        angle = np.deg2rad(0.2)
        expected_rotation = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        np.testing.assert_allclose(relative_rotation, expected_rotation, atol=1e-12)

    def test_pelvis_imu_under_reports_pitch_to_drive_forward_compensation(self):
        env = self._make_env()
        self.assertEqual(env.config["PELVIS_IMU_PITCH_BIAS_DEG"], -1.0)

        true_quat = np.array([1.0, 0.0, 0.0, 0.0])
        reported_quat = apply_local_pitch_bias_wxyz(true_quat, np.deg2rad(-1.0))
        expected_quat = np.array(
            [np.cos(np.deg2rad(0.5)), 0.0, -np.sin(np.deg2rad(0.5)), 0.0]
        )
        np.testing.assert_allclose(reported_quat, expected_quat, atol=1e-12)

        env.mj_data.qpos[:7] = [0.0, 0.0, 1.0, *true_quat]
        mujoco.mj_forward(env.mj_model, env.mj_data)
        obs = env.prepare_obs()
        np.testing.assert_array_equal(obs["floating_base_pose"][3:7], true_quat)

    def test_waist_backlash_moves_torso_after_closed_loop_outputs(self):
        env = self._make_env()
        carrier_id = env.mj_model.body("waist_roll_pitch_output_carrier").id
        torso_id = env.mj_model.body("torso_link").id
        original_output_angles = {
            name: float(env.mj_data.qpos[env.mj_model.joint(name).qposadr[0]])
            for name in ("waist_roll_joint", "waist_pitch_joint")
        }

        for axis_name, expected_axis in (
            ("roll", np.array([1.0, 0.0, 0.0])),
            ("pitch", np.array([0.0, 1.0, 0.0])),
        ):
            for backlash_name in ("roll", "pitch"):
                joint = env.mj_model.joint(f"waist_{backlash_name}_backlash_joint")
                env.mj_data.qpos[joint.qposadr[0]] = 0.0
            joint = env.mj_model.joint(f"waist_{axis_name}_backlash_joint")
            env.mj_data.qpos[joint.qposadr[0]] = np.deg2rad(1.0)
            mujoco.mj_forward(env.mj_model, env.mj_data)

            relative_rotation = (
                env.mj_data.xmat[carrier_id].reshape(3, 3).T
                @ env.mj_data.xmat[torso_id].reshape(3, 3)
            )
            angle = np.deg2rad(1.0)
            skew_axis = np.array(
                [
                    [0.0, -expected_axis[2], expected_axis[1]],
                    [expected_axis[2], 0.0, -expected_axis[0]],
                    [-expected_axis[1], expected_axis[0], 0.0],
                ]
            )
            expected_rotation = (
                np.eye(3)
                + np.sin(angle) * skew_axis
                + (1.0 - np.cos(angle)) * (skew_axis @ skew_axis)
            )
            np.testing.assert_allclose(relative_rotation, expected_rotation, atol=1e-12)

        for name, original_angle in original_output_angles.items():
            self.assertEqual(
                float(env.mj_data.qpos[env.mj_model.joint(name).qposadr[0]]),
                original_angle,
            )


if __name__ == "__main__":
    unittest.main()
