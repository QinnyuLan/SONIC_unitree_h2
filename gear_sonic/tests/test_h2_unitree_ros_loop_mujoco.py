import hashlib
from pathlib import Path
import unittest

import mujoco
import numpy as np
import yaml

from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = (
    REPO_ROOT
    / "gear_sonic/data/assets/robot_description/unitree_ros/h2_description"
)
CONFIG_DIR = REPO_ROOT / "gear_sonic/utils/mujoco_sim/wbc_configs"
LOOP_CONFIG_PATH = CONFIG_DIR / "h2_lora_sonic_unitree_ros_loop.yaml"
NATIVE_LOOP_CONFIG_PATH = CONFIG_DIR / "h2_native_sonic_unitree_ros_loop.yaml"
LEGACY_CONFIG_PATH = CONFIG_DIR / "h2_lora_sonic_unitree_ros.yaml"
NATIVE_CONTRACT_PATH = (
    REPO_ROOT / "sonic_h2_native_release/exported/deployment_contract.json"
)

UPSTREAM_SHA256 = {
    "H2_loop.xml": "399284179c4b7067b82cc5dc910b832936f35c0d7981dc8954897251f731e458",
    "H2_loop.urdf": "5bf8f8cc30cce3feaf7acca8a727cf5a768ea981a4cac55c547ca2dd49927c42",
    "meshes/Link-pitch-L-A3.STL": "b1fa54e1503dc29e9f16e714c1f40a86f3e317a881d3d7662c1419d10502318e",
    "meshes/Link-pitch-R-A3.STL": "aaf6a787d16b6b971926634794d3b3bd5d7cd3443720780663f15037a5ecaf06",
    "meshes/left_ankle_A_link.STL": "32fa65107d904a016990a94edae01eab17ce5a84a0eef02af96056cdbc7a478c",
    "meshes/right_ankle_A_link.STL": "46d6fa85f043e50228b45cc2f6f6f95f22be327e9b68a767417ec5fc1c9ceb06",
}


class H2UnitreeRosLoopMujocoTest(unittest.TestCase):
    def _make_env(self, config_path: Path) -> DefaultEnv:
        with config_path.open() as config_file:
            config = yaml.safe_load(config_file)
        config["PRINT_SCENE_INFORMATION"] = False
        return DefaultEnv(config, onscreen=False)

    def test_imported_files_match_upstream_commit(self):
        for relative_path, expected_hash in UPSTREAM_SHA256.items():
            digest = hashlib.sha256((ASSET_DIR / relative_path).read_bytes()).hexdigest()
            self.assertEqual(digest, expected_hash, relative_path)

    def test_loop_model_has_official_motor_and_constraint_structure(self):
        model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "H2_loop.xml"))
        self.assertEqual((model.nq, model.nv, model.njnt, model.nu, model.neq), (68, 61, 44, 31, 6))
        self.assertEqual(
            [model.equality(i).name for i in range(model.neq)],
            [
                "left_ankle_connect",
                "right_ankle_connect",
                "left_knee_connect",
                "right_knee_connect",
                "right_waist_connect",
                "left_waist_connect",
            ],
        )
        actuator_names = [model.actuator(i).name for i in range(model.nu)]
        self.assertIn("left_knee_motor_joint", actuator_names)
        self.assertIn("left_ankle_A_joint", actuator_names)
        self.assertIn("torso_constraint_R_joint", actuator_names)
        self.assertNotIn("left_knee_joint", actuator_names)
        self.assertNotIn("left_ankle_pitch_joint", actuator_names)
        self.assertNotIn("waist_roll_joint", actuator_names)

    def test_loop_profile_projects_default_pose_and_uses_true_addresses(self):
        env = self._make_env(LOOP_CONFIG_PATH)
        expected = np.asarray(env.config["DEFAULT_DOF_ANGLES"], dtype=float)
        np.testing.assert_allclose(
            env.mj_data.qpos[env.body_qpos_index], expected, atol=2e-9
        )

        equality_type = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        equality_rows = [
            row
            for row in range(env.mj_data.nefc)
            if int(env.mj_data.efc_type[row]) == equality_type
        ]
        self.assertEqual(len(equality_rows), 18)
        self.assertLess(np.linalg.norm(env.mj_data.efc_pos[equality_rows]), 1e-9)

        knee_position = env.body_joint_names.index("left_knee_joint")
        self.assertEqual(
            env.mj_model.actuator(int(env.body_actuator_index[knee_position])).name,
            "left_knee_motor_joint",
        )
        right_hip_position = env.body_joint_names.index("right_hip_pitch_joint")
        self.assertNotEqual(
            int(env.body_qpos_index[right_hip_position]),
            int(env.body_joint_index[right_hip_position] + env.qpos_offset - 1),
        )

    def test_native_profile_matches_official_h2_actuator_contract(self):
        import json

        with NATIVE_LOOP_CONFIG_PATH.open() as config_file:
            config = yaml.safe_load(config_file)
        contract = json.loads(NATIVE_CONTRACT_PATH.read_text())

        self.assertEqual(config["MOTOR_KP"][:6], [200.0, 200.0, 200.0, 200.0, 40.0, 40.0])
        self.assertEqual(config["MOTOR_KD"][:6], [4.0, 4.0, 4.0, 4.0, 2.0, 2.0])
        self.assertEqual(config["MOTOR_KP"][12:17], [150.0, 150.0, 150.0, 40.0, 40.0])
        self.assertEqual(config["MOTOR_KD"][12:17], [3.0, 3.0, 3.0, 2.0, 2.0])
        self.assertEqual(contract["action"]["action_scale"][0], 0.45)
        self.assertEqual(contract["action"]["action_scale"][13], 0.4125)
        self.assertEqual(contract["action"]["action_scale"][17], 0.11875)
        self.assertEqual(contract["runtime"]["actuator_profile"], "unitree_rl_mjlab_h2")

    def test_native_profile_increases_floor_friction_at_runtime(self):
        official_model = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "H2_loop.xml"))
        official_floor = official_model.geom("floor")
        np.testing.assert_allclose(
            official_model.geom_friction[official_floor.id],
            [1.0, 0.005, 0.0001],
            atol=0.0,
        )

        env = self._make_env(NATIVE_LOOP_CONFIG_PATH)
        floor = env.mj_model.geom("floor")
        np.testing.assert_allclose(
            env.mj_model.geom_friction[floor.id],
            [1.2, 0.005, 0.0001],
            atol=0.0,
        )

    def test_closed_loop_jacobians_are_full_rank_and_preserve_virtual_work(self):
        env = self._make_env(LOOP_CONFIG_PATH)
        for group in env.closed_loop_transmissions:
            jacobian = env._closed_loop_output_to_motor_jacobian(group)
            self.assertEqual(np.linalg.matrix_rank(jacobian), jacobian.shape[0])
            output_torque = np.arange(1, jacobian.shape[0] + 1, dtype=float)
            motor_velocity = np.linspace(0.2, 0.4, jacobian.shape[1])
            motor_torque = jacobian.T @ output_torque
            output_velocity = jacobian @ motor_velocity
            self.assertAlmostEqual(
                float(motor_torque @ motor_velocity),
                float(output_torque @ output_velocity),
                places=12,
            )

        groups = {group["name"]: group for group in env.closed_loop_transmissions}
        left_ankle = env._closed_loop_output_to_motor_jacobian(
            groups["left_ankle_loop"]
        )
        waist = env._closed_loop_output_to_motor_jacobian(
            groups["waist_roll_pitch_loop"]
        )
        np.testing.assert_allclose(left_ankle[0], [1.0, 0.0], atol=1e-12)
        self.assertLess(float(left_ankle[1, 1]), 0.0)
        self.assertGreater(float(waist[0, 0] * waist[0, 1]), 0.0)
        self.assertLess(float(waist[1, 0] * waist[1, 1]), 0.0)

    def test_legacy_unitree_ros_profile_remains_direct_drive(self):
        env = self._make_env(LEGACY_CONFIG_PATH)
        self.assertEqual((env.mj_model.njnt, env.mj_model.nu, env.mj_model.neq), (32, 31, 0))
        self.assertEqual(env.closed_loop_transmissions, [])
        np.testing.assert_array_equal(env.body_joint_index, env.body_actuator_joint_index)


if __name__ == "__main__":
    unittest.main()
