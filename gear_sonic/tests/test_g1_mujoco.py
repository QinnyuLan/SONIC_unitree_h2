from pathlib import Path
import unittest

import numpy as np
import yaml

from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml"


class G1MujocoTest(unittest.TestCase):
    def test_legacy_effort_limits_follow_mujoco_actuator_order(self):
        with CONFIG_PATH.open() as config_file:
            config = yaml.safe_load(config_file)
        config["PRINT_SCENE_INFORMATION"] = False
        config["enable_waist"] = True

        env = DefaultEnv(config, onscreen=False)
        configured_limits = np.asarray(config["motor_effort_limit_list"], dtype=float)

        self.assertEqual(config["MOTOR_EFFORT_LIMIT_ORDER"], "actuator")
        np.testing.assert_array_equal(env.torque_limit, configured_limits)

        body_limits = dict(zip(env.body_joint_names, env.body_torque_limit))
        self.assertEqual(body_limits["right_shoulder_pitch_joint"], 25.0)
        self.assertEqual(body_limits["right_shoulder_roll_joint"], 25.0)
        self.assertEqual(body_limits["right_shoulder_yaw_joint"], 25.0)
        self.assertEqual(body_limits["right_elbow_joint"], 25.0)

        np.testing.assert_array_equal(
            env.torque_limit[env.left_hand_actuator_index],
            [2.45, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
        )
        np.testing.assert_array_equal(
            env.torque_limit[env.right_hand_actuator_index],
            [2.45, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
        )


if __name__ == "__main__":
    unittest.main()
