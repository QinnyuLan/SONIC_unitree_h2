from copy import deepcopy
import unittest

import numpy as np

from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import (
    apply_local_pitch_bias_wxyz,
)


class H2NativeImuPitchBiasTest(unittest.TestCase):
    @staticmethod
    def _load(version: str) -> dict:
        return SimLoopConfig(wbc_version=version, enable_onscreen=False).load_wbc_yaml()

    def test_profiles_only_override_reported_pelvis_pitch(self):
        baseline = self._load("h2_native_unitree_ros_loop")
        positive = self._load("h2_native_unitree_ros_loop_imu_pitch_bias_pos1")
        negative = self._load("h2_native_unitree_ros_loop_imu_pitch_bias_neg1")
        negative_two = self._load("h2_native_unitree_ros_loop_imu_pitch_bias_neg2")

        self.assertNotIn("PELVIS_IMU_PITCH_BIAS_DEG", baseline)
        self.assertEqual(positive["PELVIS_IMU_PITCH_BIAS_DEG"], 1.0)
        self.assertEqual(negative["PELVIS_IMU_PITCH_BIAS_DEG"], -1.0)
        self.assertEqual(negative_two["PELVIS_IMU_PITCH_BIAS_DEG"], -2.0)

        expected = deepcopy(baseline)
        expected.pop("VERSION")
        for variant in (positive, negative, negative_two):
            actual = deepcopy(variant)
            actual.pop("VERSION")
            actual.pop("PELVIS_IMU_PITCH_BIAS_DEG")
            self.assertEqual(actual, expected)
            self.assertNotIn("H2_ANKLE_ROLL_BACKLASH_DEG", variant)
            self.assertNotIn("H2_ANKLE_PITCH_BACKLASH_DEG", variant)
            self.assertNotIn("H2_WAIST_BACKLASH_DEG", variant)

    def test_bias_sign_matches_reported_pitch(self):
        level_quat = np.array([1.0, 0.0, 0.0, 0.0])
        for bias_deg in (-2.0, -1.0, 1.0):
            reported = apply_local_pitch_bias_wxyz(
                level_quat, np.deg2rad(bias_deg)
            )
            reported_pitch = np.arcsin(
                2.0 * (reported[0] * reported[2] - reported[3] * reported[1])
            )
            self.assertAlmostEqual(np.rad2deg(reported_pitch), bias_deg, places=12)


if __name__ == "__main__":
    unittest.main()
