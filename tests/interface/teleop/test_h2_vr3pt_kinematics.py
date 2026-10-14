import unittest

import numpy as np

from interface.kinematics.h2 import (
    H2_DEFAULT_ANGLES_LOCAL,
    H2_JOINT_NAMES_LOCAL,
    instantiate_h2_robot_model,
)
from interface.teleop.pico import ThreePointPose
from interface.teleop.h2_vr3pt_kinematics import (
    G1_FEEDBACK_DEFAULT_ANGLES,
    G1_FEEDBACK_POLICY_INDEX,
    H2_NATIVE_DEFAULT_ANGLES_POLICY,
    H2_NATIVE_POLICY_TO_LOCAL,
    h2_configuration_from_g1_compatible_feedback,
    h2_local_joints_from_g1_compatible_feedback,
)

H2_DEFAULT_KEYPOINT_POSITIONS = np.asarray(
    [
        [0.241776, 0.269187, -0.070610],
        [0.241776, -0.269187, -0.070610],
        [0.0, 0.0, 0.473037],
    ]
)
H2_DEFAULT_KEYPOINT_QUATERNIONS = np.asarray(
    [
        [0.916733, 0.089932, 0.386588, 0.045410],
        [0.916733, -0.089932, 0.386588, -0.045410],
        [1.0, 0.0, 0.0, 0.0],
    ]
)


class H2VR3PointKinematicsTest(unittest.TestCase):
    def test_feedback_inverse_recovers_h2_policy_delta(self):
        policy_delta = np.linspace(-0.14, 0.14, 29)
        feedback = G1_FEEDBACK_DEFAULT_ANGLES.copy()
        for feedback_index, policy_index in enumerate(G1_FEEDBACK_POLICY_INDEX):
            feedback[feedback_index] += policy_delta[policy_index]

        local = h2_local_joints_from_g1_compatible_feedback(feedback)
        expected = H2_DEFAULT_ANGLES_LOCAL.copy()
        expected[H2_NATIVE_POLICY_TO_LOCAL] = (
            H2_NATIVE_DEFAULT_ANGLES_POLICY + policy_delta
        )
        np.testing.assert_allclose(local, expected, atol=1e-12)

    def test_default_feedback_builds_h2_default_configuration(self):
        model = instantiate_h2_robot_model()
        configuration = h2_configuration_from_g1_compatible_feedback(
            model, G1_FEEDBACK_DEFAULT_ANGLES
        )
        actual = np.asarray(
            [configuration[model.dof_index(name)] for name in H2_JOINT_NAMES_LOCAL]
        )
        np.testing.assert_allclose(actual, H2_DEFAULT_ANGLES_LOCAL, atol=1e-12)

    def test_initial_calibration_matches_h2_training_keypoints(self):
        processor = ThreePointPose(robot_profile="h2-native")
        processor.reset_with_measured_q(G1_FEEDBACK_DEFAULT_ANGLES)
        raw = np.asarray(
            [
                [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                [-0.1, -0.2, 0.4, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        calibrated = processor.process_vr_pose(raw)

        np.testing.assert_allclose(
            calibrated[:, :3], H2_DEFAULT_KEYPOINT_POSITIONS, atol=1e-6
        )
        for actual, expected in zip(
            calibrated[:, 3:], H2_DEFAULT_KEYPOINT_QUATERNIONS, strict=True
        ):
            self.assertAlmostEqual(abs(float(np.dot(actual, expected))), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
