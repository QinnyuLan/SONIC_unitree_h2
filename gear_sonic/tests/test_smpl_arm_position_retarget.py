import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.utils.teleop.smpl_arm_position_retarget import (
    SMPL_ARM_CHAINS,
    XRT_TO_ROBOT,
    retarget_smpl_arms_from_xrt_positions,
    smpl_joints_world_from_local,
)


def _angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first = a - b
    second = c - b
    cosine = np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


class SmplArmPositionRetargetTest(unittest.TestCase):
    def setUp(self):
        self.smpl = np.zeros((24, 3), dtype=np.float64)
        self.smpl[16] = [0.0, 0.2, 0.4]
        self.smpl[18] = [0.0, 0.2, 0.1]
        self.smpl[20] = [0.0, 0.2, -0.15]
        self.smpl[22] = [0.0, 0.2, -0.25]
        self.smpl[17] = [0.0, -0.2, 0.4]
        self.smpl[19] = [0.0, -0.2, 0.1]
        self.smpl[21] = [0.0, -0.2, -0.15]
        self.smpl[23] = [0.0, -0.2, -0.25]

        desired_robot = np.zeros((24, 3), dtype=np.float64)
        for shoulder, elbow, wrist, hand in SMPL_ARM_CHAINS:
            side = 0.2 if shoulder == 16 else -0.2
            desired_robot[shoulder] = [0.0, side, 0.4]
            desired_robot[elbow] = [0.3, side, 0.4]
            desired_robot[wrist] = [0.55, side, 0.4]
            desired_robot[hand] = [0.65, side, 0.4]

        self.body_poses = np.zeros((24, 7), dtype=np.float64)
        self.body_poses[:, 6] = 1.0
        self.body_poses[:, :3] = (XRT_TO_ROBOT.T @ desired_robot.T).T

    def test_arm_directions_follow_xrt_positions_and_preserve_lengths(self):
        corrected = retarget_smpl_arms_from_xrt_positions(self.body_poses, self.smpl)

        np.testing.assert_allclose(corrected[0:16], self.smpl[0:16])
        for shoulder, elbow, wrist, hand in SMPL_ARM_CHAINS:
            elbow_angle = _angle_degrees(corrected[shoulder], corrected[elbow], corrected[wrist])
            self.assertAlmostEqual(elbow_angle, 180.0, places=5)
            self.assertAlmostEqual(
                np.linalg.norm(corrected[elbow] - corrected[shoulder]),
                np.linalg.norm(self.smpl[elbow] - self.smpl[shoulder]),
            )
            self.assertAlmostEqual(
                np.linalg.norm(corrected[wrist] - corrected[elbow]),
                np.linalg.norm(self.smpl[wrist] - self.smpl[elbow]),
            )
            self.assertAlmostEqual(
                np.linalg.norm(corrected[hand] - corrected[wrist]),
                np.linalg.norm(self.smpl[hand] - self.smpl[wrist]),
            )

    def test_pelvis_rotation_is_removed_from_position_directions(self):
        root_rotation_robot = Rotation.from_euler("z", 65.0, degrees=True).as_matrix()
        root_rotation_xrt = XRT_TO_ROBOT.T @ root_rotation_robot @ XRT_TO_ROBOT
        rotated = self.body_poses.copy()
        rotated[:, :3] = (XRT_TO_ROBOT.T @ root_rotation_robot @ XRT_TO_ROBOT @ self.body_poses[:, :3].T).T
        rotated[:, 3:] = Rotation.from_matrix(root_rotation_xrt).as_quat()

        expected = retarget_smpl_arms_from_xrt_positions(self.body_poses, self.smpl)
        actual = retarget_smpl_arms_from_xrt_positions(rotated, self.smpl)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_world_conversion_round_trips_deploy_localization(self):
        root_quat = Rotation.from_euler("xyz", [10.0, -15.0, 40.0], degrees=True).as_quat(scalar_first=True)
        world = smpl_joints_world_from_local(self.smpl, root_quat)
        recovered = Rotation.from_quat(root_quat, scalar_first=True).inv().apply(world)
        np.testing.assert_allclose(recovered, self.smpl, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
