import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from gear_sonic.scripts.pico_manager_thread_server import (
    SMPL_BALANCE_NEUTRAL_JOINTS,
    capture_smpl_balance_neutral_quats,
    compute_from_body_poses,
)

PARENTS = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    22,
]


def _body_poses_from_local_rotations(local_rotvecs: np.ndarray) -> np.ndarray:
    local_rotations = Rotation.from_rotvec(local_rotvecs)
    global_rotations = []
    for joint_idx, parent_idx in enumerate(PARENTS):
        if parent_idx == -1:
            global_rotations.append(local_rotations[joint_idx])
        else:
            global_rotations.append(global_rotations[parent_idx] * local_rotations[joint_idx])

    body_poses = np.zeros((24, 7), dtype=np.float64)
    body_poses[:, 3:7] = np.asarray(
        [rotation.as_quat() for rotation in global_rotations]
    )
    return body_poses


class SmplBalanceNeutralTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        local_rotvecs = rng.normal(scale=0.08, size=(24, 3))
        local_rotvecs[0] = [0.0, 0.35, 0.0]
        self.body_poses = _body_poses_from_local_rotations(local_rotvecs)
        self.device = torch.device("cpu")

    def test_calibration_zeros_balance_joints_without_changing_upper_body(self):
        uncalibrated = compute_from_body_poses(PARENTS, self.device, self.body_poses)
        neutral = capture_smpl_balance_neutral_quats(PARENTS, self.body_poses)
        calibrated = compute_from_body_poses(
            PARENTS,
            self.device,
            self.body_poses,
            balance_neutral_quats=neutral,
        )

        pose_uncalibrated = uncalibrated["smpl_pose"][0].reshape(23, 3)
        pose_calibrated = calibrated["smpl_pose"][0].reshape(23, 3)
        balance_indices = [joint_idx - 1 for joint_idx in SMPL_BALANCE_NEUTRAL_JOINTS]
        upper_indices = list(range(11, 23))

        torch.testing.assert_close(
            pose_calibrated[balance_indices], torch.zeros((11, 3)), atol=1e-6, rtol=0.0
        )
        torch.testing.assert_close(
            pose_calibrated[upper_indices],
            pose_uncalibrated[upper_indices],
            atol=1e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            calibrated["smpl_root_quat"],
            uncalibrated["smpl_root_quat"],
            atol=1e-6,
            rtol=0.0,
        )

    def test_relative_leg_motion_is_preserved_after_calibration(self):
        neutral = capture_smpl_balance_neutral_quats(PARENTS, self.body_poses)
        moved_body_poses = self.body_poses.copy()

        local_rotations = []
        raw_global = Rotation.from_quat(self.body_poses[:, 3:7])
        for joint_idx, parent_idx in enumerate(PARENTS):
            if parent_idx == -1:
                local_rotations.append(raw_global[joint_idx])
            else:
                local_rotations.append(raw_global[parent_idx].inv() * raw_global[joint_idx])
        local_rotations[4] = local_rotations[4] * Rotation.from_euler("x", 10.0, degrees=True)
        moved_body_poses = _body_poses_from_local_rotations(
            np.asarray([rotation.as_rotvec() for rotation in local_rotations])
        )

        calibrated = compute_from_body_poses(
            PARENTS,
            self.device,
            moved_body_poses,
            balance_neutral_quats=neutral,
        )
        left_knee_rotvec = calibrated["smpl_pose"][0].reshape(23, 3)[3]
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(left_knee_rotvec)), np.deg2rad(10.0), places=5
        )

    def test_invalid_calibration_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            compute_from_body_poses(
                PARENTS,
                self.device,
                self.body_poses,
                balance_neutral_quats=np.zeros((23, 4)),
            )


if __name__ == "__main__":
    unittest.main()
