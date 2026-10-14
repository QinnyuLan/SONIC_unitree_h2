import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.utils.teleop.smpl_joint_frame_calibration import (
    corrected_xrt_global_rotations,
    default_joint_frame_quaternions,
    load_joint_frame_calibration,
)


class SmplJointFrameCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.body_poses = np.zeros((24, 7), dtype=np.float64)
        self.body_poses[:, 6] = 1.0

    def test_default_matches_original_uniform_y_180_rotation(self):
        actual = corrected_xrt_global_rotations(self.body_poses).as_matrix()
        expected = Rotation.from_euler("y", 180.0, degrees=True).as_matrix()
        np.testing.assert_allclose(actual, np.repeat(expected[None], 24, axis=0), atol=1e-12)

    def test_loaded_joint_override_is_right_multiplied(self):
        override = Rotation.from_euler("xyz", [10.0, 20.0, -30.0], degrees=True)
        payload = {
            "format_version": 1,
            "default_joint_frame_quat_wxyz": default_joint_frame_quaternions()[0].tolist(),
            "joint_corrections": [
                {
                    "joint_name": "left_shoulder",
                    "joint_index": 16,
                    "child_index": 18,
                    "quat_wxyz": override.as_quat(scalar_first=True).tolist(),
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            calibration = load_joint_frame_calibration(path)

        actual = corrected_xrt_global_rotations(
            self.body_poses, calibration
        ).as_matrix()
        np.testing.assert_allclose(actual[16], override.as_matrix(), atol=1e-12)
        np.testing.assert_allclose(
            actual[17], Rotation.from_euler("y", 180.0, degrees=True).as_matrix(), atol=1e-12
        )

    def test_duplicate_joint_override_is_rejected(self):
        quat = default_joint_frame_quaternions()[0].tolist()
        payload = {
            "format_version": 1,
            "joint_corrections": [
                {"joint_index": 16, "quat_wxyz": quat},
                {"joint_index": 16, "quat_wxyz": quat},
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_joint_frame_calibration(path)


if __name__ == "__main__":
    unittest.main()
