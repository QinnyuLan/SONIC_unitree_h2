import unittest

import numpy as np

from gear_sonic.scripts.pico_manager_thread_server import _process_xrt_3pt_pose
from scripts.make_h2_vr3pt_arms_forward_recording import generate_arrays


class H2VR3PointArmsForwardRecordingTest(unittest.TestCase):
    def test_replay_round_trips_robot_targets_and_uses_mode_one(self):
        arrays, metadata = generate_arrays(
            fps=10.0,
            neutral_seconds=0.2,
            transition_seconds=0.3,
            hold_seconds=0.2,
        )
        self.assertEqual(metadata["encoder_mode"], 1)
        expected_start = np.asarray(metadata["neutral_robot_vr3_pose_wxyz"])
        expected_end = np.asarray(metadata["target_robot_vr3_pose_wxyz"])

        for frame, expected in ((0, expected_start), (-1, expected_end)):
            actual = _process_xrt_3pt_pose(
                arrays["left_controller_pose"][frame],
                arrays["right_controller_pose"][frame],
                arrays["headset_pose"][frame],
            )
            np.testing.assert_allclose(actual[:, :3], expected[:, :3], atol=1e-6)
            for actual_quat, expected_quat in zip(actual[:, 3:], expected[:, 3:], strict=True):
                self.assertAlmostEqual(abs(float(np.dot(actual_quat, expected_quat))), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
