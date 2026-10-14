import json
import struct
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from interface.teleop.head_pose_command import (
    XRT_LEFT_COLLAR_INDEX,
    XRT_NECK_INDEX,
    XRT_RIGHT_COLLAR_INDEX,
    XRT_SPINE3_INDEX,
    XRT_TO_ROBOT,
    HeadPoseCommandTracker,
)
from interface.teleop.transport.packets import (
    HEADER_SIZE,
    build_head_pose_message,
)


def _xrt_pose_from_robot_rotation(rotation: Rotation) -> np.ndarray:
    robot_matrix = rotation.as_matrix()
    xrt_matrix = XRT_TO_ROBOT.T @ robot_matrix @ XRT_TO_ROBOT
    return np.concatenate(([0.0, 0.0, 0.0], Rotation.from_matrix(xrt_matrix).as_quat()))


def _xrt_body_poses_from_robot_rotation(rotation: Rotation) -> np.ndarray:
    robot_positions = np.zeros((24, 3), dtype=np.float64)
    robot_positions[XRT_SPINE3_INDEX] = [0.0, 0.0, 0.35]
    robot_positions[XRT_NECK_INDEX] = [0.0, 0.0, 0.60]
    robot_positions[XRT_LEFT_COLLAR_INDEX] = [0.0, 0.20, 0.55]
    robot_positions[XRT_RIGHT_COLLAR_INDEX] = [0.0, -0.20, 0.55]
    robot_positions = rotation.apply(robot_positions)

    poses = np.zeros((24, 7), dtype=np.float64)
    poses[:, :3] = (XRT_TO_ROBOT.T @ robot_positions.T).T
    poses[:, 6] = 1.0
    return poses


class HeadPoseCommandTest(unittest.TestCase):
    def test_calibration_removes_initial_orientation(self):
        tracker = HeadPoseCommandTracker()
        body = _xrt_body_poses_from_robot_rotation(Rotation.from_euler("Z", 0.4))
        initial = _xrt_pose_from_robot_rotation(Rotation.from_euler("Z", 0.7))
        tracker.calibrate(initial, body)
        np.testing.assert_allclose(tracker.command(initial, body), [0.0, 0.0], atol=1e-6)

    def test_pitch_and_yaw_follow_h2_joint_order(self):
        tracker = HeadPoseCommandTracker()
        body = _xrt_body_poses_from_robot_rotation(Rotation.identity())
        tracker.calibrate(_xrt_pose_from_robot_rotation(Rotation.identity()), body)
        target = Rotation.from_euler("YZX", [0.25, -0.6, 0.3])
        command = tracker.command(_xrt_pose_from_robot_rotation(target), body)
        np.testing.assert_allclose(command, [0.25, -0.6], atol=1e-6)

    def test_common_body_and_head_rotation_produces_zero_command(self):
        tracker = HeadPoseCommandTracker()
        neutral_body = _xrt_body_poses_from_robot_rotation(Rotation.identity())
        neutral_head = _xrt_pose_from_robot_rotation(Rotation.identity())
        tracker.calibrate(neutral_head, neutral_body)

        body_rotation = Rotation.from_euler("XYZ", [0.2, -0.35, 0.8])
        moved_body = _xrt_body_poses_from_robot_rotation(body_rotation)
        moved_head = _xrt_pose_from_robot_rotation(body_rotation)
        np.testing.assert_allclose(
            tracker.command(moved_head, moved_body), [0.0, 0.0], atol=1e-6
        )

    def test_command_uses_head_motion_in_current_body_frame(self):
        tracker = HeadPoseCommandTracker()
        mounting_offset = Rotation.from_euler("XYZ", [0.15, -0.2, 0.1])
        neutral_body = _xrt_body_poses_from_robot_rotation(Rotation.identity())
        tracker.calibrate(
            _xrt_pose_from_robot_rotation(mounting_offset), neutral_body
        )

        body_rotation = Rotation.from_euler("XYZ", [-0.25, 0.3, -0.7])
        relative_head = Rotation.from_euler("YZX", [0.35, -0.45, 0.2])
        moved_body = _xrt_body_poses_from_robot_rotation(body_rotation)
        moved_head = _xrt_pose_from_robot_rotation(
            body_rotation * relative_head * mounting_offset
        )
        np.testing.assert_allclose(
            tracker.command(moved_head, moved_body), [0.35, -0.45], atol=1e-6
        )

    def test_rejects_invalid_quaternion(self):
        tracker = HeadPoseCommandTracker()
        body = _xrt_body_poses_from_robot_rotation(Rotation.identity())
        with self.assertRaisesRegex(ValueError, "zero quaternion"):
            tracker.calibrate(np.zeros(7), body)

    def test_rejects_degenerate_body_frame(self):
        tracker = HeadPoseCommandTracker()
        with self.assertRaisesRegex(ValueError, "degenerate"):
            tracker.calibrate(
                _xrt_pose_from_robot_rotation(Rotation.identity()), np.zeros((24, 7))
            )

    def test_independent_wire_message(self):
        message = build_head_pose_message(0.25, -0.5, timestamp_monotonic=12.0)
        self.assertTrue(message.startswith(b"head_pose"))
        offset = len(b"head_pose")
        header = json.loads(message[offset : offset + HEADER_SIZE].rstrip(b"\x00"))
        self.assertEqual(header["fields"][0]["name"], "head_pitch_yaw")
        pitch, yaw = struct.unpack_from("<ff", message, offset + HEADER_SIZE)
        self.assertAlmostEqual(pitch, 0.25)
        self.assertAlmostEqual(yaw, -0.5)


if __name__ == "__main__":
    unittest.main()
