"""Convert an XR headset pose into calibrated H2 head joint commands."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

XRT_TO_ROBOT = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)

XRT_BODY_JOINT_COUNT = 24
XRT_SPINE3_INDEX = 9
XRT_NECK_INDEX = 12
XRT_LEFT_COLLAR_INDEX = 13
XRT_RIGHT_COLLAR_INDEX = 14


class HeadPitchJoystick:
    """Integrate right-stick Y into an H2 pitch target; hold yaw constant.

    Angles are joint targets, not measured feedback. Limits match the C++
    h2_head_control.hpp clamp. Positive stick Y increases pitch (+Y, look down).
    """

    PITCH_LIMITS = (-0.5236, 0.83775)
    YAW_LIMITS = (-1.7453, 1.7453)

    def __init__(self, pitch_deg=0.0, yaw_deg=0.0, rate_deg_s=5.0, deadzone=0.15):
        if not all(math.isfinite(v) for v in (pitch_deg, yaw_deg, rate_deg_s, deadzone)):
            raise ValueError("head angle, rate and deadzone must be finite")
        self.pitch = math.radians(pitch_deg)
        self.yaw = math.radians(yaw_deg)
        if not self.PITCH_LIMITS[0] <= self.pitch <= self.PITCH_LIMITS[1]:
            raise ValueError("head pitch must be within approximately -30 to +48 degrees")
        if not self.YAW_LIMITS[0] <= self.yaw <= self.YAW_LIMITS[1]:
            raise ValueError("head yaw must be within approximately -100 to +100 degrees")
        if not 0.0 < rate_deg_s <= 30.0:
            raise ValueError("head pitch rate must be in (0, 30] degrees/second")
        if not 0.0 <= deadzone < 1.0:
            raise ValueError("head stick deadzone must be in [0, 1)")
        self.rate = math.radians(rate_deg_s)
        self.deadzone = deadzone
        self.moving = False

    def update(self, stick_y: float, dt: float) -> tuple[float, float]:
        self.moving = False
        if not math.isfinite(stick_y) or not math.isfinite(dt) or dt <= 0:
            return self.pitch, self.yaw
        value = min(1.0, max(-1.0, stick_y))
        if abs(value) > self.deadzone:
            value = math.copysign((abs(value) - self.deadzone) / (1 - self.deadzone), value)
            # Never integrate a stall/pause into a large target jump.
            delta = value * self.rate * min(dt, 0.1)
            updated = min(self.PITCH_LIMITS[1], max(self.PITCH_LIMITS[0], self.pitch + delta))
            self.moving = updated != self.pitch
            self.pitch = updated
        return self.pitch, self.yaw


def _headset_rotation_robot_frame(headset_pose: np.ndarray) -> Rotation:
    """Return the headset rotation in the robot's X-forward, Z-up frame."""
    pose = np.asarray(headset_pose, dtype=np.float64).reshape(-1)
    if pose.shape != (7,):
        raise ValueError(f"headset pose must have shape (7,), got {pose.shape}")
    if not np.all(np.isfinite(pose)):
        raise ValueError("headset pose contains non-finite values")

    quaternion = pose[3:]
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError("headset pose contains a zero quaternion")

    xrt_rotation = Rotation.from_quat(quaternion / norm).as_matrix()
    return Rotation.from_matrix(XRT_TO_ROBOT @ xrt_rotation @ XRT_TO_ROBOT.T)


def _normalized(vector: np.ndarray, *, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError(f"cannot construct body frame: {name} vector is degenerate")
    return vector / norm


def _body_rotation_robot_frame(body_poses: np.ndarray) -> Rotation:
    """Return the upper torso orientation in the robot's anatomical frame.

    X points forward, Y left, and Z up. The frame is constructed from tracked
    joint positions so it does not depend on XRT's joint-local axis convention.
    """
    poses = np.asarray(body_poses, dtype=np.float64)
    if poses.shape != (XRT_BODY_JOINT_COUNT, 7):
        raise ValueError(
            f"body poses must have shape ({XRT_BODY_JOINT_COUNT}, 7), got {poses.shape}"
        )
    if not np.all(np.isfinite(poses)):
        raise ValueError("body poses contain non-finite values")

    positions = (XRT_TO_ROBOT @ poses[:, :3].T).T
    left = _normalized(
        positions[XRT_LEFT_COLLAR_INDEX] - positions[XRT_RIGHT_COLLAR_INDEX],
        name="collar",
    )
    up = positions[XRT_NECK_INDEX] - positions[XRT_SPINE3_INDEX]
    up = _normalized(up - np.dot(up, left) * left, name="upper-spine")
    forward = _normalized(np.cross(left, up), name="forward")
    left = np.cross(up, forward)
    return Rotation.from_matrix(np.column_stack((forward, left, up)))


class FixedHeadPose(HeadPitchJoystick):
    """Keep configured head joint angles, ignoring all joystick input."""

    def update(self, stick_y: float, dt: float) -> tuple[float, float]:
        self.moving = False
        return self.pitch, self.yaw


class HeadPoseCommandTracker:
    """Emit H2 head angles from the human head orientation relative to torso.

    H2 applies pitch about +Y followed by yaw about the pitch link's +Z axis.
    The intrinsic ``YZX`` decomposition matches that chain; headset roll is
    deliberately discarded.
    """

    def __init__(self) -> None:
        self._reference_head_in_body: Rotation | None = None

    @property
    def is_calibrated(self) -> bool:
        return self._reference_head_in_body is not None

    @staticmethod
    def _head_in_body(headset_pose: np.ndarray, body_poses: np.ndarray) -> Rotation:
        head = _headset_rotation_robot_frame(headset_pose)
        body = _body_rotation_robot_frame(body_poses)
        return body.inv() * head

    def calibrate(self, headset_pose: np.ndarray, body_poses: np.ndarray) -> None:
        self._reference_head_in_body = self._head_in_body(headset_pose, body_poses)

    def command(self, headset_pose: np.ndarray, body_poses: np.ndarray) -> np.ndarray:
        if self._reference_head_in_body is None:
            raise RuntimeError("head pose tracker must be calibrated before use")

        current_head_in_body = self._head_in_body(headset_pose, body_poses)
        relative = current_head_in_body * self._reference_head_in_body.inv()
        pitch, yaw, _roll = relative.as_euler("YZX", degrees=False)
        return np.asarray([pitch, yaw], dtype=np.float32)
