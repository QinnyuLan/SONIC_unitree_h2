# Pico SMPL stream server for body tracking visualization

"""

# Recommended Command Line Arguments:
    # With VR3 PT visualization (by --vis_vr3pt) and optional SMPL body visualization (by --vis_smpl)
    # If you want to enable waist tracking in the VR3 PT visualization, please add --waist_tracking
    python -m interface.teleop.pico --manager \
        --vis_vr3pt --vis_smpl \
        --waist_tracking

    # VR3 PT visualization only (without SMPL body) — lower latency
    python -m interface.teleop.pico --manager --vis_vr3pt

# DEBUG VR3 PT VISUALIZATION:
    # A standalone test mode that captures one live frame and visualizes it.
    python -m interface.teleop.pico --vr3pt_live

# TIMING COMPARISON:
    # The visualizer automatically reports timing every 5 seconds when running:
    #   [Vis Timing] vr3pt: X.XXms | smpl: X.XXms | render: X.XXms | vr3pt_only: X.XXms | both(vr3pt+smpl): X.XXms

"""

from collections import defaultdict, deque
from enum import Enum, IntEnum
import os
import subprocess
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as sRot
import torch
import zmq

from interface.math.torch_transform import (
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
    quaternion_to_rotation_matrix,
)
from interface.teleop.h2_vr3pt_kinematics import (
    h2_configuration_from_g1_compatible_feedback,
)
from interface.teleop.head_pose_command import HeadPoseCommandTracker, HeadPitchJoystick, FixedHeadPose
from interface.teleop.manager_state_rate import ManagerStateRateLimiter
from interface.teleop.smpl_arm_position_retarget import (
    retarget_smpl_arms_from_xrt_positions,
    smpl_joints_world_from_local,
    xrt_joint_positions_in_pelvis_frame,
)
from interface.teleop.smpl_joint_frame_calibration import (
    corrected_xrt_global_rotations,
    load_joint_frame_calibration,
)
from interface.teleop.smpl_wrist_kinematics import (
    g1_wrist_reference_from_smpl_pose,
    h2_wrist_reference_from_smpl_pose,
)
from interface.teleop.transport.receiver import ZMQPoller

try:
    from interface.teleop.transport.packets import (
        build_command_message,
        build_head_pose_message,
        build_planner_message,
        pack_pose_message,
    )
except ImportError:

    def build_command_message(*args, **kwargs) -> bytes:
        raise RuntimeError("build_command_message unavailable")

    def build_head_pose_message(*args, **kwargs) -> bytes:
        raise RuntimeError("build_head_pose_message unavailable")

    def build_planner_message(*args, **kwargs) -> bytes:
        raise RuntimeError("build_planner_message unavailable")

    def pack_pose_message(*args, **kwargs) -> bytes:
        raise RuntimeError("pack_pose_message unavailable")


try:
    from interface.math.rotations import remove_smpl_base_rot, smpl_root_ytoz_up
except ImportError:
    print("Warning: interface.math.rotations not available.")
    remove_smpl_base_rot = None
    smpl_root_ytoz_up = None

try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    xrt = None


def initialize_xr_source():
    """Initialize either the real XRoboToolkit service or an offline backend."""
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk or provide "
            "--offline-vr-file."
        )
    if not getattr(xrt, "is_offline", False):
        subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()


def is_direct_offline_vr() -> bool:
    """Return whether the active source contains recorded headset/controller poses only."""
    return bool(xrt is not None and getattr(xrt, "is_offline_3pt", False))


def xr_tracking_data_available() -> bool:
    if is_direct_offline_vr():
        return bool(xrt.is_vr_3pt_data_available())
    return bool(xrt.is_body_data_available())


try:
    from interface.teleop.ik.gripper import (
        G1GripperInverseKinematicsSolver,
    )
except ImportError:
    print("Warning: G1GripperInverseKinematicsSolver not available.")
    G1GripperInverseKinematicsSolver = None

try:
    from interface.teleop.visualization import VR3PtPoseVisualizer
except ImportError:
    print("Warning: VR3PtPoseVisualizer not available (pyvista may not be installed).")
    VR3PtPoseVisualizer = None

try:
    from interface.teleop.visualization import (
        get_g1_key_frame_poses as get_robot_key_frame_poses,
    )
except ImportError:
    print("Warning: robot key-frame FK is unavailable (pyvista may not be installed).")
    get_robot_key_frame_poses = None


class LocomotionMode(IntEnum):
    """Locomotion mode enum for robot movement."""

    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3
    IDLE_SQUAT = 4
    IDLE_KNEEL_TWO_LEGS = 5
    IDLE_KNEEL = 6
    IDLE_LYING_FACE_DOWN = 7
    CRAWLING = 8
    IDLE_BOXING = 9
    WALK_BOXING = 10
    LEFT_PUNCH = 11
    RIGHT_PUNCH = 12
    RANDOM_PUNCH = 13
    ELBOW_CRAWLING = 14
    LEFT_HOOK = 15
    RIGHT_HOOK = 16
    FORWARD_JUMP = 17
    STEALTH_WALK = 18
    INJURED_WALK = 19


class StreamMode(Enum):
    OFF = 0
    POSE = 1
    PLANNER = 2
    PLANNER_FROZEN_UPPER_BODY = 3
    POSE_PAUSE = 4
    PLANNER_VR_3PT = 5


### Parse 3 point pose from SMPL
#
# OFFSETS: Rotation corrections applied to each keypoint to align SMPL joint frames
# with the desired robot/visualization coordinate convention.
#
# Index mapping (based on [0, 22, 23, 12].index(joint_id)):
#   - OFFSETS[0]: Root/Pelvis (joint 0)
#   - OFFSETS[1]: Left Wrist (joint 22)
#   - OFFSETS[2]: Right Wrist (joint 23)
#   - OFFSETS[3]: Neck (joint 12) - more stable than Head (joint 15) for body tracking
#
# Scipy euler rotation convention:
#   - Lowercase "xyz" = EXTRINSIC rotations (about the FIXED/ORIGINAL frame's axes)
#   - Uppercase "XYZ" = INTRINSIC rotations (about the ROTATING body's axes)
#
# For EXTRINSIC "xyz" with angles [a, b, c]:
#   All rotations are about the ORIGINAL frame's axes (before any rotation):
#     R_total = R_z(c) @ R_y(b) @ R_x(a)   (matrix multiplication order)
#   Applied as: first rotate 'a' about original X, then 'b' about original Y, then 'c' about original Z
#
# For INTRINSIC "XYZ" with angles [a, b, c]:
#   Each rotation is about the CURRENT (rotated) frame's axis:
#     R_total = R_x(a) @ R_y(b) @ R_z(c)   (matrix multiplication order)
#   Applied as: first rotate 'a' about X, then 'b' about NEW Y, then 'c' about NEW Z
#
OFFSETS = [
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),  # Root: yaw -90° about fixed Z
    sRot.from_euler("xyz", [90, 0, 0], degrees=True),  # L-Wrist: roll +90° about fixed X
    sRot.from_euler(
        "xyz", [-90, 0, 180], degrees=True
    ),  # R-Wrist: roll -90° about fixed X, then yaw 180° about fixed Z
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),  # Neck: yaw -90° about fixed Z
]


def _compute_rel_transform(pose, world_frame, scalar_first=True):
    """
    Transform a pose from Unity coordinate frame to robot coordinate frame.

    Args:
        pose: np.ndarray shape (7,) - [x, y, z, qx, qy, qz, qw] in Unity frame
        world_frame: np.ndarray shape (7,) - reference frame to compute relative transform
        scalar_first: bool - if True, quaternion is [qw, qx, qy, qz]; if False, [qx, qy, qz, qw]

    Returns:
        rel_pos: np.ndarray (3,) - position in robot frame
        rel_rot: np.ndarray (4,) - quaternion [qw, qx, qy, qz] in robot frame

    Coordinate transform matrix Q converts Unity (Y-up, left-handed) to Robot (Z-up, right-handed):
        Unity:  X-right, Y-up, Z-forward
        Robot:  X-forward, Y-left, Z-up
    """
    world_frame = world_frame.copy()

    # Q transforms Unity coordinates to Robot coordinates
    # Unity [x, y, z] -> Robot [-x, z, y]
    Q = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0.0]])
    pose[:3] = Q @ pose[:3]
    world_frame[:3] = Q @ world_frame[:3]
    rot_base = sRot.from_quat(world_frame[3:], scalar_first=scalar_first).as_matrix()
    rot = sRot.from_quat(pose[3:], scalar_first=scalar_first).as_matrix()
    rel_rot = sRot.from_matrix(Q @ (rot_base.T @ rot) @ Q.T)
    rel_pos = sRot.from_matrix(Q @ rot_base.T @ Q.T).apply(pose[:3] - world_frame[:3])
    return rel_pos, rel_rot.as_quat(scalar_first=True)


def _process_xrt_3pt_pose(
    left_controller_pose: np.ndarray,
    right_controller_pose: np.ndarray,
    headset_pose: np.ndarray,
) -> np.ndarray:
    """Convert three raw XRT poses to the robot frame without using SMPL.

    Input poses are absolute XRT ``[x,y,z,qx,qy,qz,qw]`` values. Output rows
    are left controller, right controller, and headset in
    ``[x,y,z,qw,qx,qy,qz]`` order. The coordinate matrix matches the official
    XRoboToolkit Python teleoperation sample's headset-to-world transform.
    """
    xrt_to_robot = np.array(
        [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    result = np.zeros((3, 7), dtype=np.float32)
    for index, pose in enumerate((left_controller_pose, right_controller_pose, headset_pose)):
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (7,):
            raise ValueError(f"XRT three-point pose must have shape (7,), got {pose.shape}")
        result[index, :3] = xrt_to_robot @ pose[:3]
        rotation = sRot.from_quat(pose[3:]).as_matrix()
        result[index, 3:] = sRot.from_matrix(
            xrt_to_robot @ rotation @ xrt_to_robot.T
        ).as_quat(scalar_first=True)
    return result


def _process_3pt_pose(smpl_pose_np):
    """
    Extract 3-point VR pose (L-Wrist, R-Wrist, Neck) from full SMPL body joint poses.

    NOTE: We use Neck (joint 12) instead of Head (joint 15) because:
      - Neck is more rigidly coupled to the torso
      - Head has high DoF (looking around) which doesn't reflect body pose
      - Neck provides more stable tracking for upper body orientation

    Args:
        smpl_pose_np: np.ndarray shape (24, 7) - 24 SMPL joints, each [x, y, z, qx, qy, qz, qw]
                      in Unity frame (scalar-last quaternion format)

    Returns:
        vr_3pt_pose: np.ndarray shape (3, 7) - 3 keypoints in robot frame
                     Each row is [x, y, z, qw, qx, qy, qz] (scalar-FIRST quaternion format)
                     Row 0: Left Wrist (SMPL joint 22)
                     Row 1: Right Wrist (SMPL joint 23)
                     Row 2: Neck (SMPL joint 12)

                     IMPORTANT: Positions and orientations are RELATIVE TO ROOT (pelvis).

    Processing Steps:
        1. Transform all 24 joints from Unity frame to robot frame
        2. Extract 4 keypoints: Root(0), L-Wrist(22), R-Wrist(23), Neck(12)
        3. Apply per-joint rotation OFFSETS to align joint frames
        4. Make L-Wrist, R-Wrist, Neck relative to Root (both position and orientation)
        5. Return only the 3 non-root keypoints

    Note: Position calibration (wrist offsets, neck kinematic chain) is done in
          ThreePointPose.apply_calibration() to ensure consistency with calibrated
          orientations.
    """

    # Defensive copy: _compute_rel_transform modifies pose[:3] in-place, which would
    # corrupt the caller's array (e.g. PicoReader._latest) and cause wrong results
    # if the same sample is processed more than once.
    smpl_pose_np = smpl_pose_np.copy()

    # =========================================================================
    # STEP 1: Transform all joints from Unity frame to robot frame
    # =========================================================================
    # Input: smpl_pose_np[i] = [x, y, z, qx, qy, qz, qw] in Unity frame (scalar-last)
    # Output: body_poses[i] = [x, y, z, qw, qx, qy, qz] in robot frame (scalar-first)
    body_poses = np.zeros((smpl_pose_np.shape[0], 7), dtype=np.float32)
    for i in range(smpl_pose_np.shape[0]):
        pos, orn = _compute_rel_transform(
            smpl_pose_np[i], [0, 0, 0, 0, 0, 0, 1], scalar_first=False
        )
        body_poses[i, :3] = pos  # Position in robot frame
        body_poses[i, 3:] = orn  # Quaternion [qw, qx, qy, qz] in robot frame

    # =========================================================================
    # STEP 2 & 3: Extract 4 keypoints and apply rotation OFFSETS
    # =========================================================================
    # We only care about these SMPL joint indices:
    #   - Joint 0:  Root/Pelvis (reference frame)
    #   - Joint 22: Left Wrist
    #   - Joint 23: Right Wrist
    #   - Joint 12: Neck (more stable than Head joint 15)
    #
    # kp_poses maps these to indices 0, 1, 2, 3 respectively
    positions = np.array([[p[0], p[1], p[2]] for p in body_poses])
    kp_poses = np.zeros((4, 7), dtype=np.float32)

    for i, pose in enumerate(body_poses):
        if i not in [0, 22, 23, 12]:
            continue  # Skip joints we don't care about

        pos = positions[i]

        # Map SMPL joint index to our keypoint index (0-3)
        # rel_i: 0=Root, 1=L-Wrist, 2=R-Wrist, 3=Neck
        rel_i = [0, 22, 23, 12].index(i)

        # Extract quaternion and apply rotation offset
        # pose[3:7] is [qw, qx, qy, qz] (scalar-first from _compute_rel_transform)
        quat = np.array([pose[3], pose[4], pose[5], pose[6]])

        # Apply offset: new_rotation = original_rotation * OFFSET
        # This post-multiplies the offset (intrinsic rotation)
        rot_quat = (sRot.from_quat(quat, scalar_first=True) * OFFSETS[rel_i]).as_quat(
            scalar_first=False
        )

        kp_poses[rel_i, 3:] = rot_quat  # Store as scalar-last temporarily for scipy compatibility
        kp_poses[rel_i, :3] = pos

    # =========================================================================
    # STEP 4: Make positions and orientations RELATIVE TO ROOT
    # =========================================================================
    # This transforms everything into the root's local coordinate frame.
    # After this step:
    #   - Root's position would be (0,0,0) and orientation identity (but we don't return root)
    #   - Other keypoints are expressed relative to root
    root_pos = kp_poses[0, :3].copy()
    root_quat = kp_poses[0, 3:].copy()  # Still scalar-last for scipy

    for i in range(1, 4):
        # Position: subtract root position, then rotate by inverse of root orientation
        kp_poses[i, :3] = sRot.from_quat(root_quat).inv().apply(kp_poses[i, :3] - root_pos)

        # Orientation: compute relative rotation (root_inv * keypoint_rot)
        # Result stored as scalar-FIRST [qw, qx, qy, qz]
        kp_poses[i, 3:] = (
            sRot.from_quat(root_quat).inv() * sRot.from_quat(kp_poses[i, 3:])
        ).as_quat(scalar_first=True)

    # =========================================================================
    # STEP 5: Return only L-Wrist, R-Wrist, Neck (skip Root)
    # =========================================================================
    # NOTE: Position and orientation calibration (including neck position via kinematic
    #       chain) is done in ThreePointPose.apply_calibration() to ensure consistency
    #       between calibrated orientation and computed neck position.
    # kp_poses[1:] = indices 1, 2, 3 = L-Wrist, R-Wrist, Neck
    # Each row: [x, y, z, qw, qx, qy, qz] relative to root, scalar-first quaternion
    return kp_poses[1:]


# =============================================================================
# VR 3-Point Pose Visualization Functions
# =============================================================================


def run_vr3pt_visualizer_test():
    """
    Standalone test for VR 3-point pose visualizer using PyVista.
    Run this to verify the reference frames are displayed correctly.
    """
    if VR3PtPoseVisualizer is None:
        raise ImportError("VR3PtPoseVisualizer not available. Install pyvista: pip install pyvista")

    print("=" * 60)
    print("VR 3-Point Pose Visualizer Test (PyVista)")
    print("=" * 60)
    print("\nExpected reference frames (all with RGB axes for XYZ):")
    print("  1. WHITE ball at origin (0, 0, 0) - World frame")
    print("  2. CYAN ball at (0, 0, 0.35) - Looking forward (identity)")
    print("  3. MAGENTA ball at (0, 0.4, 0.25) - Looking left (yaw +90°)")
    print("  4. YELLOW ball at (0.4, 0, 0.15) - Looking down (pitch +90°)")
    print("\nClose the window to exit.")
    print("=" * 60)

    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)
    visualizer.show_static()


def run_vr3pt_live_visualizer():
    """
    Live visualizer for real VR 3-point pose data from Pico.
    Captures one frame from Pico and displays it alongside reference frames.
    """
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to use live visualizer."
        )

    if VR3PtPoseVisualizer is None:
        raise ImportError("VR3PtPoseVisualizer not available. Install pyvista: pip install pyvista")

    print("=" * 60)
    print("VR 3-Point Pose Live Visualizer (PyVista)")
    print("=" * 60)

    # Initialize XRT
    initialize_xr_source()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)

    print("Body data available! Capturing VR 3-point pose...")

    # Capture body poses and compute vr_3pt_pose
    body_poses = xrt.get_body_joints_pose()
    body_poses_np = np.array(body_poses)

    # Process to get 3-point pose (L-Wrist, R-Wrist, Neck)
    vr_3pt_pose = _process_3pt_pose(body_poses_np)

    print(f"\nCaptured vr_3pt_pose shape: {vr_3pt_pose.shape}")
    print(f"  L-Wrist: pos={vr_3pt_pose[0, :3]}, quat_wxyz={vr_3pt_pose[0, 3:]}")
    print(f"  R-Wrist: pos={vr_3pt_pose[1, :3]}, quat_wxyz={vr_3pt_pose[1, 3:]}")
    print(f"  Neck:    pos={vr_3pt_pose[2, :3]}, quat_wxyz={vr_3pt_pose[2, 3:]}")

    print("\nDisplaying visualization...")
    print("Close the window to exit.")
    print("=" * 60)

    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)
    visualizer.show_with_vr_pose(vr_3pt_pose)


def run_vr3pt_realtime_visualizer(update_hz: int = 10):
    """
    Real-time visualizer for VR 3-point pose data from Pico.
    Continuously updates the visualization with live data.

    Args:
        update_hz: Update rate in Hz (default 10)
    """
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to use realtime visualizer."
        )

    if VR3PtPoseVisualizer is None:
        raise ImportError("VR3PtPoseVisualizer not available. Install pyvista: pip install pyvista")

    print("=" * 60)
    print("VR 3-Point Pose Real-time Visualizer (PyVista)")
    print("=" * 60)

    # Initialize XRT
    initialize_xr_source()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)

    print("Body data available! Starting real-time visualization...")
    print(f"Update rate: {update_hz} Hz")
    print("Close the window or press 'q' to exit.")
    print("=" * 60)

    # Use the VR3PtPoseVisualizer for real-time visualization with G1 robot
    visualizer = VR3PtPoseVisualizer(axis_length=0.08, ball_radius=0.015, with_g1_robot=True)
    visualizer.create_realtime_plotter(interactive=True)

    try:
        while visualizer.is_open:
            # Get new data from Pico
            body_poses = xrt.get_body_joints_pose()
            body_poses_np = np.array(body_poses)
            vr_3pt_pose = _process_3pt_pose(body_poses_np)

            # Update visualization
            visualizer.update_vr_poses(vr_3pt_pose)
            visualizer.render()

            time.sleep(1.0 / update_hz)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        visualizer.close()


def process_smpl_joints(body_pose, global_orient, transl):
    """Process SMPL parameters to compute local joints.

    Args:
        body_pose: Body pose tensor, shape (T, 69)
        global_orient: Global orientation tensor, shape (T, 3)
        transl: Translation tensor, shape (T, 3)

    Returns:
        Dictionary with processed joints and parameters
    """
    # Convert global_orient to quaternion and apply transformations (robust if utils missing)
    global_orient_quat = angle_axis_to_quaternion(global_orient)
    if smpl_root_ytoz_up is not None:
        global_orient_quat = smpl_root_ytoz_up(global_orient_quat)
    global_orient_new = quaternion_to_angle_axis(global_orient_quat)

    # Compute joints and vertices using SMPL model (single forward pass)
    joints = compute_human_joints(
        body_pose=body_pose[..., :63],
        global_orient=global_orient_new,
    )  # (*, 24, 3)
    joints = joints - joints[..., :1, :]

    # Apply base rotation removal and compute local joints
    if remove_smpl_base_rot is not None:
        global_orient_quat = remove_smpl_base_rot(global_orient_quat, w_last=False)
    smpl_root_quat = global_orient_quat.clone()

    global_orient_quat_inv = quat_inv(global_orient_quat).unsqueeze(1).repeat(1, joints.shape[1], 1)
    smpl_joints_local = quat_apply(global_orient_quat_inv, joints)
    global_orient_mat = quaternion_to_rotation_matrix(global_orient_quat)
    global_orient_6d = global_orient_mat[..., :2].reshape(1, 6)

    return {
        "smpl_pose": body_pose,
        "joints": joints,
        "smpl_joints_local": smpl_joints_local,
        "smpl_root_quat": smpl_root_quat,
        "global_orient_quat": global_orient_quat,
        "global_orient_6d": global_orient_6d,
        "adjusted_transl": transl,
    }


def generate_finger_data(hand: str, trigger: float, grip: float) -> np.ndarray:
    """
    Generate finger position data from Pico controller button states.

    Args:
        hand: "left" or "right"
        trigger: Trigger button value (0-1)
        grip: Grip button value (0-1)

    Returns:
        Array of shape [25, 4, 4] representing fingertip positions
    """
    fingertips = np.zeros([25, 4, 4])

    thumb = 0
    middle = 10
    # Control thumb based on shoulder button state (index 4 is thumb tip)
    fingertips[4 + thumb, 0, 3] = 1.0  # open thumb
    if trigger > 0.5:
        fingertips[4 + middle, 0, 3] = 1.0  # close middle

    return fingertips


# Joystick deadzone threshold
JOYSTICK_DEADZONE = 0.15
DEFAULT_PLANNER_SPEED_SCALE = 0.60
DEFAULT_PLANNER_MAX_SPEED = 0.70


def planner_speed_command(
    mode: LocomotionMode,
    joystick_magnitude: float,
    speed_scale: float,
    max_speed: float,
) -> float:
    """Return an explicit, limited planner speed for a moving command."""
    magnitude = float(np.clip(joystick_magnitude, 0.0, 1.0))
    if not 0.0 < speed_scale <= 1.0:
        raise ValueError(f"planner speed scale must be in (0, 1], got {speed_scale}")
    if max_speed <= 0.0:
        raise ValueError(f"planner max speed must be positive, got {max_speed}")

    if mode == LocomotionMode.SLOW_WALK:
        base_speed = 0.1 + 0.65 * magnitude
    elif mode == LocomotionMode.WALK:
        # WALK previously sent -1 and therefore used the planner's much faster default.
        base_speed = 0.4 + 0.6 * magnitude
    elif mode == LocomotionMode.RUN:
        base_speed = 1.5 + 3.0 * magnitude
    else:
        base_speed = magnitude
    return min(base_speed * speed_scale, max_speed)


class YawAccumulator:
    """Accumulates yaw heading angle based on joystick input."""

    def __init__(self, yaw_gain: float = 1.5, deadzone: float = JOYSTICK_DEADZONE):
        self.yaw_gain = yaw_gain
        self.deadzone = deadzone
        self.reset()

    def reset(self):
        """Reset facing direction to default (1,0,0)."""
        self.heading = [1.0, 0.0, 0.0]
        self.yaw_angle_rad = 0.0
        self.dyaw = 0.0
        print("YawAccumulator: reset yaw angle to 0.0")

    def yaw_angle(self) -> float:
        """Get current yaw angle in radians."""
        return self.yaw_angle_rad

    def yaw_angle_change(self) -> float:
        """Get current yaw angle change in radians."""
        return self.dyaw

    def update(self, rx: float, dt: float) -> list[float]:
        """
        Update facing direction based on right stick x-axis input.

        Args:
            rx: Right stick x-axis value (-1 to 1)
            dt: Time delta in seconds

        Returns:
            Facing direction as [x, y, 0.0]
        """
        self.dyaw = self.yaw_gain * (-rx) * dt
        if abs(rx) >= self.deadzone:
            self.yaw_angle_rad += self.dyaw
            self.heading = [np.cos(self.yaw_angle_rad), np.sin(self.yaw_angle_rad), 0.0]
        return self.heading


SMPL_BALANCE_NEUTRAL_JOINTS = tuple(range(1, 12))


def _smpl_local_rotations_from_body_poses(
    parent_indices: list[int],
    body_poses_np: np.ndarray,
    joint_frame_quats_wxyz: np.ndarray | None = None,
) -> list[sRot]:
    """Convert PICO global joint frames to SMPL local joint rotations."""
    global_rots = corrected_xrt_global_rotations(
        body_poses_np, joint_frame_quats_wxyz
    )

    local_rots = []
    for i in range(24):
        if parent_indices[i] == -1:
            local_rots.append(global_rots[i])
        else:
            local_rots.append(global_rots[parent_indices[i]].inv() * global_rots[i])
    return local_rots


def capture_smpl_balance_neutral_quats(
    parent_indices: list[int],
    body_poses_np: np.ndarray,
    joint_frame_quats_wxyz: np.ndarray | None = None,
) -> np.ndarray:
    """Capture the operator's current lower-body and torso pose as neutral."""
    local_rots = _smpl_local_rotations_from_body_poses(
        parent_indices, body_poses_np, joint_frame_quats_wxyz
    )
    return np.asarray(
        [rotation.as_quat(scalar_first=True) for rotation in local_rots],
        dtype=np.float64,
    )


def compute_from_body_poses(
    parent_indices: list,
    device,
    body_poses_np: np.ndarray,
    balance_neutral_quats: np.ndarray | None = None,
    joint_frame_quats_wxyz: np.ndarray | None = None,
):
    """
    Compute local joints and body orientation from provided body_poses_np.
    """
    positions = body_poses_np[:, :3]
    local_rots = _smpl_local_rotations_from_body_poses(
        parent_indices, body_poses_np, joint_frame_quats_wxyz
    )
    if balance_neutral_quats is not None:
        neutral_quats = np.asarray(balance_neutral_quats, dtype=np.float64)
        if neutral_quats.shape != (24, 4):
            raise ValueError(
                f"SMPL balance neutral quaternions must have shape (24, 4), "
                f"got {neutral_quats.shape}"
            )
        for joint_idx in SMPL_BALANCE_NEUTRAL_JOINTS:
            neutral_rot = sRot.from_quat(neutral_quats[joint_idx], scalar_first=True)
            local_rots[joint_idx] = neutral_rot.inv() * local_rots[joint_idx]

    pose_aa = np.array([rot.as_rotvec() for rot in local_rots])

    body_pose = torch.from_numpy(pose_aa[1:].flatten()).float().to(device).unsqueeze(0)
    global_orient = torch.from_numpy(pose_aa[0]).float().to(device).unsqueeze(0)
    transl = torch.from_numpy(positions[0]).float().to(device).unsqueeze(0)

    return process_smpl_joints(body_pose, global_orient, transl)


# def compute_latest_frame(parent_indices: list, device) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Pull body data from XRoboToolkit, compute local SMPL joints and body orientation.
#     Returns (smpl_joints_local_np [24,3], global_orient_quat_np [4,])
#     """
#     body_poses = xrt.get_body_joints_pose()
#     body_poses_np = np.array(body_poses)
#     return compute_from_body_poses(parent_indices, device, body_poses_np)


def init_hand_ik_solvers():
    """Initialize hand IK solvers if available."""
    if G1GripperInverseKinematicsSolver is not None:
        left_solver = G1GripperInverseKinematicsSolver(side="left")
        right_solver = G1GripperInverseKinematicsSolver(side="right")
        print("Hand IK solvers initialized")
        return left_solver, right_solver
    print("Warning: Hand IK solvers not available")
    return None, None


def get_controller_inputs():
    """Fetch controller button/trigger states from XRoboToolkit."""
    left_trigger = xrt.get_left_trigger()
    right_trigger = xrt.get_right_trigger()
    left_grip = xrt.get_left_grip()
    right_grip = xrt.get_right_grip()
    left_menu_button = xrt.get_left_menu_button()
    return left_menu_button, left_trigger, right_trigger, left_grip, right_grip


def get_controller_axes():
    """Fetch joystick axes (lx, ly, rx, ry). Falls back to zeros if not available."""
    if xrt is None:
        return 0.0, 0.0, 0.0, 0.0
    try:
        left_axis = xrt.get_left_axis()  # expected [x, y]
        right_axis = xrt.get_right_axis()  # expected [x, y]
        lx = float(left_axis[0]) if len(left_axis) >= 1 else 0.0
        ly = float(left_axis[1]) if len(left_axis) >= 2 else 0.0
        rx = float(right_axis[0]) if len(right_axis) >= 1 else 0.0
        ry = float(right_axis[1]) if len(right_axis) >= 2 else 0.0
        return lx, ly, rx, ry
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def get_menu_buttons():
    """Fetch both menu buttons (left, right). Falls back to False if not available."""
    if xrt is None:
        return False, False

    def _safe_btn(attr):
        try:
            fn = getattr(xrt, attr)
            return bool(fn())
        except Exception:
            return False

    left = _safe_btn("get_left_menu_button")
    right = _safe_btn("get_right_menu_button")
    return left, right


def get_axis_clicks():
    """Fetch both axis click buttons (left, right). Falls back to False if not available."""
    if xrt is None:
        return False, False

    def _safe_btn(attr):
        try:
            fn = getattr(xrt, attr)
            return bool(fn())
        except Exception:
            return False

    left = _safe_btn("get_left_axis_click")
    right = _safe_btn("get_right_axis_click")
    return left, right


def get_face_buttons():
    """Fetch primary face buttons A and X. Returns (a_pressed, x_pressed)."""
    if xrt is None:
        return False, False
    try:
        a_pressed = bool(xrt.get_A_button())
        x_pressed = bool(xrt.get_X_button())
        return a_pressed, x_pressed
    except Exception:
        return False, False


def get_abxy_buttons():
    """Fetch A,B,X,Y face buttons as booleans (a,b,x,y)."""
    if xrt is None:
        return False, False, False, False
    try:
        a_pressed = bool(xrt.get_A_button())
        b_pressed = bool(xrt.get_B_button())
        x_pressed = bool(xrt.get_X_button())
        y_pressed = bool(xrt.get_Y_button())
        return a_pressed, b_pressed, x_pressed, y_pressed
    except Exception:
        return False, False, False, False


def compute_hand_joints_from_inputs(
    left_solver, right_solver, left_trigger, left_grip, right_trigger, right_grip
) -> tuple[np.ndarray, np.ndarray]:
    """Compute left/right hand joints using IK solvers, or zeros if unavailable."""
    if left_solver is not None and right_solver is not None:
        left_finger_data = generate_finger_data("left", left_trigger, left_grip)
        right_finger_data = generate_finger_data("right", right_trigger, right_grip)
        left_hand_joints = left_solver({"position": left_finger_data})
        right_hand_joints = right_solver({"position": right_finger_data})
    else:
        left_hand_joints = np.zeros((1, 7), dtype=np.float32)
        right_hand_joints = np.zeros((1, 7), dtype=np.float32)
    return left_hand_joints, right_hand_joints


def _quat_lerp_normalized(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """
    Linear interpolate two same-order quaternions and renormalize. Input shape (4,).
    Ensures shortest path by flipping sign if dot < 0.
    """
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
    q = (1.0 - alpha) * q0 + alpha * q1
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    return q


def _interp_pose_axis_angle(
    prev_pose: np.ndarray, curr_pose: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Interpolate axis-angle joint poses by converting to quats, lerp-normalize, then back.
    prev_pose, curr_pose: (21,3) axis-angle (rotvec)
    Returns (21,3) axis-angle.
    """
    prev_quats = sRot.from_rotvec(prev_pose.reshape(-1, 3)).as_quat()  # (N,4) xyzw
    curr_quats = sRot.from_rotvec(curr_pose.reshape(-1, 3)).as_quat()
    out_quats = np.empty_like(prev_quats)
    for i in range(prev_quats.shape[0]):
        out_quats[i] = _quat_lerp_normalized(prev_quats[i], curr_quats[i], alpha)
    out_pose = sRot.from_quat(out_quats).as_rotvec().reshape(prev_pose.shape)
    return out_pose


class PicoReader:
    """
    Background reader that pulls Pico/XRT data as fast as possible and computes dt/FPS.
    """

    def __init__(self, max_queue_size: int = 15):
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_t = None
        self._fps_ema = 0.0
        self._last_stamp_ns = None
        self._latest = None
        self._lock = threading.Lock()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def get_latest(self):
        with self._lock:
            return self._latest

    def _run(self):
        last_report = time.time()
        while not self._stop.is_set():
            if not xr_tracking_data_available():
                time.sleep(0.001)
                continue
            stamp_ns = xrt.get_time_stamp_ns()
            prev_stamp_ns = self._last_stamp_ns
            if prev_stamp_ns is not None and stamp_ns == prev_stamp_ns:
                time.sleep(0.000001)
                continue
            # Compute device-based dt/fps using timestamp deltas (ns -> s)
            device_dt = ((stamp_ns - prev_stamp_ns) * 1e-9) if prev_stamp_ns is not None else 0.0
            if device_dt > 0.0:
                inst = 1.0 / device_dt
                self._fps_ema = inst if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * inst)
            self._last_stamp_ns = stamp_ns
            t_realtime = time.time()
            t_monotonic = time.monotonic()
            try:
                sample = {
                    "timestamp_realtime": t_realtime,
                    "timestamp_monotonic": t_monotonic,
                    "timestamp_ns": stamp_ns,
                    "dt": device_dt,
                    "fps": self._fps_ema,
                    "right_stick_y": get_controller_axes()[3],
                }
                if is_direct_offline_vr():
                    recorded_poses = xrt.get_vr_3pt_poses()
                    sample["headset_pose"] = np.asarray(recorded_poses[2], dtype=np.float64)
                    sample["vr_3pt_pose_raw"] = _process_xrt_3pt_pose(
                        recorded_poses[0], recorded_poses[1], recorded_poses[2]
                    )
                else:
                    sample["body_poses_np"] = np.asarray(xrt.get_body_joints_pose())
                    sample["headset_pose"] = np.asarray(xrt.get_headset_pose(), dtype=np.float64)
                with self._lock:
                    self._latest = sample
                now = time.time()
                if now - last_report >= 5.0:
                    print(
                        f"[PicoReader] dt_ts: {device_dt*1000.0:.2f} ms, fps: {self._fps_ema:.2f}"
                    )
                    last_report = now
            except Exception as e:
                print(f"[PicoReader] read error: {e}")


def _pose_stream_common(
    socket,
    buffer_size: int,
    num_frames_to_send: int,
    target_fps: int,
    use_cuda: bool,
    record_dir: str,
    record_format: str,
    stop_event: threading.Event | None = None,
    log_prefix: str = "PoseLoop",
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
    smpl_wrist_source: str = "g1",
    smpl_arm_source: str = "rotation",
    smpl_vis_source: str = "sent",
    smpl_joint_frame_calibration: str = "",
    vr3pt_robot_profile: str = "g1",
):
    """Shared pose streaming loop used by run_pico."""
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run pose streaming."
        )

    # Create reader and start it
    reader = PicoReader(max_queue_size=buffer_size)
    reader.start()

    # Create 3-point pose processor with visualization settings
    three_point = ThreePointPose(
        enable_vis_vr3pt=enable_vis_vr3pt,
        with_g1_robot=with_g1_robot,
        enable_waist_tracking=enable_waist_tracking,
        enable_smpl_vis=enable_smpl_vis,
        log_prefix=log_prefix,
        robot_profile=vr3pt_robot_profile,
    )

    streamer = PoseStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        num_frames_to_send=num_frames_to_send,
        target_fps=target_fps,
        use_cuda=use_cuda,
        record_dir=record_dir,
        record_format=record_format,
        log_prefix=log_prefix,
        smpl_wrist_source=smpl_wrist_source,
        smpl_arm_source=smpl_arm_source,
        smpl_vis_source=smpl_vis_source,
        smpl_joint_frame_calibration=smpl_joint_frame_calibration,
    )

    if stop_event is None:
        stop_event = threading.Event()

    try:
        while not stop_event.is_set():
            streamer.run_once()
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup resources
        reader.stop()
        three_point.close()


class ThreePointPose:
    """
    Encapsulates everything around calculating 3-point pose from SMPL input.

    This includes:
    - Processing SMPL poses to extract 3-point VR pose (L-Wrist, R-Wrist, Neck)
    - Calibration logic to align VR poses with G1 robot
    - Optional visualization of 3-point poses

    Calibration is done in two steps:
    1. Neck orientation: Captures initial neck orientation to align subsequent poses as upright
    2. Wrist positions: Aligns wrist positions to match G1 robot key frame positions
    """

    # Kinematic chain constants for neck position (matches VR3PtPoseVisualizer)
    TORSO_LINK_OFFSET_Z = 0.05  # meters from root to torso_link
    NECK_LINK_LENGTH = 0.35  # meters from torso_link to neck along neck's local Z

    def __init__(
        self,
        enable_vis_vr3pt: bool = False,
        with_g1_robot: bool = True,
        enable_waist_tracking: bool = False,
        enable_smpl_vis: bool = False,
        log_prefix: str = "ThreePointPose",
        robot_model=None,
        robot_profile: str = "g1",
    ):
        """
        Initialize 3-point pose processor.

        Args:
            enable_vis_vr3pt: Whether to enable VR 3pt pose visualization (requires display)
            with_g1_robot: Whether to include G1 robot in visualization
            enable_waist_tracking: Whether to enable waist tracking in visualization
            enable_smpl_vis: Whether to render SMPL body joints in the VR3pt visualizer
            log_prefix: Prefix for log messages
            robot_model: Optional pre-instantiated RobotModel. If None, will create one.
                        Used for FK-based calibration (no display required).
            robot_profile: FK calibration geometry, either ``g1`` or ``h2-native``.
        """
        self.log_prefix = log_prefix
        self.with_g1_robot = with_g1_robot
        self.enable_waist_tracking = enable_waist_tracking
        self.enable_smpl_vis = enable_smpl_vis
        if robot_profile not in {"g1", "h2-native"}:
            raise ValueError(f"Unsupported VR3Point robot profile: {robot_profile}")
        self.robot_profile = robot_profile

        # Robot model for FK-based calibration (headless, no display required)
        self._robot_model = robot_model
        if self._robot_model is None:
            if self.robot_profile == "h2-native":
                from interface.kinematics.h2 import (
                    instantiate_h2_robot_model,
                )

                self._robot_model = instantiate_h2_robot_model()
            else:
                from interface.kinematics.g1 import (
                    instantiate_g1_robot_model,
                )

                self._robot_model = instantiate_g1_robot_model()
            print(
                f"[{log_prefix}] {self.robot_profile} robot model loaded for FK calibration"
            )

        # Optional visualization (requires display + PyVista)
        self.vr3pt_visualizer = None
        if enable_vis_vr3pt:
            if VR3PtPoseVisualizer is None:
                raise ImportError(
                    "VR3PtPoseVisualizer could not be imported but --vis_vr3pt was requested. "
                    "Ensure pyvista is installed: pip install pyvista"
                )
            self.vr3pt_visualizer = VR3PtPoseVisualizer(
                axis_length=0.08,
                ball_radius=0.015,
                with_g1_robot=with_g1_robot,
                robot_model=self._robot_model,
                enable_waist_tracking=enable_waist_tracking,
                enable_smpl_vis=enable_smpl_vis,
            )
            self.vr3pt_visualizer.create_realtime_plotter(interactive=True)
            g1_str = " with G1 robot" if with_g1_robot else ""
            waist_str = " + waist tracking" if enable_waist_tracking else ""
            smpl_str = " + SMPL body" if enable_smpl_vis else ""
            print(f"[{log_prefix}] VR 3pt pose visualization enabled{g1_str}{waist_str}{smpl_str}")

        # Calibration state — triggered explicitly by calibrate_now() or reset_with_measured_q()
        self._calibration_pending = False
        self._calibration_neck_quat_inv: np.ndarray | None = None  # inv(initial neck quat)
        self._calibration_lwrist_offset: np.ndarray | None = None  # position offset
        self._calibration_rwrist_offset: np.ndarray | None = None
        self._calibration_lwrist_rot_offset: sRot | None = None  # orientation offset
        self._calibration_rwrist_rot_offset: sRot | None = None
        self._calibration_neck_rot_offset: sRot | None = None
        self._calibration_torso_origin: np.ndarray | None = None
        # Override robot q for FK during recalibration (e.g. measured joints for VR 3PT)
        self._override_robot_q: np.ndarray | None = None

    @property
    def is_pending(self) -> bool:
        """Check if calibration is pending."""
        return self._calibration_pending

    @property
    def is_calibrated(self) -> bool:
        """Check if calibration has been captured."""
        return self._calibration_neck_quat_inv is not None

    def process_smpl_pose(
        self,
        smpl_pose_np: np.ndarray,
        smpl_joints_local: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Process SMPL pose to extract and calibrate 3-point VR pose.

        Args:
            smpl_pose_np: np.ndarray shape (24, 7) - 24 SMPL joints
            smpl_joints_local: Optional np.ndarray shape (24, 3) - SMPL local joint
                               positions for body visualization. If provided and SMPL
                               visualization is enabled, the joint spheres are updated.

        Returns:
            vr_3pt_pose: np.ndarray shape (3, 7) - Calibrated 3-point pose
                         [L-Wrist, R-Wrist, Neck], each row [x, y, z, qw, qx, qy, qz]
        """
        # Extract raw 3-point pose from SMPL
        vr_3pt_pose_raw = _process_3pt_pose(smpl_pose_np)

        return self._process_vr_pose(vr_3pt_pose_raw, smpl_joints_local=smpl_joints_local)

    def process_vr_pose(self, vr_3pt_pose_raw: np.ndarray) -> np.ndarray:
        """Calibrate a direct headset/controller three-point sample (no SMPL input)."""
        vr_3pt_pose_raw = np.asarray(vr_3pt_pose_raw, dtype=np.float32)
        if vr_3pt_pose_raw.shape != (3, 7):
            raise ValueError(
                f"direct VR three-point pose must have shape (3, 7), got {vr_3pt_pose_raw.shape}"
            )
        return self._process_vr_pose(vr_3pt_pose_raw)

    def _process_vr_pose(
        self,
        vr_3pt_pose_raw: np.ndarray,
        *,
        smpl_joints_local: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._calibration_pending:
            self._capture_calibration(vr_3pt_pose_raw)

        # Apply calibration to get the final pose
        vr_3pt_pose = self._apply_calibration(vr_3pt_pose_raw)

        if self.vr3pt_visualizer is not None:
            self.vr3pt_visualizer.update_from_vr_pose(vr_3pt_pose, waist_scale=1.0)
            if smpl_joints_local is not None:
                self.vr3pt_visualizer.update_smpl_joints(smpl_joints_local)
            self.vr3pt_visualizer.render()

        return vr_3pt_pose

    def close(self) -> None:
        """Close and cleanup visualizer resources."""
        if self.vr3pt_visualizer is not None:
            try:
                self.vr3pt_visualizer.close()
            except Exception as e:
                print(f"[{self.log_prefix}] Warning: Error closing VR3pt visualizer: {e}")

    def calibrate_now(self, body_poses_np: np.ndarray) -> bool:
        """Calibrate using current SMPL frame against the robot reference pose.
        Operator should be in zero-reference pose when calling this."""
        try:
            vr_3pt_pose_raw = _process_3pt_pose(body_poses_np)
            self._override_robot_q = (
                None
                if self.robot_profile == "h2-native"
                else np.zeros(29, dtype=np.float64)
            )
            self._capture_calibration(vr_3pt_pose_raw)
            print(
                f"[{self.log_prefix}] Calibration completed "
                f"({self.robot_profile} reference)"
            )
            return True
        except Exception as e:
            print(f"[{self.log_prefix}] Calibration failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def calibrate_vr_now(self, vr_3pt_pose_raw: np.ndarray) -> bool:
        """Calibrate directly from a recorded headset/controller three-point frame."""
        try:
            vr_3pt_pose_raw = np.asarray(vr_3pt_pose_raw, dtype=np.float32)
            if vr_3pt_pose_raw.shape != (3, 7):
                raise ValueError(
                    f"direct VR three-point pose must have shape (3, 7), got {vr_3pt_pose_raw.shape}"
                )
            self._override_robot_q = (
                None
                if self.robot_profile == "h2-native"
                else np.zeros(29, dtype=np.float64)
            )
            self._capture_calibration(vr_3pt_pose_raw)
            print(
                f"[{self.log_prefix}] Direct VR calibration completed "
                f"({self.robot_profile} reference)"
            )
            return True
        except Exception as e:
            print(f"[{self.log_prefix}] Direct VR calibration failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _capture_calibration(self, vr_3pt_pose: np.ndarray) -> None:
        """Capture calibration offsets against the selected robot's FK reference.
        If neck calibration already exists (e.g. from calibrate_now), it is preserved
        to avoid jumps from SMPL noise during recalibration."""

        # Step 1: Neck orientation — only capture if not already set
        if self._calibration_neck_quat_inv is None:
            neck_quat_wxyz = vr_3pt_pose[2, 3:].copy()
            neck_rot = sRot.from_quat(neck_quat_wxyz, scalar_first=True)
            self._calibration_neck_quat_inv = neck_rot.inv().as_quat(scalar_first=True)
        calib_inv_rot = sRot.from_quat(self._calibration_neck_quat_inv, scalar_first=True)

        # Step 2: Rotate VR wrist positions/orientations by neck inverse
        lwrist_pos_corrected = calib_inv_rot.apply(vr_3pt_pose[0, :3].copy())
        rwrist_pos_corrected = calib_inv_rot.apply(vr_3pt_pose[1, :3].copy())
        lwrist_rot_corrected = calib_inv_rot * sRot.from_quat(vr_3pt_pose[0, 3:], scalar_first=True)
        rwrist_rot_corrected = calib_inv_rot * sRot.from_quat(vr_3pt_pose[1, 3:], scalar_first=True)

        # Step 3: Get robot-specific FK reference poses
        if self._robot_model is None:
            raise RuntimeError(
                "Robot model is required for calibration but was not loaded. "
                "Ensure the selected robot model and URDF are available."
            )
        if get_robot_key_frame_poses is None:
            raise RuntimeError(
                "Robot key-frame FK could not be imported. "
                "Ensure interface.teleop.visualization is available."
            )

        # Convert deploy's 29D feedback contract to the selected model configuration.
        if self._override_robot_q is not None:
            if self.robot_profile == "h2-native":
                robot_q = h2_configuration_from_g1_compatible_feedback(
                    self._robot_model, self._override_robot_q
                )
            else:
                robot_q = self._robot_model.get_configuration_from_actuated_joints(
                    body_actuated_joint_values=self._override_robot_q[:29]
                )
        else:
            robot_q = None
        robot_poses = get_robot_key_frame_poses(self._robot_model, q=robot_q)

        robot_lwrist_pos = robot_poses["left_wrist"]["position"]
        robot_rwrist_pos = robot_poses["right_wrist"]["position"]
        robot_lwrist_rot = sRot.from_quat(
            robot_poses["left_wrist"]["orientation_wxyz"], scalar_first=True
        )
        robot_rwrist_rot = sRot.from_quat(
            robot_poses["right_wrist"]["orientation_wxyz"], scalar_first=True
        )

        # Compute position offsets: calibrated = neck_corrected - offset
        self._calibration_lwrist_offset = lwrist_pos_corrected - robot_lwrist_pos
        self._calibration_rwrist_offset = rwrist_pos_corrected - robot_rwrist_pos

        # Compute orientation offsets: calibrated = rot_offset * neck_corrected
        self._calibration_lwrist_rot_offset = robot_lwrist_rot * lwrist_rot_corrected.inv()
        self._calibration_rwrist_rot_offset = robot_rwrist_rot * rwrist_rot_corrected.inv()

        if self.robot_profile == "h2-native":
            torso_pose = get_robot_key_frame_poses(
                self._robot_model, q=robot_q, apply_offset=False
            )["torso"]
            torso_rot = sRot.from_quat(
                torso_pose["orientation_wxyz"], scalar_first=True
            )
            neck_rot_corrected = calib_inv_rot * sRot.from_quat(
                vr_3pt_pose[2, 3:], scalar_first=True
            )
            self._calibration_neck_rot_offset = (
                torso_rot * neck_rot_corrected.inv()
            )
            self._calibration_torso_origin = torso_pose["position"].copy()

        used_feedback_q = self._override_robot_q is not None
        self._calibration_pending = False
        self._override_robot_q = None

        # Log summary
        source = "feedback q" if used_feedback_q else "default"
        print(
            f"[{self.log_prefix}] Calibration captured "
            f"(robot={self.robot_profile}, FK ref={source}):\n"
            f"  L-Wrist pos offset: [{self._calibration_lwrist_offset[0]:.4f}, "
            f"{self._calibration_lwrist_offset[1]:.4f}, {self._calibration_lwrist_offset[2]:.4f}]\n"
            f"  R-Wrist pos offset: [{self._calibration_rwrist_offset[0]:.4f}, "
            f"{self._calibration_rwrist_offset[1]:.4f}, {self._calibration_rwrist_offset[2]:.4f}]"
        )

    def _apply_calibration(self, vr_3pt_pose: np.ndarray) -> np.ndarray:
        """Apply stored calibration offsets to raw VR 3-point pose."""
        if self._calibration_neck_quat_inv is None:
            return vr_3pt_pose

        calibrated = vr_3pt_pose.copy()
        calib_inv_rot = sRot.from_quat(self._calibration_neck_quat_inv, scalar_first=True)

        # Neck orientation: calibrated = robot-frame offset * inv(initial) * current
        neck_rot = sRot.from_quat(vr_3pt_pose[2, 3:], scalar_first=True)
        neck_rot_corrected = calib_inv_rot * neck_rot
        if self._calibration_neck_rot_offset is not None:
            neck_rot_corrected = self._calibration_neck_rot_offset * neck_rot_corrected
        calibrated[2, 3:] = neck_rot_corrected.as_quat(scalar_first=True)

        # Wrist positions: rotate by neck inverse, then subtract offset
        if self._calibration_lwrist_offset is not None:
            calibrated[0, :3] = (
                calib_inv_rot.apply(vr_3pt_pose[0, :3]) - self._calibration_lwrist_offset
            )
        if self._calibration_rwrist_offset is not None:
            calibrated[1, :3] = (
                calib_inv_rot.apply(vr_3pt_pose[1, :3]) - self._calibration_rwrist_offset
            )

        # Wrist orientations: rot_offset * (neck_inv * current)
        if self._calibration_lwrist_rot_offset is not None:
            lw_corrected = calib_inv_rot * sRot.from_quat(vr_3pt_pose[0, 3:], scalar_first=True)
            calibrated[0, 3:] = (self._calibration_lwrist_rot_offset * lw_corrected).as_quat(
                scalar_first=True
            )
        if self._calibration_rwrist_rot_offset is not None:
            rw_corrected = calib_inv_rot * sRot.from_quat(vr_3pt_pose[1, 3:], scalar_first=True)
            calibrated[1, 3:] = (self._calibration_rwrist_rot_offset * rw_corrected).as_quat(
                scalar_first=True
            )

        # Neck position via robot torso origin + the training-time local offset.
        neck_z = sRot.from_quat(calibrated[2, 3:], scalar_first=True).apply([0, 0, 1])
        torso_origin = (
            self._calibration_torso_origin
            if self._calibration_torso_origin is not None
            else np.array([0, 0, self.TORSO_LINK_OFFSET_Z])
        )
        calibrated[2, :3] = (torso_origin + self.NECK_LINK_LENGTH * neck_z).astype(
            np.float32
        )

        return calibrated

    def _clear_calibration(self):
        """Clear all calibration state."""
        self._calibration_neck_quat_inv = None
        self._calibration_lwrist_offset = None
        self._calibration_rwrist_offset = None
        self._calibration_lwrist_rot_offset = None
        self._calibration_rwrist_rot_offset = None
        self._calibration_neck_rot_offset = None
        self._calibration_torso_origin = None
        self._override_robot_q = None

    def reset(self) -> None:
        """Reset calibration. Next process_smpl_pose() call will recalibrate."""
        self._clear_calibration()
        self._calibration_pending = True
        print(f"[{self.log_prefix}] Calibration reset, will re-calibrate on next frame")

    def reset_with_measured_q(self, body_q_measured: np.ndarray) -> None:
        """Recalibrate wrist offsets using measured robot joints (29 DOFs).
        Preserves neck calibration to avoid jumps from SMPL noise.
        Next process_smpl_pose() will recompute wrist offsets against FK of these joints."""
        # Preserve neck calibration — only clear wrist offsets
        self._calibration_lwrist_offset = None
        self._calibration_rwrist_offset = None
        self._calibration_lwrist_rot_offset = None
        self._calibration_rwrist_rot_offset = None
        self._calibration_neck_rot_offset = None
        self._calibration_torso_origin = None
        self._override_robot_q = body_q_measured.copy()
        self._calibration_pending = True
        print(f"[{self.log_prefix}] Wrist recalibration pending (neck preserved, measured q)")

    def reset_with_default_q(self) -> None:
        """Schedule calibration against the selected robot's default configuration."""
        self._calibration_lwrist_offset = None
        self._calibration_rwrist_offset = None
        self._calibration_lwrist_rot_offset = None
        self._calibration_rwrist_rot_offset = None
        self._calibration_neck_rot_offset = None
        self._calibration_torso_origin = None
        self._override_robot_q = None
        self._calibration_pending = True
        print(f"[{self.log_prefix}] Recalibration pending (default {self.robot_profile} q)")


class PoseStreamer:
    """Stream consecutive SMPL observations from Pico in encoder mode 2."""

    def __init__(
        self,
        socket,
        reader: PicoReader,
        three_point: ThreePointPose,
        num_frames_to_send: int,
        target_fps: int,
        use_cuda: bool,
        record_dir: str,
        record_format: str,
        log_prefix: str = "PoseLoop",
        smpl_wrist_source: str = "g1",
        smpl_arm_source: str = "rotation",
        smpl_vis_source: str = "sent",
        smpl_joint_frame_calibration: str = "",
    ):
        self.socket = socket
        self.reader = reader
        if num_frames_to_send < 10:
            raise ValueError("SMPL mode-2 pose windows require at least 10 consecutive frames")
        self.num_frames_to_send = num_frames_to_send
        self.target_fps = target_fps
        self.record_dir = record_dir
        self.log_prefix = log_prefix
        self.three_point = three_point
        if smpl_wrist_source not in {"g1", "h2-kinematic"}:
            raise ValueError(f"Unsupported SMPL wrist source: {smpl_wrist_source}")
        self.smpl_wrist_source = smpl_wrist_source
        if smpl_arm_source not in {"rotation", "xrt-position"}:
            raise ValueError(f"Unsupported SMPL arm source: {smpl_arm_source}")
        self.smpl_arm_source = smpl_arm_source
        if smpl_vis_source not in {"sent", "xrt"}:
            raise ValueError(f"Unsupported SMPL visualization source: {smpl_vis_source}")
        self.smpl_vis_source = smpl_vis_source
        self.smpl_joint_frame_calibration = smpl_joint_frame_calibration
        self.joint_frame_quats_wxyz = (
            load_joint_frame_calibration(smpl_joint_frame_calibration)
            if smpl_joint_frame_calibration
            else None
        )
        self.device = (
            torch.device("cuda") if use_cuda and torch.cuda.is_available() else torch.device("cpu")
        )

        if record_dir:
            os.makedirs(record_dir, exist_ok=True)
        self.record_idx = 0

        self.left_hand_ik_solver, self.right_hand_ik_solver = init_hand_ik_solvers()
        self.parent_indices = [
            -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
            9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23,
        ][:24]

        self.step = 0
        self.last_fps_report = time.time()
        self.fps_counter = 0
        self.frame_time = 0.95 / max(1, target_fps)
        self.frame_buffer = defaultdict(lambda: deque(maxlen=num_frames_to_send))
        self.prev_stamp_ns = None
        self.prev_smpl_pose_np = None
        self.prev_smpl_joints_np = None
        self.prev_body_quat_np = None
        self.prev_smpl_root_quat_np = None
        self.next_target_ns = None
        self.frame_start = time.time()
        self.balance_neutral_quats: np.ndarray | None = None
        self.toggle_data_collection_last = False
        self.toggle_data_abort_last = False
        self.buffer_cleared = True
        self.yaw_accumulator = YawAccumulator()
        print(
            f"[{self.log_prefix}] Pico SMPL streaming enabled "
            f"(pose protocol v3, encoder mode 2, wrist={self.smpl_wrist_source}, "
            f"arms={self.smpl_arm_source}, vis={self.smpl_vis_source}, "
            f"joint_frames={self.smpl_joint_frame_calibration or 'original-y180'})"
        )

    def reset_yaw(self):
        """Called when entering pose mode. Resets yaw only."""
        self.yaw_accumulator.reset()

    def calibrate_balance_neutral(self, body_poses_np: np.ndarray) -> None:
        """Map the current PICO legs and torso to the SMPL neutral pose."""
        self.balance_neutral_quats = capture_smpl_balance_neutral_quats(
            self.parent_indices, body_poses_np, self.joint_frame_quats_wxyz
        )
        calibrated = compute_from_body_poses(
            self.parent_indices,
            self.device,
            body_poses_np,
            balance_neutral_quats=self.balance_neutral_quats,
            joint_frame_quats_wxyz=self.joint_frame_quats_wxyz,
        )
        pose_indices = [joint_idx - 1 for joint_idx in SMPL_BALANCE_NEUTRAL_JOINTS]
        balance_pose = calibrated["smpl_pose"][0].reshape(23, 3)[pose_indices]
        residual_deg = float(
            torch.rad2deg(torch.linalg.vector_norm(balance_pose, dim=-1)).max()
        )
        print(
            f"[{self.log_prefix}] SMPL balance neutral calibrated "
            f"(joints 1-11, max residual={residual_deg:.6f} deg)"
        )

    def on_mode_exit(self):
        self.frame_buffer.clear()
        self.prev_stamp_ns = None
        self.prev_smpl_pose_np = None
        self.prev_smpl_joints_np = None
        self.prev_body_quat_np = None
        self.prev_smpl_root_quat_np = None
        self.next_target_ns = None
        self.buffer_cleared = True
        self.step = 0

    def run_once(self):
        """Execute one iteration of the pose streaming loop."""
        sample = self.reader.get_latest()
        if sample is None:
            time.sleep(0.005)
            return
        if "body_poses_np" not in sample:
            raise RuntimeError(
                "POSE mode requires Pico full-body tracking; only 3-point VR is available"
            )
        if self.balance_neutral_quats is None:
            self.calibrate_balance_neutral(sample["body_poses_np"])

        latest_data = compute_from_body_poses(
            self.parent_indices,
            self.device,
            sample["body_poses_np"],
            balance_neutral_quats=self.balance_neutral_quats,
            joint_frame_quats_wxyz=self.joint_frame_quats_wxyz,
        )
        _, left_trigger, right_trigger, left_grip, right_grip = get_controller_inputs()
        a_pressed, b_pressed, _, _ = get_abxy_buttons()

        toggle_data_collection_tmp = a_pressed and left_grip > 0.5
        toggle_data_abort_tmp = b_pressed and left_grip > 0.5
        toggle_data_collection = (
            toggle_data_collection_tmp and not self.toggle_data_collection_last
        )
        toggle_data_abort = toggle_data_abort_tmp and not self.toggle_data_abort_last
        self.toggle_data_collection_last = toggle_data_collection_tmp
        self.toggle_data_abort_last = toggle_data_abort_tmp

        left_hand_joints, right_hand_joints = compute_hand_joints_from_inputs(
            self.left_hand_ik_solver,
            self.right_hand_ik_solver,
            left_trigger,
            left_grip,
            right_trigger,
            right_grip,
        )
        smpl_pose_np = (
            latest_data["smpl_pose"].detach().cpu().numpy()[:, :63].reshape(-1, 21, 3)[0]
        ).astype(np.float32)
        body_quat_np = (
            latest_data["global_orient_quat"].detach().cpu().numpy()[0].astype(np.float32)
        )
        smpl_root_quat_np = (
            latest_data["smpl_root_quat"].detach().cpu().numpy()[0].astype(np.float32)
        )
        if self.smpl_arm_source == "xrt-position":
            smpl_joints_local_np = retarget_smpl_arms_from_xrt_positions(
                sample["body_poses_np"],
                latest_data["smpl_joints_local"].detach().cpu().numpy()[0],
            )
            smpl_joints_np = smpl_joints_world_from_local(
                smpl_joints_local_np, smpl_root_quat_np
            )
        else:
            smpl_joints_local_np = (
                latest_data["smpl_joints_local"].detach().cpu().numpy()[0].astype(np.float32)
            )
            smpl_joints_np = (
                latest_data["joints"].detach().cpu().numpy()[0].astype(np.float32)
            )
        curr_stamp_ns = int(sample.get("timestamp_ns", 0))
        step_ns = int(1e9 / max(1, self.target_fps))
        if self.prev_stamp_ns is None:
            self.prev_stamp_ns = curr_stamp_ns
            self.prev_smpl_pose_np = smpl_pose_np
            self.prev_smpl_joints_np = smpl_joints_np
            self.prev_body_quat_np = body_quat_np
            self.prev_smpl_root_quat_np = smpl_root_quat_np
            self.next_target_ns = curr_stamp_ns
            return
        if curr_stamp_ns <= self.prev_stamp_ns:
            return
        if self.next_target_ns is None:
            self.next_target_ns = self.prev_stamp_ns + step_ns
        if self.next_target_ns < self.prev_stamp_ns:
            self.next_target_ns = self.prev_stamp_ns
        if self.next_target_ns > curr_stamp_ns:
            return

        denom = float(curr_stamp_ns - self.prev_stamp_ns)
        alpha = float(self.next_target_ns - self.prev_stamp_ns) / denom if denom > 0.0 else 1.0
        alpha = min(1.0, max(0.0, alpha))
        use_joints = (1.0 - alpha) * self.prev_smpl_joints_np + alpha * smpl_joints_np
        use_pose = _interp_pose_axis_angle(
            self.prev_smpl_pose_np, smpl_pose_np, alpha
        ).astype(np.float32)
        use_body_quat = _quat_lerp_normalized(
            self.prev_body_quat_np, body_quat_np, alpha
        ).astype(np.float32)
        use_smpl_root_quat = _quat_lerp_normalized(
            self.prev_smpl_root_quat_np, smpl_root_quat_np, alpha
        ).astype(np.float32)
        if self.smpl_wrist_source == "g1":
            joint_pos = g1_wrist_reference_from_smpl_pose(use_pose)
        else:
            joint_pos = h2_wrist_reference_from_smpl_pose(use_pose)

        smpl_joints_for_vis = None
        if self.three_point.enable_smpl_vis:
            if self.smpl_vis_source == "xrt":
                smpl_joints_for_vis = xrt_joint_positions_in_pelvis_frame(
                    sample["body_poses_np"]
                ).astype(np.float32)
            else:
                smpl_joints_for_vis = smpl_joints_local_np
        vr_3pt_pose = self.three_point.process_smpl_pose(
            sample["body_poses_np"], smpl_joints_local=smpl_joints_for_vis
        )

        self.frame_buffer["smpl_pose"].append(use_pose)
        self.frame_buffer["smpl_joints"].append(use_joints)
        self.frame_buffer["smpl_root_quat"].append(use_smpl_root_quat)
        self.frame_buffer["body_quat_w"].append(use_body_quat)
        self.frame_buffer["frame_index"].append(int(self.step))
        self.frame_buffer["joint_pos"].append(joint_pos)
        pico_dt = float(sample.get("dt", 0.0))
        pico_fps = float(sample.get("fps", 0.0))
        frame_count = len(self.frame_buffer["frame_index"])

        buffer_is_full = frame_count >= self.num_frames_to_send
        if buffer_is_full and self.buffer_cleared:
            self.buffer_cleared = False

        _, _, rx, _ = get_controller_axes()
        self.yaw_accumulator.update(rx, self.frame_time)

        if buffer_is_full and not self.buffer_cleared:
            numpy_data = {
                "smpl_pose": np.stack(self.frame_buffer["smpl_pose"], axis=0),
                "smpl_joints": np.stack(self.frame_buffer["smpl_joints"], axis=0),
                "smpl_root_quat": np.stack(
                    self.frame_buffer["smpl_root_quat"], axis=0
                ),
                "body_quat_w": np.stack(self.frame_buffer["body_quat_w"], axis=0),
                "joint_pos": np.stack(self.frame_buffer["joint_pos"], axis=0),
                "joint_vel": np.zeros((frame_count, 29)),
                "vr_position": vr_3pt_pose[:, :3].flatten(),
                "vr_orientation": vr_3pt_pose[:, 3:].flatten(),
                "frame_index": np.array(
                    self.frame_buffer["frame_index"], dtype=np.int64
                ),
                "left_trigger": np.array([left_trigger], dtype=np.float32),
                "right_trigger": np.array([right_trigger], dtype=np.float32),
                "left_grip": np.array([left_grip], dtype=np.float32),
                "right_grip": np.array([right_grip], dtype=np.float32),
                "pico_dt": np.array([pico_dt], dtype=np.float32),
                "pico_fps": np.array([pico_fps], dtype=np.float32),
                "timestamp_realtime": np.array(
                    [sample.get("timestamp_realtime", 0.0)], dtype=np.float64
                ),
                "timestamp_monotonic": np.array(
                    [sample.get("timestamp_monotonic", 0.0)], dtype=np.float64
                ),
                "left_hand_joints": left_hand_joints.reshape(-1).astype(np.float32),
                "right_hand_joints": right_hand_joints.reshape(-1).astype(np.float32),
                "toggle_data_collection": np.array(
                    [toggle_data_collection], dtype=bool
                ),
                "toggle_data_abort": np.array([toggle_data_abort], dtype=bool),
                "heading_increment": np.array(
                    [self.yaw_accumulator.yaw_angle_change()], dtype=np.float32
                ),
            }
            self.socket.send(pack_pose_message(numpy_data, topic="pose", version=3))

            if self.record_dir:
                out_path = os.path.join(
                    self.record_dir, f"pose_{self.record_idx:06d}.npz"
                )
                np.savez_compressed(out_path, **numpy_data)
                self.record_idx += 1

        self.step += 1
        self.next_target_ns += step_ns
        self.prev_stamp_ns = curr_stamp_ns
        self.prev_smpl_pose_np = smpl_pose_np
        self.prev_smpl_joints_np = smpl_joints_np
        self.prev_body_quat_np = body_quat_np
        self.prev_smpl_root_quat_np = smpl_root_quat_np
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_report >= 5.0:
            fps = self.fps_counter / (current_time - self.last_fps_report)
            print(f"[{self.log_prefix}] FPS: {fps:.2f}, Step: {self.step}")
            self.fps_counter = 0
            self.last_fps_report = current_time
        elapsed = time.time() - self.frame_start
        if elapsed < self.frame_time:
            time.sleep(self.frame_time - elapsed)
        self.frame_start = time.time()


def run_pico(
    buffer_size: int = 15,
    port: int = 5556,
    num_frames_to_send: int = 10,
    target_fps: int = 50,
    use_cuda: bool = False,
    record_dir: str = "",
    record_format: str = "npz",
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
    smpl_wrist_source: str = "g1",
    smpl_arm_source: str = "rotation",
    smpl_vis_source: str = "sent",
    smpl_joint_frame_calibration: str = "",
    vr3pt_robot_profile: str = "g1",
):
    """Run Pico body tracking with real-time visualization and ZMQ streaming."""
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run Pico streaming."
        )
    initialize_xr_source()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    time.sleep(0.1)
    print(f"ZMQ socket bound to port {port}")
    if build_command_message is not None and build_planner_message is not None:
        try:
            socket.send(build_command_message(start=False, stop=False, planner=False))
            socket.send(build_planner_message(0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], -1.0, -1.0))
        except Exception as e:
            print(f"Warning: failed to send initial command/planner messages: {e}")
    try:
        _pose_stream_common(
            socket=socket,
            buffer_size=buffer_size,
            num_frames_to_send=num_frames_to_send,
            target_fps=target_fps,
            use_cuda=use_cuda,
            record_dir=record_dir,
            record_format=record_format,
            stop_event=None,
            log_prefix="Main",
            enable_vis_vr3pt=enable_vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=enable_waist_tracking,
            enable_smpl_vis=enable_smpl_vis,
            smpl_wrist_source=smpl_wrist_source,
            smpl_arm_source=smpl_arm_source,
            smpl_vis_source=smpl_vis_source,
            smpl_joint_frame_calibration=smpl_joint_frame_calibration,
            vr3pt_robot_profile=vr3pt_robot_profile,
        )
    finally:
        socket.close()
        context.term()
        print("Threads stopped, ZMQ socket closed")


class FeedbackReader:
    """Reads feedback from robot via ZMQ and processes measured upper body position to use as frozen targets."""

    def __init__(self, zmq_feedback_host: str = "localhost", zmq_feedback_port: int = 5559):
        self.poller = ZMQPoller(host=zmq_feedback_host, port=zmq_feedback_port, topic="")

        self.upper_body_joint_indices = self._get_upper_body_joint_indices()

        self.upper_body_position_target = None
        self.left_hand_position_target = None
        self.right_hand_position_target = None
        # Full body joint configuration (29 DOFs) as measured from robot,
        # used for FK when recalibrating VR 3PT tracking against actual robot pose
        self.full_body_q_measured: np.ndarray | None = None

    def _get_upper_body_joint_indices(self) -> list[int]:
        # TODO: get from robot model, not hardcoded
        # robot_model = instantiate_g1_robot_model()
        # return robot_model.get_joint_group_indices("upper_body")
        return [12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]

    def poll_feedback(self, timeout_ms: int = 0):
        """Poll for feedback once, and update internal state."""
        (
            self.upper_body_position_target,
            self.left_hand_position_target,
            self.right_hand_position_target,
            self.full_body_q_measured,
        ) = self._process_upper_body_position_targets(timeout_ms=timeout_ms)
        print("[PlannerLoop] Saved upper body position target:", self.upper_body_position_target)

    def _process_upper_body_position_targets(
        self,
        timeout_ms: int = 0,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        data = self.poller.get_data(timeout_ms=timeout_ms)

        if data is None:
            print("[PlannerLoop] No feedback data received")
            return None, None, None, None

        import json
        from interface.teleop.h2_vr3pt_kinematics import g1_feedback_from_runtime_state
        unpacked = json.loads(data)
        if unpacked.get("schema") != "sonic.h2.state.v1" or abs(time.time() * 1000 - unpacked["sent_at_ms"]) > 500:
            return None, None, None, None
        unpacked["body_q_measured"] = g1_feedback_from_runtime_state(unpacked["joint_pos"])
        full_body_q = None
        if "body_q_measured" in unpacked:
            body_q_swizzled = unpacked["body_q_measured"]
            full_body_q = np.array(body_q_swizzled, dtype=np.float64)
            body_q = [body_q_swizzled[i] for i in self.upper_body_joint_indices]
        else:
            print("[PlannerLoop] body_q_measured not in feedback data")
            body_q = None

        if "left_hand_q_measured" in unpacked:
            left_hand_q = unpacked["left_hand_q_measured"]
        else:
            print("[PlannerLoop] left_hand_q_measured not in feedback data")
            left_hand_q = None

        if "right_hand_q_measured" in unpacked:
            right_hand_q = unpacked["right_hand_q_measured"]
        else:
            print("[PlannerLoop] right_hand_q_measured not in feedback data")
            right_hand_q = None

        return body_q, left_hand_q, right_hand_q, full_body_q


class PlannerStreamer:
    """Encapsulates the planner control loop state and logic."""

    def __init__(
        self,
        socket,
        reader: PicoReader,
        three_point: ThreePointPose,
        poll_hz: int = 20,
        zmq_feedback_host: str = "localhost",
        zmq_feedback_port: int = 5559,
        speed_scale: float = DEFAULT_PLANNER_SPEED_SCALE,
        max_speed: float = DEFAULT_PLANNER_MAX_SPEED,
    ):
        self.socket = socket
        self.reader = reader
        self.three_point = three_point
        self.feedback_reader = FeedbackReader(
            zmq_feedback_host=zmq_feedback_host, zmq_feedback_port=zmq_feedback_port
        )

        self.dt = 1.0 / max(1, poll_hz)
        if not 0.0 < speed_scale <= 1.0:
            raise ValueError(f"planner speed scale must be in (0, 1], got {speed_scale}")
        if max_speed <= 0.0:
            raise ValueError(f"planner max speed must be positive, got {max_speed}")
        self.speed_scale = float(speed_scale)
        self.max_speed = float(max_speed)
        # Current locomotion mode, default IDLE
        self.mode = LocomotionMode.IDLE
        self.prev_ab = False
        self.prev_xy = False
        # Persistent facing buffer (unit vector on XY plane)
        self.yaw_accumulator = YawAccumulator()
        self.last_send = time.time()
        self.last_xrt_timestamp = None
        self.last_vr_diag_report = 0.0

        # Hand IK solvers for Trigger-controlled hand open/close in planner modes
        self.left_hand_ik_solver, self.right_hand_ik_solver = init_hand_ik_solvers()
        print(
            f"[PlannerLoop] Speed limiter: scale={self.speed_scale:.2f}, "
            f"max={self.max_speed:.2f} m/s"
        )

    def reset_yaw(self):
        """Called when entering planner mode. Resets state for fresh start."""
        self.yaw_accumulator.reset()

    def save_upper_body_position_target(self):
        """Poll feedback and save upper body position target."""
        self.feedback_reader.poll_feedback(timeout_ms=500)

    def recalibrate_for_vr3pt(self):
        """
        Recalibrate VR 3-point pose tracking using the robot's current measured joints.

        Polls the runtime state feedback to get the robot's actual joint state, then
        schedules recalibration so VR tracking aligns with the robot's current pose.
        This prevents sudden jumps when entering VR 3PT mode from PLANNER mode.
        """
        self.feedback_reader.poll_feedback(timeout_ms=500)
        if self.feedback_reader.full_body_q_measured is not None:
            self.three_point.reset_with_measured_q(self.feedback_reader.full_body_q_measured)
            print("[PlannerLoop] VR 3PT recalibration scheduled with measured robot pose")
        else:
            print(
                "[PlannerLoop] WARNING: No feedback data for VR 3PT recalibration, "
                f"using default {self.three_point.robot_profile} q as fallback"
            )
            self.three_point.reset_with_default_q()

    def run_once(self, stream_mode: StreamMode):
        """Execute one iteration of the planner control loop."""
        try:
            # Avoid sending old commands if XRT timestamp hasn't advanced, in case of headset disconnect
            xrt_timestamp = xrt.get_time_stamp_ns()
            if xrt_timestamp == self.last_xrt_timestamp:
                return
            self.last_xrt_timestamp = xrt_timestamp

            # A+B => next mode; X+Y => previous mode (rising edges)
            a_pressed, b_pressed, x_pressed, y_pressed = get_abxy_buttons()
            ab_now = bool(a_pressed) and bool(b_pressed)
            xy_now = bool(x_pressed) and bool(y_pressed)
            if ab_now and not self.prev_ab:
                self.mode = LocomotionMode(min(LocomotionMode.INJURED_WALK, self.mode + 1))
                print(f"[PlannerLoop] Mode -> {self.mode.value}: {self.mode.name}")
            if xy_now and not self.prev_xy:
                self.mode = LocomotionMode(max(LocomotionMode.IDLE, self.mode - 1))
                print(f"[PlannerLoop] Mode -> {self.mode.value}: {self.mode.name}")
            self.prev_ab = ab_now
            self.prev_xy = xy_now

            # Read axes/joysticks to control movement, facing, speed and mode
            lx, ly, rx, ry = get_controller_axes()

            # Facing from RIGHT stick: continuous yaw based on rx (right = turn right, left = turn left)
            facing = self.yaw_accumulator.update(rx, self.dt)

            raw_mag = np.hypot(lx, ly)
            raw_mag = np.clip(raw_mag, 0.0, 1.0)
            if np.abs(raw_mag) < JOYSTICK_DEADZONE:
                mag = 0.0
                speed = -1.0
                mode_to_send = LocomotionMode.IDLE
            else:
                mag = (raw_mag - JOYSTICK_DEADZONE) / (1.0 - JOYSTICK_DEADZONE)
                if mag > 1.0:
                    mag = 1.0
                mode_to_send = self.mode
                speed = planner_speed_command(
                    self.mode,
                    mag,
                    speed_scale=self.speed_scale,
                    max_speed=self.max_speed,
                )

            denom = raw_mag if raw_mag > 0.0 else 1.0
            scale = mag / denom
            movement_local = np.array([-lx, ly]) * scale
            perp_x, perp_y = -facing[1], facing[0]
            rotation_facing = np.array([[perp_x, perp_y], [facing[0], facing[1]]])
            movement_global = rotation_facing @ movement_local

            movement = [movement_global[0], movement_global[1], 0.0]

            upper_body_position = None
            left_hand_position = None
            right_hand_position = None
            if stream_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:
                upper_body_position = self.feedback_reader.upper_body_position_target

            # Hand control is independent of the body planner sub-mode. Always
            # attach fresh Trigger-derived targets so DEX3 and Inspire behave
            # consistently in PLANNER, FROZEN, and VR_3PT modes.
            (
                left_menu_button,
                left_trigger,
                right_trigger,
                left_grip,
                right_grip,
            ) = get_controller_inputs()
            lh_joints, rh_joints = compute_hand_joints_from_inputs(
                self.left_hand_ik_solver,
                self.right_hand_ik_solver,
                left_trigger,
                left_grip,
                right_trigger,
                right_grip,
            )
            left_hand_position = lh_joints.reshape(-1).astype(np.float32).tolist()
            right_hand_position = rh_joints.reshape(-1).astype(np.float32).tolist()

            vr_3pt_position = None
            vr_3pt_orientation = None
            vr_3pt_compliance = None
            if stream_mode == StreamMode.PLANNER_VR_3PT:
                sample = self.reader.get_latest()
                if sample is not None:
                    if "vr_3pt_pose_raw" in sample:
                        vr_3pt_pose = self.three_point.process_vr_pose(
                            sample["vr_3pt_pose_raw"]
                        )
                    else:
                        vr_3pt_pose = self.three_point.process_smpl_pose(
                            sample["body_poses_np"]
                        )
                    vr_3pt_position = (vr_3pt_pose[:, :3].flatten()).tolist()
                    vr_3pt_orientation = vr_3pt_pose[:, 3:].flatten().tolist()
                    diag_now = time.monotonic()
                    if diag_now - self.last_vr_diag_report >= 0.5:
                        neck_rpy_deg = sRot.from_quat(
                            vr_3pt_pose[2, 3:], scalar_first=True
                        ).as_euler("xyz", degrees=True)
                        wrist_midpoint = 0.5 * (
                            vr_3pt_pose[0, :3] + vr_3pt_pose[1, :3]
                        )
                        print(
                            "[VR3Diag] "
                            f"wrist_mid_xyz={np.round(wrist_midpoint, 3).tolist()} "
                            f"neck_xyz={np.round(vr_3pt_pose[2, :3], 3).tolist()} "
                            f"neck_rpy_deg={np.round(neck_rpy_deg, 1).tolist()}"
                        )
                        self.last_vr_diag_report = diag_now

            msg = build_planner_message(
                mode_to_send.value,
                movement,
                facing,
                speed=speed,
                height=-1.0,
                upper_body_position=upper_body_position,
                left_hand_position=left_hand_position,
                right_hand_position=right_hand_position,
                vr_3pt_position=vr_3pt_position,
                vr_3pt_orientation=vr_3pt_orientation,
                vr_3pt_compliance=vr_3pt_compliance,
            )
            self.socket.send(msg)
        except Exception as e:
            import traceback

            print(f"[PlannerLoop] error: {e}")
            traceback.print_exc()
            raise

        # pacing
        now = time.time()
        sleep_t = self.dt - (now - self.last_send)
        if sleep_t > 0:
            time.sleep(sleep_t)
        self.last_send = time.time()


class HeadPoseStreamer:
    """Publish head-relative-to-torso pitch/yaw on an independent ZMQ topic."""

    def __init__(self, socket, reader: PicoReader, pitch_joystick=None):
        self.socket = socket
        self.reader = reader
        self.tracker = HeadPoseCommandTracker()
        self.last_timestamp_ns: int | None = None
        self.pitch_joystick = pitch_joystick
        self.last_update_time = None
        self.last_print_time = None
        self.was_adjusting = False

    def calibrate_now(self, sample: dict | None = None) -> bool:
        if self.pitch_joystick is not None:
            # Body/wrist calibration must not reset the manually selected head angle.
            self.last_update_time = None
            return True
        sample = self.reader.get_latest() if sample is None else sample
        if sample is None or "headset_pose" not in sample:
            print("[HeadPose] WARNING: no headset pose available for calibration")
            return False
        if "body_poses_np" not in sample:
            print(
                "[HeadPose] WARNING: full-body tracking is unavailable; "
                "relative head control is disabled"
            )
            return False
        try:
            self.tracker.calibrate(sample["headset_pose"], sample["body_poses_np"])
            self.last_timestamp_ns = None
            print("[HeadPose] Calibrated current head-to-torso orientation as head zero")
            return True
        except (ValueError, RuntimeError) as exc:
            print(f"[HeadPose] Calibration failed: {exc}")
            return False

    def run_once(self) -> None:
        sample = self.reader.get_latest()
        if self.pitch_joystick is not None:
            self._run_pitch_joystick(sample)
            return
        if (
            sample is None
            or "headset_pose" not in sample
            or "body_poses_np" not in sample
        ):
            return
        timestamp_ns = int(sample.get("timestamp_ns", 0))
        if timestamp_ns == self.last_timestamp_ns:
            return
        if not self.tracker.is_calibrated and not self.calibrate_now(sample):
            return

        try:
            pitch, yaw = self.tracker.command(
                sample["headset_pose"], sample["body_poses_np"]
            )
        except (ValueError, RuntimeError) as exc:
            print(f"[HeadPose] Ignoring invalid head/body pose: {exc}")
            return
        self.socket.send(
            build_head_pose_message(
                float(pitch),
                float(yaw),
                timestamp_monotonic=float(sample.get("timestamp_monotonic", 0.0)),
            )
        )
        self.last_timestamp_ns = timestamp_ns

    def _run_pitch_joystick(self, sample) -> None:
        now = time.monotonic()
        if sample is None:
            self.last_update_time = None
            return
        received = float(sample.get("timestamp_monotonic", float("nan")))
        age = now - received
        if not np.isfinite(age) or not 0.0 <= age <= 0.25:
            self.last_update_time = None
            return
        stamp = int(sample.get("timestamp_ns", 0))
        if stamp == self.last_timestamp_ns:
            return
        dt = 0.0 if self.last_update_time is None else now - self.last_update_time
        pitch, yaw = self.pitch_joystick.update(float(sample.get("right_stick_y", 0.0)), dt)
        self.socket.send(build_head_pose_message(pitch, yaw, timestamp_monotonic=received))
        moving = self.pitch_joystick.moving
        if (self.last_print_time is None
                or (moving and now - self.last_print_time >= 0.25)
                or (self.was_adjusting and not moving)):
            print(
                f"[{'HeadFixed' if isinstance(self.pitch_joystick, FixedHeadPose) else 'HeadAdjust'}] "
                f"pitch_target={np.degrees(pitch):+.2f} deg "
                f"({pitch:+.6f} rad), yaw_fixed={np.degrees(yaw):+.2f} deg "
                f"[{'adjusting' if moving else 'hold'}]", flush=True,
            )
            self.last_print_time = now
        self.last_update_time = now
        self.last_timestamp_ns = stamp
        self.was_adjusting = moving


def run_pico_manager(
    port: int = 5556,
    buffer_size: int = 15,
    num_frames_to_send: int = 10,
    target_fps: int = 50,
    use_cuda: bool = False,
    record_dir: str = "",
    record_format: str = "npz",
    zmq_feedback_host: str = "localhost",
    zmq_feedback_port: int = 5559,
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
    smpl_wrist_source: str = "g1",
    planner_speed_scale: float = DEFAULT_PLANNER_SPEED_SCALE,
    planner_max_speed: float = DEFAULT_PLANNER_MAX_SPEED,
    offline_auto_pose: bool = False,
    offline_auto_vr3pt: bool = False,
    vr3pt_robot_profile: str = "g1",
    smpl_arm_source: str = "rotation",
    smpl_vis_source: str = "sent",
    smpl_joint_frame_calibration: str = "",
    head_pitch_joystick: HeadPitchJoystick | None = None,
):
    """
    Manager: creates shared PUB socket and runs pose/planner streamers based on current mode.
    Controller input:
      A+X: Toggle between planner and pose mode
      A+B+X+Y: Toggle policy start/stop
    """
    if xrt is None:
        raise ImportError(
            "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run the manager."
        )
    initialize_xr_source()
    tracking_name = "recorded VR three-point" if is_direct_offline_vr() else "body tracking"
    print(f"Waiting for {tracking_name} data...")
    while not xr_tracking_data_available():
        print(f"waiting for {tracking_name} data...")
        time.sleep(1)

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    time.sleep(0.1)
    print(f"[Manager] ZMQ socket bound to port {port}")

    # Print available locomotion modes
    try:
        print("[Manager] Available modes:")
        for mode in LocomotionMode:
            print(f"  {mode.value}: {mode.name}")
    except Exception:
        pass

    # Create shared reader and 3-point pose processor
    reader = PicoReader(max_queue_size=buffer_size)
    reader.start()

    three_point = ThreePointPose(
        enable_vis_vr3pt=enable_vis_vr3pt,
        with_g1_robot=with_g1_robot,
        enable_waist_tracking=enable_waist_tracking,
        enable_smpl_vis=enable_smpl_vis,
        log_prefix="PoseLoop",
        robot_profile=vr3pt_robot_profile,
    )

    pose_streamer = PoseStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        num_frames_to_send=num_frames_to_send,
        target_fps=target_fps,
        use_cuda=use_cuda,
        record_dir=record_dir,
        record_format=record_format,
        log_prefix="PoseLoop",
        smpl_wrist_source=smpl_wrist_source,
        smpl_arm_source=smpl_arm_source,
        smpl_vis_source=smpl_vis_source,
        smpl_joint_frame_calibration=smpl_joint_frame_calibration,
    )
    planner_streamer = PlannerStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        poll_hz=20,
        zmq_feedback_host=zmq_feedback_host,
        zmq_feedback_port=zmq_feedback_port,
        speed_scale=planner_speed_scale,
        max_speed=planner_max_speed,
    )
    head_pose_streamer = HeadPoseStreamer(socket=socket, reader=reader,
                                          pitch_joystick=head_pitch_joystick)
    if isinstance(head_pitch_joystick, FixedHeadPose):
        print("[HeadFixed] Head pitch/yaw targets are fixed by configuration; "
              "joysticks and headset orientation do not change them.")
    elif head_pitch_joystick is not None:
        print("[HeadAdjust] Right stick forward/back adjusts head pitch; release to hold. "
              "Head yaw is fixed; headset orientation does not drive the head. "
              "Right stick left/right still controls BODY heading.")

    # State machine diagram:
    #
    #   Chain 1 (by_pressed enters/exits, left_axis_click toggles sub-mode):
    #     POSE <--(by)--> PLANNER_FROZEN_UPPER_BODY <--(left_axis_click)--> PLANNER_VR_3PT
    #                                                                         |
    #                                                                    (by)--> POSE
    #
    #   Chain 2 (ax_pressed enters/exits, left_axis_click toggles sub-mode):
    #     POSE <--(ax)--> PLANNER <--(left_axis_click)--> PLANNER_VR_3PT
    #                                                        |
    #                                                   (ax)--> POSE
    #
    #   Emergency stop from any mode: A+B+X+Y (start_combo) --> OFF
    #   POSE_PAUSE: left_menu_button held --> POSE_PAUSE, released --> POSE
    #
    deterministic_offline_mode = offline_auto_pose or offline_auto_vr3pt
    if offline_auto_pose:
        print("[Manager] Offline auto-POSE enabled; recorded mode buttons are ignored")
    elif offline_auto_vr3pt:
        print("[Manager] Offline auto-VR3Point enabled; recorded mode buttons are ignored")
    else:
        print("Manager controls: A+X=toggle mode, A+B+X+Y=start/stop policy")
    current_mode = StreamMode.OFF
    auto_pose_pending = offline_auto_pose
    auto_vr3pt_pending = offline_auto_vr3pt
    last_command_heartbeat = 0.0
    # Track which mode VR_3PT was entered from, so left_axis_click returns to it.
    # Will be either PLANNER or PLANNER_FROZEN_UPPER_BODY.
    vr3pt_parent_mode = StreamMode.PLANNER
    prev_toggle_dc = False
    prev_toggle_da = False
    # Cumulative episode events let collectors detect a missed PUB/SUB packet.
    # Legacy boolean fields below remain for the official G1 exporter.
    record_session_id = time.time_ns()
    record_toggle_count = 0
    record_abort_count = 0
    manager_state_rate = ManagerStateRateLimiter(hz=50)
    try:
        prev_ax_pressed = False
        prev_by_pressed = False
        prev_start_combo = False
        prev_left_axis_click = False
        while True:
            # Poll Pico controller for buttons/axes
            a_pressed, b_pressed, x_pressed, y_pressed = get_abxy_buttons()

            (
                left_menu_button,
                left_trigger_mgr,
                right_trigger_mgr,
                left_grip_mgr,
                _,
            ) = get_controller_inputs()

            left_axis_click, _ = get_axis_clicks()

            # A recorded controller combination must not interrupt deterministic
            # full-body replay. Ctrl-C remains the stop control for this demo mode.
            if deterministic_offline_mode:
                a_pressed = b_pressed = x_pressed = y_pressed = False
                left_menu_button = False
                left_axis_click = False

            # Rising edge: A+X pressed together -> toggle POSE/PLANNER mode
            ax_pressed = (a_pressed) and (x_pressed)

            # Rising edge: B+Y pressed together -> toggle POSE/PLANNER_FROZEN_UPPER_BODY mode
            by_pressed = (b_pressed) and (y_pressed)

            # Rising edge: A+B+X+Y pressed together -> toggle policy start/stop (planner=True)
            start_combo = (a_pressed) and (b_pressed) and (x_pressed) and (y_pressed)

            new_mode = current_mode
            if current_mode == StreamMode.OFF:
                auto_sample = (
                    reader.get_latest()
                    if auto_pose_pending or auto_vr3pt_pending
                    else None
                )
                if auto_pose_pending and auto_sample is not None and "body_poses_np" in auto_sample:
                    new_mode = StreamMode.POSE
                    auto_pose_pending = False
                    pose_streamer.calibrate_balance_neutral(auto_sample["body_poses_np"])
                    head_pose_streamer.calibrate_now(auto_sample)
                    three_point.calibrate_now(auto_sample["body_poses_np"])
                elif auto_vr3pt_pending and auto_sample is not None:
                    new_mode = StreamMode.PLANNER_VR_3PT
                    auto_vr3pt_pending = False
                    head_pose_streamer.calibrate_now(auto_sample)
                elif start_combo and not prev_start_combo:
                    new_mode = StreamMode.PLANNER
                    # Capture SMPL balance and VR 3-point neutral references together.
                    # The operator must be in the documented zero-reference pose.
                    sample = reader.get_latest()
                    if sample is not None:
                        if "body_poses_np" in sample:
                            pose_streamer.calibrate_balance_neutral(sample["body_poses_np"])
                        head_pose_streamer.calibrate_now(sample)
                        if "vr_3pt_pose_raw" in sample:
                            three_point.calibrate_vr_now(sample["vr_3pt_pose_raw"])
                        else:
                            three_point.calibrate_now(sample["body_poses_np"])
                    else:
                        print("[Manager] WARNING: No VR tracking data available for calibration")

            elif current_mode == StreamMode.PLANNER:
                # Chain 2: POSE <--(ax)--> PLANNER <--(left_axis_click)--> VR_3PT
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF
                elif ax_pressed and not prev_ax_pressed:
                    if is_direct_offline_vr():
                        print(
                            "[Manager] Ignoring POSE request: this offline recording contains "
                            "only headset and controller tracking"
                        )
                    else:
                        new_mode = StreamMode.POSE
                elif left_axis_click and not prev_left_axis_click:
                    new_mode = StreamMode.PLANNER_VR_3PT

            elif current_mode == StreamMode.POSE:
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF
                elif ax_pressed and not prev_ax_pressed:
                    new_mode = StreamMode.PLANNER  # Enter chain 2
                elif by_pressed and not prev_by_pressed:
                    new_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY  # Enter chain 1
                elif left_menu_button:
                    new_mode = StreamMode.POSE_PAUSE

            elif current_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:
                # Chain 1: POSE <--(by)--> FROZEN <--(left_axis_click)--> VR_3PT
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF
                elif by_pressed and not prev_by_pressed:
                    new_mode = StreamMode.POSE
                elif left_axis_click and not prev_left_axis_click:
                    new_mode = StreamMode.PLANNER_VR_3PT

            elif current_mode == StreamMode.POSE_PAUSE:
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF
                elif not left_menu_button:
                    new_mode = StreamMode.POSE

            elif current_mode == StreamMode.PLANNER_VR_3PT:
                # VR_3PT is reachable from both chains:
                #   left_axis_click → return to parent (PLANNER or FROZEN)
                #   ax_pressed      → POSE (chain 2 exit)
                #   by_pressed      → POSE (chain 1 exit)
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF
                elif left_axis_click and not prev_left_axis_click:
                    new_mode = vr3pt_parent_mode  # Return to parent mode
                elif ax_pressed and not prev_ax_pressed:
                    if is_direct_offline_vr():
                        print(
                            "[Manager] Ignoring POSE request: direct three-point recordings "
                            "do not contain body joints"
                        )
                    else:
                        new_mode = StreamMode.POSE
                elif by_pressed and not prev_by_pressed:
                    if is_direct_offline_vr():
                        print(
                            "[Manager] Ignoring POSE request: direct three-point recordings "
                            "do not contain body joints"
                        )
                    else:
                        new_mode = StreamMode.POSE

            # Handle mode transitions before running loop
            if new_mode != current_mode:
                if current_mode == StreamMode.POSE:
                    pose_streamer.on_mode_exit()

                # Track parent when entering VR_3PT
                if new_mode == StreamMode.PLANNER_VR_3PT:
                    vr3pt_parent_mode = current_mode
                    print(f"[Manager] VR_3PT parent: {vr3pt_parent_mode.name}")

                if new_mode == StreamMode.POSE:
                    pose_streamer.reset_yaw()
                elif new_mode == StreamMode.PLANNER and current_mode != StreamMode.PLANNER_VR_3PT:
                    # Only reset yaw when freshly entering PLANNER from POSE,
                    # not when returning from VR_3PT sub-mode
                    planner_streamer.reset_yaw()
                elif new_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:
                    if current_mode != StreamMode.PLANNER_VR_3PT:
                        # Freshly entering from POSE: reset yaw and grab initial targets
                        planner_streamer.reset_yaw()
                    # Always re-grab the latest robot state as frozen targets,
                    # whether entering from POSE or returning from VR_3PT
                    # (the old targets are stale after VR_3PT moved the arms)
                    planner_streamer.save_upper_body_position_target()
                elif new_mode == StreamMode.PLANNER_VR_3PT:
                    # Recalibrate VR tracking against the robot's actual current pose
                    # (read via runtime state feedback + FK) to prevent sudden jumps
                    planner_streamer.recalibrate_for_vr3pt()

            # Run one iteration of the new mode
            if new_mode == StreamMode.POSE:
                pose_streamer.run_once()
            elif (
                new_mode == StreamMode.PLANNER
                or new_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY
                or new_mode == StreamMode.PLANNER_VR_3PT
            ):
                planner_streamer.run_once(new_mode)

            if new_mode != StreamMode.OFF:
                head_pose_streamer.run_once()

            # Make sure to send command messages after loop iteration to ensure data arrives before mode switch
            if new_mode != current_mode:
                if new_mode == StreamMode.OFF:
                    socket.send(build_command_message(start=False, stop=True, planner=True))
                    exit()
                elif (
                    new_mode == StreamMode.PLANNER
                    or new_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY
                    or new_mode == StreamMode.PLANNER_VR_3PT
                ):
                    socket.send(build_command_message(start=True, stop=False, planner=True))
                elif new_mode == StreamMode.POSE:
                    socket.send(build_command_message(start=True, stop=False, planner=False))

                print(f"[Manager] StreamMode switch: {current_mode.name} -> {new_mode.name}")
                current_mode = new_mode
                last_command_heartbeat = time.monotonic()

            # PUB/SUB delivery is not reliable for a single transition packet.
            # Reassert the active start/mode state so late or reconnecting deploy
            # subscribers converge without keyboard intervention.
            now_monotonic = time.monotonic()
            if current_mode != StreamMode.OFF and now_monotonic - last_command_heartbeat >= 0.25:
                planner_mode = current_mode in (
                    StreamMode.PLANNER,
                    StreamMode.PLANNER_FROZEN_UPPER_BODY,
                    StreamMode.PLANNER_VR_3PT,
                )
                socket.send(
                    build_command_message(start=True, stop=False, planner=planner_mode)
                )
                last_command_heartbeat = now_monotonic

            # Mode-independent: send manager_state for data exporter
            toggle_dc_tmp = bool(a_pressed) and left_grip_mgr > 0.5
            toggle_da_tmp = bool(b_pressed) and left_grip_mgr > 0.5
            toggle_dc = toggle_dc_tmp and not prev_toggle_dc
            toggle_da = toggle_da_tmp and not prev_toggle_da
            prev_toggle_dc = toggle_dc_tmp
            prev_toggle_da = toggle_da_tmp
            record_toggle_count += int(toggle_dc)
            record_abort_count += int(toggle_da)
            # Stale tracking can make run_once return immediately: never flood the recorder.
            if manager_state_rate.due(time.monotonic(), force=toggle_dc or toggle_da):
                socket.send(
                    pack_pose_message(
                        {
                            "stream_mode": np.array([current_mode.value], dtype=np.int32),
                            "left_trigger": np.array([left_trigger_mgr], dtype=np.float32),
                            "right_trigger": np.array([right_trigger_mgr], dtype=np.float32),
                            "timestamp_monotonic": np.array([time.monotonic()], dtype=np.float64),
                            "toggle_data_collection": np.array([toggle_dc], dtype=bool),
                            "toggle_data_abort": np.array([toggle_da], dtype=bool),
                            "record_session_id": np.array([record_session_id], dtype=np.int64),
                            "record_toggle_count": np.array([record_toggle_count], dtype=np.int64),
                            "record_abort_count": np.array([record_abort_count], dtype=np.int64),
                        },
                        topic="manager_state",
                    )
                )

            prev_ax_pressed = ax_pressed
            prev_by_pressed = by_pressed
            prev_start_combo = start_combo
            prev_left_axis_click = left_axis_click

    except KeyboardInterrupt:
        print("\nStopping manager...")
    finally:
        # Cleanup resources
        reader.stop()
        three_point.close()
        socket.close()
        context.term()
        print("[Manager] Shutdown complete")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--head-control", choices=("headset", "right-stick", "fixed"), default="headset",
                        help="H2 head source; right-stick adjusts pitch, fixed holds both angles (requires --manager)")
    parser.add_argument("--head-pitch-deg", type=float, default=0.0,
                        help="Initial/fixed head joint pitch in right-stick/fixed mode (degrees)")
    parser.add_argument("--head-yaw-deg", type=float, default=0.0,
                        help="Fixed head joint yaw in right-stick/fixed mode (degrees)")
    parser.add_argument("--head-pitch-rate-deg-s", type=float, default=5.0,
                        help="Maximum right-stick pitch adjustment rate (0, 30] deg/s")
    parser.add_argument("--head-stick-deadzone", type=float, default=0.15)
    parser.add_argument("--buffer_size", type=int, default=15, help="Sliding window buffer size")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ server port (default: 5556)")
    parser.add_argument(
        "--num_frames_to_send",
        type=int,
        default=10,
        help="Consecutive SMPL frames per streamed pose window (default: 10)",
    )
    parser.add_argument("--target_fps", type=int, default=50, help="Target loop FPS (default: 50)")
    parser.add_argument(
        "--cuda", action="store_true", help="Use CUDA for tensors and model (default: CPU)"
    )
    parser.add_argument(
        "--record_dir",
        type=str,
        default="",
        help="Directory to save sent batches (default: disabled)",
    )
    parser.add_argument(
        "--record_format",
        type=str,
        default="npz",
        help="Recording format: 'npz' or 'bin' (default: npz)",
    )
    parser.add_argument(
        "--manager",
        action="store_true",
        help="Run manager with planner and pose threads (interactive)",
    )
    parser.add_argument(
        "--offline_vr_file",
        "--offline-vr-file",
        type=str,
        default="",
        help=(
            "Replay a real three-point VR recording: an official XRoboToolkit "
            "RobotDataRecorder directory or a raw .npz"
        ),
    )
    parser.add_argument(
        "--offline_vr_fps",
        "--offline-vr-fps",
        type=float,
        default=50.0,
        help="Frame rate for official head.txt/controller.txt recordings (default: 50)",
    )
    parser.add_argument(
        "--offline_vr_speed",
        "--offline-vr-speed",
        type=float,
        default=1.0,
        help="Offline VR playback speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--offline_vr_loop",
        "--offline-vr-loop",
        action="store_true",
        help="Loop the raw VR recording when it reaches the end",
    )
    parser.add_argument(
        "--offline_auto_pose",
        "--offline-auto-pose",
        action="store_true",
        help="Calibrate from the first full-body frame and start directly in SMPL POSE mode",
    )
    parser.add_argument(
        "--offline_auto_vr3pt",
        "--offline-auto-vr3pt",
        action="store_true",
        help="Calibrate from the first frame and start directly in planner VR 3-point mode",
    )
    parser.add_argument(
        "--zmq_feedback_host",
        type=str,
        default="localhost",
        help="ZMQ feedback host (default: localhost)",
    )
    parser.add_argument(
        "--zmq_feedback_port",
        type=int,
        default=5559,
        help="ZMQ feedback port (default: 5559)",
    )
    parser.add_argument(
        "--vr3pt_test",
        action="store_true",
        help="Run VR 3-point pose visualizer test (reference frames only)",
    )
    parser.add_argument(
        "--vr3pt_live",
        action="store_true",
        help="Capture one frame of VR 3-point pose and visualize with reference frames",
    )
    parser.add_argument(
        "--vr3pt_realtime",
        action="store_true",
        help="Run standalone real-time VR 3-point pose visualizer",
    )
    parser.add_argument(
        "--vis_vr3pt",
        action="store_true",
        help="Enable inline VR 3-point pose visualization in pose streaming mode",
    )
    parser.add_argument(
        "--vr3pt_hz",
        type=int,
        default=10,
        help="Update rate for real-time VR visualization in Hz (default: 10)",
    )
    parser.add_argument(
        "--no_g1",
        action="store_true",
        help="Disable G1 robot visualization in VR 3pt pose view (G1 is shown by default)",
    )
    parser.add_argument(
        "--waist_tracking",
        action="store_true",
        help="Enable G1 robot waist to follow VR head orientation (disabled by default for performance)",
    )
    parser.add_argument(
        "--vis_smpl",
        action="store_true",
        help="Enable SMPL body joint visualization (24 joint spheres) in the VR3pt viewer",
    )
    parser.add_argument(
        "--smpl_wrist_source",
        "--smpl-wrist-source",
        choices=("g1", "h2-kinematic"),
        default="g1",
        help=(
            "Mode-2 W source: original G1 PoseStreamer formula or stateless local "
            "SMPL-to-H2 wrist kinematics"
        ),
    )
    parser.add_argument(
        "--smpl-arm-source",
        choices=("rotation", "xrt-position"),
        default="rotation",
        help=(
            "Mode-2 arm geometry source: quaternion FK or XRT shoulder/elbow/wrist "
            "position directions with fixed SMPL bone lengths"
        ),
    )
    parser.add_argument(
        "--smpl-vis-source",
        choices=("sent", "xrt"),
        default="sent",
        help="SMPL visualization skeleton: streamed encoder target or raw XRT joint positions",
    )
    parser.add_argument(
        "--smpl-joint-frame-calibration",
        type=str,
        default="",
        help=(
            "Optional JSON with per-joint XRT-to-SMPL bind-frame corrections; "
            "applied before any optional position-based arm override"
        ),
    )
    parser.add_argument(
        "--vr3pt-robot-profile",
        choices=("g1", "h2-native"),
        default="g1",
        help="Robot FK geometry used to calibrate VR 3-point targets (default: g1)",
    )
    parser.add_argument(
        "--planner-speed-scale",
        type=float,
        default=DEFAULT_PLANNER_SPEED_SCALE,
        help=(
            "Scale explicit planner locomotion speeds; must be in (0, 1] "
            f"(default: {DEFAULT_PLANNER_SPEED_SCALE})"
        ),
    )
    parser.add_argument(
        "--planner-max-speed",
        type=float,
        default=DEFAULT_PLANNER_MAX_SPEED,
        help=(
            "Maximum planner locomotion speed in m/s "
            f"(default: {DEFAULT_PLANNER_MAX_SPEED})"
        ),
    )
    args = parser.parse_args()

    head_pitch_joystick = None
    if args.head_control in ("right-stick", "fixed"):
        if not args.manager:
            parser.error(f"--head-control {args.head_control} requires --manager")
        try:
            head_controller_type = FixedHeadPose if args.head_control == "fixed" else HeadPitchJoystick
            head_pitch_joystick = head_controller_type(
                args.head_pitch_deg, args.head_yaw_deg,
                args.head_pitch_rate_deg_s, args.head_stick_deadzone,
            )
        except ValueError as exc:
            parser.error(str(exc))

    if not 0.0 < args.planner_speed_scale <= 1.0:
        parser.error("--planner-speed-scale must be in (0, 1]")
    if args.planner_max_speed <= 0.0:
        parser.error("--planner-max-speed must be positive")
    if args.offline_auto_pose and not args.manager:
        parser.error("--offline-auto-pose requires --manager")
    if args.offline_auto_vr3pt and not args.manager:
        parser.error("--offline-auto-vr3pt requires --manager")
    if args.offline_auto_pose and args.offline_auto_vr3pt:
        parser.error("--offline-auto-pose and --offline-auto-vr3pt are mutually exclusive")

    if args.offline_vr_file:
        from interface.teleop.xr.offline_xrt import OfflineXRoboToolkit

        xrt = OfflineXRoboToolkit.from_path(
            args.offline_vr_file,
            fps=args.offline_vr_fps,
            speed=args.offline_vr_speed,
            loop=args.offline_vr_loop,
        )
        if args.vis_smpl and xrt.is_offline_3pt:
            parser.error("--vis-smpl is unavailable for a three-point offline VR recording")
        tracking_kind = "three-point" if xrt.is_offline_3pt else "full-body"
        print(
            f"[XR] Replaying {tracking_kind} recording: {args.offline_vr_file} "
            f"({xrt.frame_count} frames at {xrt.fps:.3f} Hz)"
        )
    elif args.offline_auto_pose or args.offline_auto_vr3pt:
        parser.error("offline auto mode requires --offline-vr-file")

    # Standalone VR3Pt test modes (exit after finishing)
    if args.vr3pt_test:
        print("Running VR 3-point pose visualizer test...")
        run_vr3pt_visualizer_test()
        print("VR 3-point pose visualizer test completed")
        exit(0)

    if args.vr3pt_live:
        print("Running VR 3-point pose live capture...")
        run_vr3pt_live_visualizer()
        print("VR 3-point pose live visualizer completed")
        exit(0)

    if args.vr3pt_realtime:
        print("Running VR 3-point pose real-time visualizer...")
        run_vr3pt_realtime_visualizer(update_hz=args.vr3pt_hz)
        print("VR 3-point pose real-time visualizer completed")
        exit(0)

    # Main execution modes
    # G1 robot visualization is enabled by default when vis_vr3pt is used
    with_g1_robot = not args.no_g1

    if args.manager:
        run_pico_manager(
            port=args.port,
            buffer_size=args.buffer_size,
            num_frames_to_send=args.num_frames_to_send,
            target_fps=args.target_fps,
            use_cuda=args.cuda,
            record_dir=args.record_dir,
            record_format=args.record_format,
            zmq_feedback_host=args.zmq_feedback_host,
            zmq_feedback_port=args.zmq_feedback_port,
            enable_vis_vr3pt=args.vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=args.waist_tracking,
            enable_smpl_vis=args.vis_smpl,
            smpl_wrist_source=args.smpl_wrist_source,
            smpl_arm_source=args.smpl_arm_source,
            smpl_vis_source=args.smpl_vis_source,
            smpl_joint_frame_calibration=args.smpl_joint_frame_calibration,
            planner_speed_scale=args.planner_speed_scale,
            planner_max_speed=args.planner_max_speed,
            offline_auto_pose=args.offline_auto_pose,
            offline_auto_vr3pt=args.offline_auto_vr3pt,
            vr3pt_robot_profile=args.vr3pt_robot_profile,
            head_pitch_joystick=head_pitch_joystick,
        )
    else:
        # Run legacy single-thread pose streaming
        run_pico(
            buffer_size=args.buffer_size,
            port=args.port,
            num_frames_to_send=args.num_frames_to_send,
            target_fps=args.target_fps,
            use_cuda=args.cuda,
            record_dir=args.record_dir,
            record_format=args.record_format,
            enable_vis_vr3pt=args.vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=args.waist_tracking,
            enable_smpl_vis=args.vis_smpl,
            smpl_wrist_source=args.smpl_wrist_source,
            smpl_arm_source=args.smpl_arm_source,
            smpl_vis_source=args.smpl_vis_source,
            smpl_joint_frame_calibration=args.smpl_joint_frame_calibration,
            vr3pt_robot_profile=args.vr3pt_robot_profile,
        )
