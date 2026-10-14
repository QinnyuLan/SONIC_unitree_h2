"""Factory for the Unitree H2 kinematic model used by VR 3-point calibration."""

from pathlib import Path

import numpy as np

from gear_sonic.data.robot_model.robot_model import RobotModel

H2_JOINT_NAMES_LOCAL = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_pitch_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

H2_DEFAULT_ANGLES_LOCAL = np.asarray(
    [
        -0.312, 0.0, 0.0, 0.669, 0.0, -0.363,
        -0.312, 0.0, 0.0, 0.669, 0.0, -0.363,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)


def h2_configuration_from_local_joints(
    robot_model: RobotModel, joint_values: np.ndarray
) -> np.ndarray:
    """Expand the 31 H2 local/training joints into the model configuration."""
    values = np.asarray(joint_values, dtype=np.float64)
    if values.shape != (len(H2_JOINT_NAMES_LOCAL),):
        raise ValueError(f"H2 joint values must have shape (31,), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("H2 joint values contain non-finite values")

    configuration = robot_model.q_zero
    for name, value in zip(H2_JOINT_NAMES_LOCAL, values, strict=True):
        configuration[robot_model.dof_index(name)] = value
    return configuration


def instantiate_h2_robot_model() -> RobotModel:
    """Instantiate the H2 model and assign the deploy/training default pose."""
    description_dir = (
        Path(__file__).resolve().parents[2]
        / "assets"
        / "robot_description"
        / "unitree_ros"
        / "h2_description"
    )
    robot_model = RobotModel(
        str(description_dir / "H2_loop.urdf"),
        str(description_dir),
    )
    default_configuration = h2_configuration_from_local_joints(
        robot_model, H2_DEFAULT_ANGLES_LOCAL
    )
    robot_model.default_body_pose = default_configuration.copy()
    robot_model.initial_body_pose = default_configuration.copy()
    return robot_model
