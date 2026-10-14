"""Convert legacy G1-compatible deploy feedback into H2 FK coordinates."""

import numpy as np

from gear_sonic.data.robot_model.instantiation.h2 import (
    H2_DEFAULT_ANGLES_LOCAL,
    H2_JOINT_NAMES_LOCAL,
    h2_configuration_from_local_joints,
)

# output_interface.hpp publishes body_q_measured in G1 MuJoCo order even for
# H2 profiles. Each entry selects one policy-order deviation and adds the G1
# default angle. These constants mirror policy_parameters.hpp.
G1_FEEDBACK_POLICY_INDEX = np.asarray(
    [
        0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
        11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
    ],
    dtype=np.int64,
)
G1_FEEDBACK_DEFAULT_ANGLES = np.asarray(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)

# H2-native policy index -> H2 local/training joint index. Head pitch/yaw are
# independent of the 29D body policy and remain at their default values.
H2_NATIVE_POLICY_TO_LOCAL = np.asarray(
    [
        0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9,
        17, 24, 5, 11, 18, 25, 4, 10, 19, 26,
        20, 27, 21, 28, 22, 29, 23, 30,
    ],
    dtype=np.int64,
)
H2_NATIVE_DEFAULT_ANGLES_POLICY = np.asarray(
    [
        -0.312, -0.312, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.669, 0.669, 0.2, 0.2, -0.363, -0.363, 0.2, -0.2, 0.0,
        0.0, 0.0, 0.0, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)


def h2_local_joints_from_g1_compatible_feedback(feedback_q: np.ndarray) -> np.ndarray:
    """Recover absolute H2 local joints from deploy's legacy 29D feedback."""
    feedback = np.asarray(feedback_q, dtype=np.float64)
    if feedback.shape != (29,):
        raise ValueError(f"G1-compatible feedback must have shape (29,), got {feedback.shape}")
    if not np.isfinite(feedback).all():
        raise ValueError("G1-compatible feedback contains non-finite values")

    policy_delta = np.empty(29, dtype=np.float64)
    policy_delta[G1_FEEDBACK_POLICY_INDEX] = feedback - G1_FEEDBACK_DEFAULT_ANGLES

    local = H2_DEFAULT_ANGLES_LOCAL.copy()
    local[H2_NATIVE_POLICY_TO_LOCAL] = policy_delta + H2_NATIVE_DEFAULT_ANGLES_POLICY
    return local


def h2_configuration_from_g1_compatible_feedback(
    robot_model, feedback_q: np.ndarray
) -> np.ndarray:
    """Return an H2 model configuration for one legacy deploy feedback sample."""
    return h2_configuration_from_local_joints(
        robot_model, h2_local_joints_from_g1_compatible_feedback(feedback_q)
    )


__all__ = [
    "G1_FEEDBACK_DEFAULT_ANGLES",
    "G1_FEEDBACK_POLICY_INDEX",
    "H2_JOINT_NAMES_LOCAL",
    "H2_NATIVE_DEFAULT_ANGLES_POLICY",
    "H2_NATIVE_POLICY_TO_LOCAL",
    "h2_configuration_from_g1_compatible_feedback",
    "h2_local_joints_from_g1_compatible_feedback",
]


def g1_feedback_from_runtime_state(dds_joints):
    """Convert native H2 DDS measurements for the existing PICO FK calibration."""
    dds = np.asarray(dds_joints, dtype=np.float64)
    if dds.shape != (31,) or not np.isfinite(dds).all():
        raise ValueError("expected 31 finite H2 DDS joints")
    policy_to_dds = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 5, 11,
                              16, 23, 4, 10, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])
    delta = dds[policy_to_dds] - H2_NATIVE_DEFAULT_ANGLES_POLICY
    return delta[G1_FEEDBACK_POLICY_INDEX] + G1_FEEDBACK_DEFAULT_ANGLES
