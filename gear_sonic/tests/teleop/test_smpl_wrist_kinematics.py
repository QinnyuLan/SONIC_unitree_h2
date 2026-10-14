import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.utils.teleop.smpl_wrist_kinematics import (
    H2_SEMANTIC_WRIST_INDICES,
    H2_WRIST_LOWER,
    H2_WRIST_UPPER,
    h2_wrist_reference_from_smpl_pose,
)


def test_h2_identity_wrist_pose_is_zero():
    reference = h2_wrist_reference_from_smpl_pose(np.zeros((21, 3)))

    np.testing.assert_array_equal(reference, np.zeros(29, dtype=np.float32))


def test_h2_wrist_projection_uses_mirrored_right_coordinates():
    pose = np.zeros((21, 3))
    euler = np.asarray([0.3, -0.2, 0.4])
    pose[19] = Rotation.from_euler("XYZ", euler).as_rotvec()
    pose[20] = Rotation.from_euler("XYZ", euler).as_rotvec()

    reference = h2_wrist_reference_from_smpl_pose(pose)

    np.testing.assert_allclose(
        reference[list(H2_SEMANTIC_WRIST_INDICES)],
        [0.3, -0.3, -0.2, 0.2, 0.4, 0.4],
        atol=1.0e-6,
    )


def test_h2_wrist_projection_clips_to_training_mjcf_limits():
    pose = np.zeros((21, 3))
    pose[19] = Rotation.from_euler("XYZ", [2.9, 0.9, 1.5]).as_rotvec()
    pose[20] = Rotation.from_euler("XYZ", [-2.9, -0.9, -1.5]).as_rotvec()

    reference = h2_wrist_reference_from_smpl_pose(pose)
    wrist = reference[list(H2_SEMANTIC_WRIST_INDICES)]

    assert np.all(wrist >= H2_WRIST_LOWER - 1.0e-6)
    assert np.all(wrist <= H2_WRIST_UPPER + 1.0e-6)


def test_h2_wrist_projection_rejects_invalid_input():
    with np.testing.assert_raises_regex(ValueError, "21x3"):
        h2_wrist_reference_from_smpl_pose(np.zeros((20, 3)))

    pose = np.zeros((21, 3))
    pose[19, 0] = np.nan
    with np.testing.assert_raises_regex(ValueError, "non-finite"):
        h2_wrist_reference_from_smpl_pose(pose)
