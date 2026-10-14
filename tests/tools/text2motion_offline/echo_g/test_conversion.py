"""Physical ECHO-G output conversion to standard G1 replay references."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from interface.motion_replay import Motion
from text2motion_offline.echo_g.cli import convert_motion


def test_echo_physical_joint_permutation_velocity_and_rotation_rows():
    representation = np.zeros((30, 39))
    rotations = Rotation.from_euler('xyz', np.tile([.2, -.4, 3.12], (30, 1)))
    representation[:, :6] = rotations.as_matrix()[:, :2, :].reshape(30, 6)
    representation[:, 10:] = np.arange(29) + np.arange(30)[:, None] / 30
    result = convert_motion(representation)
    assert result['joint_pos'].shape == (50, 29)
    # Distinguishes left/right, MuJoCo/semantic order and pitch/roll ankle order.
    np.testing.assert_allclose(result['joint_pos'][0, [0, 1, 2, 13, 17, 23, 28]], [0, 6, 12, 4, 5, 19, 28])
    np.testing.assert_allclose(result['joint_vel'][5:40], 1, atol=1e-5)
    np.testing.assert_allclose(result['joint_vel'][-1], 0)
    recovered = Rotation.from_quat(result['root_quat'][:, [1, 2, 3, 0]]).as_matrix()
    np.testing.assert_allclose(recovered, np.tile(rotations.as_matrix()[:1], (50, 1, 1)), atol=1e-7)
    Motion(result['joint_pos'], result['root_quat'], 50, result['joint_vel']).sample(.2)


@pytest.mark.parametrize('failure', ['normalized_zero', 'nan', 'wrong_dimension'])
def test_invalid_echo_motion_cannot_be_replayed(failure):
    value = np.zeros((3, 39))
    value[:, 0] = value[:, 4] = 1
    if failure == 'normalized_zero':
        value[:, :6] = 0
    elif failure == 'nan':
        value[1, 10] = np.nan
    else:
        value = value[:, :-1]
    with pytest.raises(ValueError):
        convert_motion(value)


def test_echo_root_resampling_takes_short_arc_across_yaw_wrap():
    representation = np.zeros((30, 39))
    rotations = Rotation.from_euler('z', np.linspace(170, 190, 30), degrees=True)
    representation[:, :6] = rotations.as_matrix()[:, :2, :].reshape(30, 6)
    reference = convert_motion(representation)
    result = Rotation.from_quat(reference['root_quat'][:, [1, 2, 3, 0]])
    angles = np.unwrap(result.as_euler('xyz')[:, 2])
    assert np.max(np.abs(np.diff(angles))) < .02
    np.testing.assert_allclose(angles[[0, -1]], np.deg2rad([170, 190]), atol=1e-6)
