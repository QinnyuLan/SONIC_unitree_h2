import numpy as np
import pytest
from interface.commands.protocol import envelope, head_command, g1_reference
from interface.motion_replay import Motion
from interface.teleop.bridge import unpack, pose_reference, standing_g1
from interface.teleop.transport.packets import pack_pose_message, build_head_pose_message


def motion():
    return Motion(np.arange(101)[:, None] * np.ones((1, 29)) * .02,
                  np.tile([1, 0, 0, 0], (101, 1)), 50)


def test_sparse_replay_resamples_time_and_velocity():
    reference = motion().sample(.5, 2)
    np.testing.assert_allclose(np.array(reference['joint_pos'])[:, 0], np.minimum(.5 + np.arange(10) * .2, 2))
    np.testing.assert_allclose(np.array(reference['joint_vel'])[:8, 0], 2)
    assert reference['joint_vel'][8][0] == 0


def test_quaternion_slerp_crosses_sign_boundary():
    q = np.tile([1., 0, 0, 0], (101, 1)); q[50:] *= -1
    m = Motion(motion().q, q, 50)
    np.testing.assert_allclose(np.abs(np.array(m.sample(.99)['root_quat'])[:, 0]), 1)


def test_npz_roundtrip_and_invalid_shape(tmp_path):
    path = tmp_path / 'motion.npz'
    np.savez(path, joint_pos=motion().q, root_quat=motion().root, fps=50)
    assert Motion.load(path).duration == 2
    with pytest.raises(ValueError):
        Motion(np.zeros((10, 31)), np.tile([1, 0, 0, 0], (10, 1)), 50)


@pytest.mark.parametrize('ttl', [49, 1001, 200.5, True])
def test_protocol_rejects_invalid_ttl(ttl):
    with pytest.raises(ValueError):
        envelope('stand', {}, source='test', session='x', seq=1, ttl_ms=ttl)


def test_head_is_independent_message():
    message = envelope('head', head_command(.3, -.4), source='head', session='x', seq=1)
    assert message['data'] == {'pitch': .3, 'yaw': -.4}
    assert 'head' not in message


def test_tracking_wire_head_and_truncation():
    topic, data = unpack(build_head_pose_message(.3, -.4))
    assert topic == 'head_pose'
    np.testing.assert_allclose(data['head_pitch_yaw'], [.3, -.4])
    with pytest.raises(ValueError):
        unpack(build_head_pose_message(.3, -.4)[:-1])


def test_smpl_bridge_uses_last_ten_frames_and_semantic_wrists():
    data = {'smpl_joints': np.zeros((15, 24, 3)), 'smpl_root_quat': np.tile([1., 0, 0, 0], (15, 1)),
            'joint_pos': np.arange(15 * 29).reshape(15, 29).astype(float)}
    topic, decoded = unpack(pack_pose_message(data))
    assert topic == 'pose'
    result = pose_reference(decoded, 'smpl', standing_g1())
    assert result['wrist_joints'][0][0] == data['joint_pos'][5, 23]
    assert len(result['joints']) == 10


def test_vr_bridge_uses_explicit_lower_body():
    data = {'vr_position': np.zeros(9), 'vr_orientation': np.tile([1., 0, 0, 0], 3)}
    reference = pose_reference(data, 'vr', motion().sample(0))
    assert reference['lower_body']['joint_pos'][1][0] == pytest.approx(.1)


def test_g1_rejects_nonfinite():
    q = np.zeros((10, 29)); q[0, 0] = np.nan
    with pytest.raises(ValueError):
        g1_reference(q, np.zeros((10, 29)), np.tile([1, 0, 0, 0], (10, 1)))
