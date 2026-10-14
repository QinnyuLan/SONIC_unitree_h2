"""Offline tests only: no DDS participant, ZMQ sockets, camera or robot commands."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS

import av
import msgpack
import numpy as np
import pyarrow.parquet as pq
import pytest

from gear_sonic.data.h2_collection.core import (
    BODY_NAMES, HAND_NAMES, RecordingEvents, Sample, decode_dds, make_row, problems, unpack_wire,
)
from gear_sonic.data.h2_collection.writer import EpisodeWriter
from gear_sonic.data.h2_collection.collector import Hub, config_problem
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    pack_pose_message, build_head_pose_message, build_command_message, build_planner_message,
)

NOW = 8_000_000_000


def manager(toggle=0, abort=0, session=123):
    return dict(stream_mode=np.array([5], dtype=np.int32), record_session_id=np.array([session], dtype=np.int64),
                record_toggle_count=np.array([toggle], dtype=np.int64),
                record_abort_count=np.array([abort], dtype=np.int64),
                toggle_data_collection=np.array([False]), toggle_data_abort=np.array([False]),
                timestamp_monotonic=np.array([NOW/1e9]))


def snapshots():
    imu = dict(quaternion=[1., 0., 0., 0.], gyroscope=[0., 0., 0.], accelerometer=[0., 0., 9.81])
    body = dict(q=list(np.linspace(-.3, .3, 31)), dq=[.2]*31, tau=[1.]*31,
                temperature=[30, 32]*31, error=[0]*31, tick=42, mode_pr=0, mode_machine=1, imu=imu, crc_ok=True)
    body['q'][29] = .65
    cmd = dict(q=[.1]*31, dq=[0.]*31, tau=[0.]*31, kp=[40.]*31, kd=[2.]*31,
               mode=[1]*31, mode_pr=0, mode_machine=1, crc_ok=True)
    cmd['q'][29] = np.deg2rad(40)
    values = dict(lowstate=body, lowcmd=cmd, hand_state=dict(q=[.7]*6+[.8]*6, lost=[0]*12),
                  hand_cmd=dict(q=[.2]*6+[.4]*6), g1_debug=dict(token_state=[.01]*64, index=8),
                  manager_state=manager(), camera=dict(image=np.zeros((48, 64, 3), np.uint8), native_size=[64, 48]),
                  head_pose=dict(head_pitch_yaw=np.array([np.deg2rad(40), 0.])),
                  planner=dict(mode=np.array([0]), movement=np.array([0., 0., 0.])))
    return {k: Sample(v, NOW, NOW+1, sequence=1) for k, v in values.items()}


def test_dds_order_matches_authoritative_bridge():
    path = Path(__file__).parents[1]/'utils/mujoco_sim/unitree_sdk2py_bridge.py'
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'H2_JOINT_NAMES_DDS' for t in node.targets):
            assert BODY_NAMES == ast.literal_eval(node.value)
            break
    else:
        pytest.fail('missing bridge order')
    assert BODY_NAMES[29:] == ['head_pitch_joint', 'head_yaw_joint']
    assert len(HAND_NAMES) == 12


def test_hand_dds_reorders_right_left_and_preserves_faults():
    values = [NS(q=i/12, lost=int(i == 7)) for i in range(12)]
    decoded = decode_dds('hand_state', NS(states=values))
    assert decoded['q'] == [i/12 for i in list(range(6, 12))+list(range(6))]
    assert decoded['lost'][1] == 1
    assert decode_dds('hand_cmd', NS(cmds=values))['q'] == decoded['q']
    with pytest.raises(ValueError):
        decode_dds('hand_state', NS(states=values[:7]))


@pytest.mark.parametrize('wire', [
    build_head_pose_message(.7, 0, 8), build_command_message(True, False, True),
    build_planner_message(0, [0, 0, 0], [1, 0, 0]),
    pack_pose_message(manager(), topic='manager_state'),
    pack_pose_message({'smpl_pose': np.ones((10, 21, 3), np.float32)}, topic='pose'),
])
def test_wire_roundtrip_and_truncation(wire):
    topic, data = unpack_wire(wire)
    assert data
    with pytest.raises((ValueError, KeyError)):
        unpack_wire(wire[:-1])
    with pytest.raises(ValueError):
        unpack_wire(wire+b'junk')


def test_counter_events_survive_packet_loss_and_ignore_late_join():
    events = RecordingEvents()
    assert events.update(manager(4)) == []
    # Boolean edge packet lost: new cumulative counter still reports exactly one event.
    assert events.update(manager(5)) == ['toggle']
    assert events.update(manager(5)) == []
    assert events.update(manager(5, 1)) == ['discard']
    assert events.update(manager(7, 1)) == ['ambiguous_record_events']
    assert events.update(manager(0, 0, session=124)) == ['manager_restarted']


def test_required_streams_and_independent_head_measurement():
    samples = snapshots()
    assert problems(samples, NOW) == []
    row = make_row(samples, NOW, NOW, 0, 50)
    assert len(row['observation.state']) == len(row['action']) == 43
    assert row['observation.state'][29] == .65
    assert row['action'][29] == np.deg2rad(40)
    assert row['observation.state'][31:] == [.7]*6+[.8]*6
    assert row['action'][31:] == [.2]*6+[.4]*6
    assert row['input.pose.json'] is None
    assert not row['timing.pose.fresh']
    del samples['camera']
    assert 'camera_missing_or_stale' in problems(samples, NOW)


@pytest.mark.parametrize('fault,expected', [
    ('offline', 'hand_state_offline'), ('nan', 'hand_state_invalid_position'),
    ('token', 'motion_token_missing_or_invalid'), ('crc', 'lowstate_bad_crc'),
    ('legacy', 'restart_updated_pico_manager'),
])
def test_bad_sources_not_accepted(fault, expected):
    samples = snapshots()
    if fault == 'offline':
        samples['hand_state'].data['lost'][0] = 1
    elif fault == 'nan':
        samples['hand_state'].data['q'][0] = float('nan')
    elif fault == 'token':
        samples['g1_debug'].data['token_state'] = []
    elif fault == 'crc':
        samples['lowstate'].data['crc_ok'] = False
    elif fault == 'legacy':
        del samples['manager_state'].data['record_toggle_count']
    assert expected in problems(samples, NOW)


def test_stale_and_inactive_pose_mask():
    samples = snapshots()
    samples['pose'] = Sample({'smpl_pose': [1., 2.]}, NOW-1_000_000_000, 0)
    assert problems(samples, NOW) == []  # planner mode doesn't pretend stale SMPL is active
    samples['manager_state'].data['stream_mode'][0] = 1
    assert 'pose_missing_or_stale' in problems(samples, NOW)
    assert 'camera_missing_or_stale' in problems(samples, NOW+400_000_000)


@pytest.mark.parametrize('status', ['saved', 'discarded', 'incomplete'])
def test_real_parquet_video_raw_roundtrip(tmp_path, status):
    samples = snapshots()
    writer = EpisodeWriter(tmp_path, dict(task='offline synthetic test'), 50, 64, 48)
    wire = build_head_pose_message(.7, 0, 8)
    writer.put('raw', dict(topic='head_pose', wire=wire, arrival_monotonic_ns=NOW))
    for i in range(6):
        # Optional data appearing after row 0 must retain string type, not Arrow null.
        if i == 2:
            samples['pose'] = Sample(dict(smpl_pose=np.ones((10, 21, 3))), NOW, 0)
        row = make_row(samples, NOW+i*20_000_000, NOW, i, 50)
        image = np.zeros((48, 64, 3), np.uint8)
        image[:, :, 2] = 200  # BGR red: catch accidental channel swap
        writer.put('frame', (row, image))
    path = writer.finish(status, 'success' if status == 'saved' else 'unlabeled')
    meta = json.loads((path/'metadata.json').read_text())
    assert meta['status'] == status and meta['frames'] == 6
    assert not path.name.endswith('.partial')
    rows = pq.read_table(path/'frames.parquet').to_pylist()
    assert len(rows) == 6 and rows[-1]['timestamp'] == .1
    assert rows[0]['input.pose.json'] is None
    assert np.asarray(json.loads(rows[2]['input.pose.json'])['smpl_pose']).shape == (10, 21, 3)
    with av.open(str(path/'head_rgb.mp4')) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 6
    assert float(frames[-1].time) == pytest.approx(.1)
    rgb = frames[0].to_ndarray(format='rgb24')
    assert rgb[:, :, 0].mean() > 190 and rgb[:, :, 2].mean() < 10
    with (path/'raw_zmq.msgpack').open('rb') as f:
        raw = list(msgpack.Unpacker(f, raw=False))
    assert unpack_wire(raw[0]['wire'])[0] == 'head_pose'
    assert meta['repeated_camera_frames'] == 5


def test_writer_failure_does_not_publish_valid_episode(tmp_path):
    writer = EpisodeWriter(tmp_path, dict(task='synthetic'), 50, 64, 48)
    row = make_row(snapshots(), NOW, NOW, 0, 50)
    writer.put('frame', (row, np.zeros((24, 32, 3), np.uint8)))
    path = writer.finish('saved')
    assert path.name.endswith('.partial')
    assert json.loads((path/'metadata.json').read_text())['status'] == 'incomplete'


def test_config_mismatch_rejected(tmp_path):
    config = dict(robot_profile='h2_native', command_motor_count=31, h2_vr_head_control=True,
                  encoder_file=str(tmp_path/'model_encoder_full.onnx'), model_path=str(tmp_path/'model_decoder.onnx'))
    assert config_problem(config, tmp_path) is None
    assert config_problem(config, tmp_path/'other')
    config['robot_profile'] = 'g1'
    assert config_problem(config, tmp_path) == 'wrong_robot_profile'


def test_hub_preserves_raw_and_event_counts():
    hub = Hub()
    hub.put('manager_state', manager(), raw=pack_pose_message(manager(), topic='manager_state'))
    hub.put('manager_state', manager(1))
    assert hub.events[0][0] == 'toggle'
    assert hub.samples['manager_state'].sequence == 2
    assert 'wire' in hub.raw_latest['manager_state']


@pytest.mark.parametrize("fail_kind", [None, "raw", "frame"])
def test_offline_complete_collection_lifecycle(tmp_path, monkeypatch, fail_kind):
    """Run real main loop + encoders using synthetic inputs; replace EVERY hardware input."""
    import time
    from gear_sonic.data.h2_collection import collector
    from gear_sonic.scripts.inspect_h2_vla import inspect_episode
    model = tmp_path/'model'
    model.mkdir()
    for name in ('model_encoder_full.onnx', 'model_decoder.onnx'):
        (model/name).write_bytes(b'offline fixture; not an executable model')

    class FakeDDS:
        def __init__(self, hub, *args):
            hub.config = dict(robot_profile='h2_native', command_motor_count=31, h2_vr_head_control=True,
                              encoder_file=str(model/'model_encoder_full.onnx'), model_path=str(model/'model_decoder.onnx'))
        def close(self):
            pass

    def synthetic_inputs(hub, *args):
        start = time.monotonic()
        values = snapshots()
        while not hub.stopping.is_set():
            elapsed = time.monotonic()-start
            count = (0 if elapsed < .2 else (1 if elapsed < .6 else 2)) if fail_kind is None else (
                0 if elapsed < .2 else (1 if elapsed < .45 else (2 if elapsed < .8 else 3)))
            values['manager_state'].data.update(manager(count))
            for name, sample in values.items():
                raw = pack_pose_message(sample.data, topic=name) if name in ('manager_state', 'head_pose', 'planner') else None
                hub.put(name, sample.data, raw=raw)
            if elapsed > (.8 if fail_kind is None else 1.0):
                hub.stopping.set()
            hub.stopping.wait(.005)

    # main's port ownership probe is also fake; this test opens no network sockets.
    class FakeSocket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def bind(self, *args): pass
    if fail_kind:
        original_writer = collector.EpisodeWriter
        created = []
        class FailFirstWriter(original_writer):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                created.append(self)
            def put(self, kind, value):
                if self is created[0] and self.frames >= 3 and kind == fail_kind:
                    raise RuntimeError('synthetic writer queue full')
                return super().put(kind, value)
        monkeypatch.setattr(collector, 'EpisodeWriter', FailFirstWriter)
    monkeypatch.setattr(collector, 'DDSInput', FakeDDS)
    monkeypatch.setattr(collector, 'zmq_input', synthetic_inputs)
    monkeypatch.setattr(collector, 'camera_input', lambda *a: None)
    monkeypatch.setattr(collector.socket, 'socket', lambda *a, **k: FakeSocket())
    monkeypatch.setattr(collector.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(collector.cv2, 'destroyAllWindows', lambda: None)
    rc = collector.main(['--root', str(tmp_path/'data'), '--task', 'offline test', '--model', str(model),
                         '--headless', '--width', '64', '--height', '48'])
    assert rc == 0
    episodes = sorted((tmp_path/'data').glob('episode_*'))
    assert len(episodes) == (2 if fail_kind else 1)
    if fail_kind:
        assert inspect_episode(episodes[0])['status'] == 'incomplete'
    result = inspect_episode(episodes[-1])
    assert result['training_eligible'], result
    assert result['outcome'] == 'unlabeled'
    assert result['frames'] > 5
    meta = json.loads((episodes[-1]/'metadata.json').read_text())
    assert meta['end_event_monotonic_ns'] > meta['start_event_monotonic_ns']


def test_manager_status_burst_is_bounded_without_losing_button_edges():
    from gear_sonic.utils.teleop.manager_state_rate import ManagerStateRateLimiter
    limiter = ManagerStateRateLimiter(50)
    sends = [i for i in range(10000) if limiter.due(i/10000)]
    assert 49 <= len(sends) <= 50
    # Two distinct button edges inside one 20ms interval must both be sent.
    assert limiter.due(1.001, force=True)
    assert limiter.due(1.002, force=True)
    assert not limiter.due(1.003)
    assert limiter.due(1.023)


def test_writer_backpressure_is_episode_scoped_and_does_not_flood_errors():
    hub = Hub()
    class FullWriter:
        calls = 0
        def put(self, *args):
            self.calls += 1
            raise RuntimeError('writer queue full')
    broken = FullWriter()
    hub.writer = broken
    for _ in range(1000):
        hub.put('planner', {}, raw=b'fixture')
    assert broken.calls == 1
    assert hub.episode_errors == {'writer queue full': 1}
    assert hub.errors == []
    assert hub.counts['writer.raw_not_queued'] == 1000
    assert hub.detach_writer() == {'writer queue full': 1}
    assert not hub.episode_errors and hub.writer is None
    class NewWriter:
        calls = 0
        def put(self, *args): self.calls += 1
    fresh = NewWriter()
    hub.writer = fresh
    hub.put('planner', {}, raw=b'fixture')
    assert fresh.calls == 1 and not hub.errors
