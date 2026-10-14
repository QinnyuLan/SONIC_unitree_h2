"""Offline conversion and replay bundles require no generation models or devices."""
import json
from pathlib import Path
import shlex
import shutil
import sys
import wave

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from interface.playback import AudioPlayer, ReplayBundle, wav_duration
from interface.replay import Motion
from interface.text2motion.echo_g import convert_motion
from interface.paths import ROOT


def wav(path, seconds=.4):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(np.zeros(round(seconds * 16000), dtype='<i2').tobytes())


def bundle(path):
    path.mkdir(exist_ok=True)
    np.savez(path / 'motion.npz', joint_pos=np.zeros((20, 29)),
             root_quat=np.tile([1, 0, 0, 0], (20, 1)), fps=50, reference_type='g1')
    wav(path / 'speech.wav')
    return ReplayBundle.write(path)


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


def test_bundle_can_move_and_detects_different_audio_of_same_duration(tmp_path):
    bundle(tmp_path / 'original')
    shutil.copytree(tmp_path / 'original', tmp_path / 'moved')
    loaded = ReplayBundle.load(tmp_path / 'moved')
    assert loaded.audio == tmp_path / 'moved/speech.wav'
    assert Motion.load(loaded.motion).reference_type == 'g1'
    data = bytearray(loaded.audio.read_bytes())
    data[-2:] = b'\x01\x00'
    loaded.audio.write_bytes(data)
    assert wav_duration(loaded.audio) == pytest.approx(.4)
    with pytest.raises(ValueError, match='audio checksum'):
        ReplayBundle.load(tmp_path / 'moved')


def test_bundle_rejects_motion_corruption_and_external_paths(tmp_path):
    b = bundle(tmp_path)
    b.motion.write_bytes(b.motion.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='motion checksum'):
        ReplayBundle.load(tmp_path)
    manifest = json.loads((tmp_path / 'replay.json').read_text())
    manifest['motion']['file'] = '../elsewhere.npz'
    (tmp_path / 'replay.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='path'):
        ReplayBundle.load(tmp_path)


def test_replay_inspect_and_mute_do_not_initialize_audio(monkeypatch, tmp_path):
    import interface.replay as replay
    bundle(tmp_path)
    monkeypatch.setattr(replay, 'AudioPlayer', lambda *a, **k: pytest.fail('must not start audio'))
    monkeypatch.setattr('sys.argv', ['replay', str(tmp_path), '--inspect'])
    replay.main()
    monkeypatch.setattr('sys.argv', ['replay', str(tmp_path), '--mute', '--inspect', '--speed', '2'])
    replay.main()


@pytest.mark.parametrize('flag', [['--speed', '2'], ['--loop'], ['--fps', '50']])
def test_audio_replay_rejects_timing_changes(monkeypatch, tmp_path, flag):
    import interface.replay as replay
    bundle(tmp_path)
    monkeypatch.setattr('sys.argv', ['replay', str(tmp_path), '--inspect', *flag])
    with pytest.raises(SystemExit) as error:
        replay.main()
    assert error.value.code == 2


def test_audio_ready_handshake_has_no_playback_before_play(tmp_path):
    audio = tmp_path / 'speech.wav'
    wav(audio)
    marker = tmp_path / 'played'
    script = tmp_path / 'player.py'
    script.write_text("import sys\nfrom pathlib import Path\nprint('READY', flush=True)\n"
                      "assert sys.stdin.readline().strip() == 'PLAY'\n"
                      f"Path({str(marker)!r}).write_text('played')\n")
    player = AudioPlayer(audio, shlex.join([sys.executable, str(script)]), ready=True)
    try:
        player.prepare(timeout=3)
        assert not marker.exists()
        player.start()
        player.finish()
        assert marker.read_text() == 'played'
    finally:
        player.close()


def test_replay_starts_audio_after_first_motion_and_releases_on_player_failure(monkeypatch, tmp_path):
    import interface.replay as replay
    bundle(tmp_path)
    events = []

    class Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *exc): events.append('released')
        def send(self, kind, data):
            events.append('motion')
            assert kind == 'g1'
            return True

    class Player:
        def __init__(self, *a, **k): self.started = False
        def prepare(self): events.append('ready')
        def start(self): self.started = True; events.append('audio')
        def check(self):
            if self.started: raise RuntimeError('player failed')
        def close(self): events.append('closed')

    monkeypatch.setattr(replay, 'RuntimeClient', Client)
    monkeypatch.setattr(replay, 'AudioPlayer', Player)
    monkeypatch.setattr('sys.argv', ['replay', str(tmp_path)])
    with pytest.raises(RuntimeError, match='player failed'):
        replay.main()
    assert events.index('ready') < events.index('motion') < events.index('audio')
    assert events[-2:] == ['released', 'closed']


def test_audio_player_failure_before_ready_stops_before_motion(monkeypatch, tmp_path):
    import interface.replay as replay
    bundle(tmp_path)
    monkeypatch.setattr(replay, 'RuntimeClient', lambda *a, **k: pytest.fail('must not connect motion'))
    command = shlex.join([sys.executable, '-c', 'raise SystemExit(3)'])
    monkeypatch.setattr('sys.argv', ['replay', str(tmp_path), '--audio-ready', '--audio-command', command])
    with pytest.raises(RuntimeError):
        replay.main()


@pytest.mark.parametrize('language', ['zh', 'en'])
def test_published_speech_gesture_bundles(language):
    assets = ReplayBundle.load(ROOT / 'motions/example_text2motion_replay' / f'echo_g_{language}')
    motion = Motion.load(assets.motion)
    metadata = assets.metadata['metadata']
    assert motion.reference_type == metadata['reference_type'] == 'g1'
    assert motion.fps == metadata['motion_fps'] == 50
    assert len(motion.times) == metadata['motion_frames']
    assert metadata['language'] == language
    assert metadata['input_text'] and metadata['tts_voice']
    assert abs(len(motion.times) / motion.fps - wav_duration(assets.audio)) < .05
    motion.sample(.2)


def test_echo_root_resampling_takes_short_arc_across_yaw_wrap():
    representation = np.zeros((30, 39))
    rotations = Rotation.from_euler('z', np.linspace(170, 190, 30), degrees=True)
    representation[:, :6] = rotations.as_matrix()[:, :2, :].reshape(30, 6)
    reference = convert_motion(representation)
    result = Rotation.from_quat(reference['root_quat'][:, [1, 2, 3, 0]])
    angles = np.unwrap(result.as_euler('xyz')[:, 2])
    assert np.max(np.abs(np.diff(angles))) < .02
    np.testing.assert_allclose(angles[[0, -1]], np.deg2rad([170, 190]), atol=1e-6)
