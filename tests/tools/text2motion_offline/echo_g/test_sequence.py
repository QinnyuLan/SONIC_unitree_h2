"""Composition keeps speech intact and removes discontinuities between utterances."""
import wave
from argparse import Namespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from interface.motion_replay import Motion
from interface.motion_replay.playback import ReplayBundle, wav_duration
from text2motion_offline.echo_g.cli import CHECKPOINT_SHA256
from text2motion_offline.echo_g.sequence import (LEG_INDICES, STANDING_LEGS,
                                              compose_bundles, generate_script, paragraphs)


def utterance(path, value, yaw, samples=1840):
    path.mkdir()
    q = np.full((6, 29), value, dtype="f4")
    root = Rotation.from_euler("xyz", [[.03, .04, yaw]] * 6).as_quat()[:, [3, 0, 1, 2]]
    np.savez(path / "motion.npz", joint_pos=q, root_quat=root,
             fps=50, reference_type="g1")
    pcm = np.arange(samples, dtype="<i2")
    with wave.open(str(path / "speech.wav"), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(pcm.tobytes())
    ReplayBundle.write(path, metadata={"generator": "ECHO-G", "language": "en",
                                      "seed": 0, "checkpoint_sha256": CHECKPOINT_SHA256})
    return pcm


def test_composition_preserves_pcm_and_timing_without_motion_or_heading_jumps(tmp_path):
    first, second, output = [tmp_path / name for name in ("first", "second", "output")]
    pcm1 = utterance(first, 0, np.deg2rad(170))
    pcm2 = utterance(second, 1, np.deg2rad(-170))
    bundle = compose_bundles([first, second], output, text="First.\n\nSecond.",
                             language="en", voice="test", pause=1)
    motion = Motion.load(bundle.motion)
    with wave.open(str(bundle.audio), "rb") as stream:
        pcm = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
    second_start = 6 * 320 + 16000
    np.testing.assert_array_equal(pcm[:len(pcm1)], pcm1)
    np.testing.assert_array_equal(pcm[second_start:second_start + len(pcm2)], pcm2)
    np.testing.assert_array_equal(pcm[len(pcm1):second_start], 0)
    assert wav_duration(bundle.audio) == pytest.approx(len(motion.q) / 50)
    assert bundle.metadata["metadata"]["segments"][1]["start_seconds"] == pytest.approx(1.12)
    assert "00:00:01,120 --> 00:00:01,235" in (output / "subtitles.srt").read_text()
    np.testing.assert_array_equal(motion.q[:6], 0)
    np.testing.assert_array_equal(motion.q[-6:], 1)
    assert np.max(np.abs(np.diff(motion.q, axis=0))) < .04
    np.testing.assert_allclose(motion.dq[-1], 0)
    rotations = Rotation.from_quat(motion.root[:, [1, 2, 3, 0]])
    np.testing.assert_allclose(rotations.as_euler("xyz"), [[.03, .04, np.deg2rad(170)]] * len(motion.q), atol=1e-7)
    assert np.all(np.sum(motion.root[:-1] * motion.root[1:], axis=1) >= 0)


def test_composition_rejects_mismatched_speech_and_missing_paragraphs(tmp_path):
    source = tmp_path / "source"
    utterance(source, 0, 0, samples=16000)
    with pytest.raises(ValueError, match="durations"):
        compose_bundles([source], tmp_path / "output", text="Test.", language="en", voice="test")
    with pytest.raises(ValueError, match="Each script paragraph"):
        compose_bundles([source], tmp_path / "output", text="First.\n\nSecond.", language="en", voice="test")


def test_transition_cannot_overshoot_even_with_fast_endpoint_motion(tmp_path):
    sources = [tmp_path / name for name in ("first", "second")]
    for source, value in zip(sources, (.3, .5)):
        utterance(source, value, 0)
        with np.load(source / "motion.npz") as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["joint_pos"][:2] = [np.full(29, value), np.full(29, value + .15)]
        arrays["joint_pos"][-2:] = [np.full(29, value + .15), np.full(29, value)]
        np.savez(source / "motion.npz", **arrays)
        ReplayBundle.write(source, metadata={"generator": "ECHO-G", "language": "en",
                                            "seed": 0, "checkpoint_sha256": CHECKPOINT_SHA256})
    bundle = compose_bundles(sources, tmp_path / "output", text="First.\n\nSecond.",
                             language="en", voice="test")
    bridge = Motion.load(bundle.motion).q[6:-6]
    assert np.all((bridge >= .3) & (bridge <= .5))
    assert np.all(np.diff(bridge, axis=0) >= 0)


def test_standing_preserves_gestures_audio_and_uses_semantic_leg_order(tmp_path):
    source = tmp_path / "source"
    utterance(source, .1, .7)
    full = compose_bundles([source], tmp_path / "full", text="Hello.", language="en", voice="test")
    standing = compose_bundles([source], tmp_path / "standing", text="Hello.", language="en",
                               voice="test", body_mode="standing")
    assert full.audio.read_bytes() == standing.audio.read_bytes()
    motion = Motion.load(standing.motion)
    np.testing.assert_allclose(motion.q[:, LEG_INDICES], np.tile(STANDING_LEGS, (6, 1)))
    np.testing.assert_allclose(motion.q[:, np.setdiff1d(np.arange(29), LEG_INDICES)], .1)
    np.testing.assert_allclose(motion.dq[:, LEG_INDICES], 0)
    np.testing.assert_array_equal(motion.root, np.tile([1, 0, 0, 0], (6, 1)))
    assert standing.metadata["metadata"]["body_mode"] == "standing"


def test_script_paragraphs_keep_punctuation_and_non_latin_text():
    assert paragraphs("  大家好！\n\n\n欢迎。\n") == ["大家好！", "欢迎。"]
    with pytest.raises(ValueError, match="empty"):
        paragraphs(" \n\n ")


def test_script_generation_sends_complete_paragraphs_and_increments_seeds(tmp_path, monkeypatch):
    source = tmp_path / "script.txt"
    source.write_text("First paragraph!\n\nSecond paragraph.")
    calls = []

    def generate(args):
        calls.append((args.text, args.seed))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        utterance(args.output, len(calls) * .1, 0)

    monkeypatch.setattr("text2motion_offline.echo_g.cli.generate", generate)
    output = tmp_path / "complete"
    generate_script(Namespace(text_file=source, output=output, language="en",
                              voice=None, seed=4, pause=1, body_mode="standing"))
    assert calls == [("First paragraph!", 4), ("Second paragraph.", 5)]
    bundle = ReplayBundle.load(output)
    assert len(bundle.metadata["metadata"]["segments"]) == 2
    assert bundle.metadata["metadata"]["body_mode"] == "standing"
    assert (output / "script.txt").read_text() == source.read_text() + "\n"
