"""Compose independently generated utterances into one timed speech/gesture bundle."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import wave

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from interface.motion_replay import Motion
from interface.motion_replay.playback import ReplayBundle, sha256, wav_duration

FPS = 50
SAMPLES_PER_FRAME = 320  # 16 kHz PCM at 50 Hz.
# SONIC semantic order: paired hips, knees and ankles, not canonical leg order.
LEG_INDICES = np.array([0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18])
STANDING_LEGS = np.array([-.312, -.312, 0, 0, 0, 0, .669, .669, -.363, -.363, 0, 0])


def paragraphs(text):
    parts = [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]
    if not parts:
        raise ValueError("Speech script is empty")
    return parts


def subtitle_time(seconds):
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def compose_bundles(directories, output, *, text, language, voice, pause=1.0,
                    body_mode="full-body"):
    """Preserve each speech waveform; insert silence and joint/rotation transitions."""
    from .cli import CHECKPOINT_SHA256, SOURCE_REVISION

    parts = paragraphs(text)
    if len(parts) != len(directories):
        raise ValueError("Each script paragraph needs one generated bundle")
    if not math.isfinite(pause) or pause < .2:
        raise ValueError("The inter-paragraph pause must be at least 0.2 seconds")
    if body_mode not in ("full-body", "standing"):
        raise ValueError("body_mode must be full-body or standing")
    gap = round(pause * FPS)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("replay.json", "motion.npz", "speech.wav")):
        raise ValueError("The composed bundle already exists")
    joint_parts, root_parts, audio_parts, segments, subtitles = [], [], [], [], []
    frame_count = 0
    for index, (directory, paragraph) in enumerate(zip(directories, parts)):
        bundle = ReplayBundle.load(directory)
        motion = Motion.load(bundle.motion)
        metadata = bundle.metadata["metadata"]
        if (motion.reference_type != "g1" or motion.fps != FPS or bundle.audio is None
                or metadata.get("generator") != "ECHO-G"
                or metadata.get("checkpoint_sha256") != CHECKPOINT_SHA256
                or metadata.get("language") != language):
            raise ValueError("Expected matching ECHO-G G1 speech bundles at 50 Hz")
        seconds = wav_duration(bundle.audio)
        if abs(len(motion.q) / FPS - seconds) > .05:
            raise ValueError("Segment motion and speech durations differ by more than 50 ms")
        with wave.open(str(bundle.audio), "rb") as stream:
            audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
        frames = max(len(motion.q), math.ceil(len(audio) / SAMPLES_PER_FRAME))
        q = np.pad(motion.q, ((0, frames - len(motion.q)), (0, 0)), mode="edge")
        roots = np.pad(motion.root, ((0, frames - len(motion.root)), (0, 0)), mode="edge")
        if body_mode == "standing":
            q[:, LEG_INDICES] = STANDING_LEGS
            roots[:] = [1, 0, 0, 0]
        if index:
            previous_q, previous_root = joint_parts[-1], root_parts[-1]
            previous_rotation = Rotation.from_quat(previous_root[-1, [1, 2, 3, 0]])
            rotations = Rotation.from_quat(roots[:, [1, 2, 3, 0]])
            a, b = previous_rotation.as_matrix(), rotations[0].as_matrix()
            heading = np.arctan2(a[1, 0], a[0, 0]) - np.arctan2(b[1, 0], b[0, 0])
            rotations = Rotation.from_euler("z", heading) * rotations
            roots = rotations.as_quat()[:, [3, 0, 1, 2]]
            duration = (gap + 1) / FPS
            times = np.arange(1, gap + 1) / FPS
            u = times / duration
            ease = u**3 * (10 - 15 * u + 6 * u**2)
            # Convex interpolation stays inside the endpoint angle interval.
            # Matching sampled endpoint velocities can overshoot joint limits.
            joint_parts.append(previous_q[-1] + ease[:, None] * (q[0] - previous_q[-1]))
            endpoints = Rotation.from_quat(np.stack((previous_root[-1], roots[0]))[:, [1, 2, 3, 0]])
            root_parts.append(Slerp([0, 1], endpoints)(ease).as_quat()[:, [3, 0, 1, 2]])
            audio_parts.append(np.zeros(gap * SAMPLES_PER_FRAME, dtype="<i2"))
            frame_count += gap
        start = frame_count / FPS
        segments.append({"index": index + 1, "text": paragraph, "start_seconds": start,
                         "speech_seconds": seconds, "padded_seconds": frames / FPS,
                         "seed": metadata["seed"], "source_motion_sha256": sha256(bundle.motion),
                         "source_audio_sha256": sha256(bundle.audio)})
        subtitles.append(f"{index + 1}\n{subtitle_time(start)} --> {subtitle_time(start + seconds)}\n{paragraph}\n")
        joint_parts.append(q)
        root_parts.append(roots)
        audio_parts.append(np.pad(audio, (0, frames * SAMPLES_PER_FRAME - len(audio))))
        frame_count += frames
    joints, roots = np.concatenate(joint_parts), np.concatenate(root_parts)
    # Keep quaternion signs continuous for downstream consumers as well as SLERP.
    for index in range(1, len(roots)):
        if np.dot(roots[index - 1], roots[index]) < 0:
            roots[index] *= -1
    velocity = np.gradient(joints, 1 / FPS, axis=0)
    velocity[-1] = 0
    np.savez_compressed(output / "motion.npz", reference_type=np.asarray("g1"),
                        joint_pos=joints.astype("f4"), joint_vel=velocity.astype("f4"),
                        root_quat=roots.astype("f4"), fps=np.float32(FPS))
    with wave.open(str(output / "speech.wav"), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(np.concatenate(audio_parts).astype("<i2").tobytes())
    (output / "script.txt").write_text(text.strip() + "\n", encoding="utf-8")
    (output / "subtitles.srt").write_text("\n".join(subtitles), encoding="utf-8")
    return ReplayBundle.write(output, metadata={
        "generator": "ECHO-G", "source_revision": SOURCE_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256, "reference_type": "g1",
        "language": language, "tts_voice": voice, "input_text": text.strip(),
        "audio_seconds": frame_count / FPS, "motion_frames": frame_count, "motion_fps": FPS,
        "composition": {"method": "bounded_quintic_joint_bridge_and_heading_aligned_slerp_v1",
                        "pause_seconds": gap / FPS, "audio": "unaltered PCM with silence padding"},
        "body_mode": body_mode,
        "standing_reference": ({"joint_indices_semantic": LEG_INDICES.tolist(),
                                "joint_angles_rad": STANDING_LEGS.tolist(),
                                "root_quat_wxyz": [1, 0, 0, 0],
                                "preserved": "generated waist and arm gestures"}
                               if body_mode == "standing" else None),
        "segments": segments,
    })


def generate_script(args):
    from .cli import generate, VOICES

    text = args.text_file.read_text(encoding="utf-8")
    parts = paragraphs(text)
    if not math.isfinite(args.pause) or args.pause < .2:
        raise ValueError("The inter-paragraph pause must be at least 0.2 seconds")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty output directory for the script")
    directories = []
    for index, paragraph in enumerate(parts):
        directory = output / "generation/segments" / f"{index + 1:02}"
        print(f"Generating paragraph {index + 1}/{len(parts)}...", flush=True)
        segment = argparse.Namespace(**vars(args))
        segment.text, segment.text_file = paragraph, None
        segment.manifest = segment.conditions = segment.audio = None
        segment.output, segment.seed = directory, args.seed + index
        generate(segment)
        directories.append(directory)
    compose_bundles(directories, output, text=text, language=args.language,
                    voice=args.voice or VOICES[args.language], pause=args.pause,
                    body_mode=args.body_mode)
    print(f"Complete speech/gesture bundle: {output}", flush=True)
