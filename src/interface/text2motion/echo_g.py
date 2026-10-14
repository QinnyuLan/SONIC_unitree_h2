"""Offline ECHO-G speech/gesture generation into a portable motion replay bundle."""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from interface.paths import ROOT
from interface.playback import ReplayBundle, sha256, wav_duration
from interface.replay import Motion

SOURCE = ROOT / "third_party/echo_g"
SOURCE_REVISION = "60b85fb932b124b1a2ddd0868f4917a73c383901"
MODELS = ROOT / "models/text2motion_echo_g"
CHECKPOINT_SHA256 = "c84f31fce140cdc8c3be5e5dbc8cb8eb3a82ce1865e2d84026a0791d4735928d"
VOICES = {"zh": "zh-CN-XiaoxiaoNeural", "en": "en-US-JennyNeural"}
# ECHO-G canonical joint order -> native G1 semantic joint order.
CANONICAL_TO_POLICY = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
                       16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]


def verify_source():
    if not (SOURCE / "pyproject.toml").is_file():
        raise ValueError("ECHO-G source is missing; run ./install.sh text2motion")
    revision = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCE_REVISION:
        raise ValueError("ECHO-G source revision differs from the supported release")


def download_models(directory=MODELS, *, checkpoint_only=False, verify_only=False):
    """Fetch pinned runtime files only; no training data or exported ONNX model."""
    verify_source()
    directory = Path(directory)
    manifest = json.loads((SOURCE / "manifests/condition_encoders.json").read_text())
    jobs = [(directory / "best.pt", CHECKPOINT_SHA256,
             "https://huggingface.co/gaopusen/ECHO-G/resolve/v0.2.0/best.pt")]
    if not checkpoint_only:
        for encoder in manifest["encoders"].values():
            for filename, identity in encoder["files"].items():
                path = directory / Path(encoder["local_dir"]).name / filename
                url = f"https://huggingface.co/{encoder['repo_id']}/resolve/{encoder['revision']}/{filename}"
                jobs.append((path, identity["sha256"], url))

    def fetch(job):
        path, digest, url = job
        if path.is_file() and sha256(path) == digest:
            return
        if verify_only:
            raise ValueError(f"Missing or mismatched model file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        # A failed download leaves a resumable partial; existing model files are
        # replaced only after the expected release hash has been verified.
        partial = path.with_name(path.name + ".partial")
        subprocess.run(["curl", "-fL", "--retry", "3", "--connect-timeout", "30",
                        "--speed-time", "60", "--speed-limit", "1024", "-C", "-",
                        "--output", str(partial), url], check=True)
        if sha256(partial) != digest:
            raise ValueError(f"Model hash mismatch; remove {partial} before retrying")
        partial.replace(path)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(fetch, jobs))
    print(f"Verified {len(jobs)} model files in {directory}", flush=True)


def convert_motion(representation, fps=30):
    """Convert physical 39D ECHO-G output to the standard 50 Hz G1 replay NPZ."""
    values = np.asarray(representation, dtype=np.float64)
    if fps != 30 or values.ndim != 2 or values.shape[1] != 39 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite physical ECHO-G robot_repr[N>=2,39] at 30 Hz")
    # Upstream 6D rotations contain two matrix ROWS (PyTorch3D convention).
    first, second = values[:, :3].copy(), values[:, 3:6].copy()
    for vector in (first, second):
        if vector is second:
            second -= np.sum(first * second, axis=1, keepdims=True) * first
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        if np.any(norm < 1e-8):
            raise ValueError("Degenerate ECHO-G root rotation")
        vector /= norm
    rotation = Rotation.from_matrix(np.stack((first, second, np.cross(first, second)), axis=1))
    source_times = np.arange(len(values)) / 30
    times = np.arange(round(len(values) * 50 / 30)) / 50
    q = values[:, 10:39][:, CANONICAL_TO_POLICY]
    q = np.column_stack([np.interp(times, source_times, column) for column in q.T])
    roots = Slerp(source_times, rotation)(np.clip(times, 0, source_times[-1])).as_quat()[:, [3, 0, 1, 2]]
    dq = np.gradient(q, .02, axis=0)
    dq[-1] = 0
    return {"reference_type": np.asarray("g1"), "joint_pos": q.astype("f4"),
            "joint_vel": dq.astype("f4"), "root_quat": roots.astype("f4"), "fps": np.float32(50)}


async def synthesize(text, output, language, voice=None):
    import edge_tts
    if not text.strip():
        raise ValueError("Speech text is empty")
    voice = voice or VOICES[language]
    words = []
    with (output / "speech.mp3").open("wb") as audio:
        async for chunk in edge_tts.Communicate(text, voice, boundary="WordBoundary").stream():
            if chunk["type"] == "audio":
                audio.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                words.append({"text": chunk["text"], "start": chunk["offset"] / 10_000_000,
                              "end": (chunk["offset"] + chunk["duration"]) / 10_000_000})
    if not words:
        raise ValueError("TTS returned no word boundaries; cannot align speech and gesture")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(output / "speech.mp3"),
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output / "speech.wav")], check=True)
    manifest = {"stem": "speech", "audio_path": "speech.wav", "transcript": text,
                "words": words, "tts_voice": voice}
    path = output / "utterance.jsonl"
    path.write_text(json.dumps(manifest, ensure_ascii=False) + "\n")
    return path


def generate(args):
    verify_source()
    import torch
    import echo_g
    from transformers.utils import logging as transformers_logging
    from echo_g.conditions import (load_manifest, extract_audio_conditions, extract_text_conditions,
                                   merge_conditions, validate_word_duration)
    from echo_g.inference import export_predictions
    if Path(echo_g.__file__).resolve().parent != (SOURCE / "src/echo_g").resolve():
        raise ValueError("Installed ECHO-G points to another checkout; rerun ./install.sh text2motion")
    torch.set_num_threads(args.threads)
    transformers_logging.disable_progress_bar()
    checkpoint = args.models / "best.pt"
    if not checkpoint.is_file() or sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("ECHO-G checkpoint missing or mismatched; run the download command")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty output directory for each utterance")
    output.mkdir(parents=True, exist_ok=True)
    text = args.text if args.text is not None else args.text_file.read_text() if args.text_file else None
    if text is not None:
        manifest = asyncio.run(synthesize(text, output, args.language, args.voice))
    elif args.manifest:
        manifest = args.manifest.resolve()
    else:
        manifest = None
    if manifest:
        utterances = load_manifest(manifest)
        if len(utterances) != 1:
            raise ValueError("Generate one utterance per bundle")
        utterance = utterances[0]
        audio = utterance.audio_path
        if audio is None:
            raise ValueError("Manifest must name the WAV used for feature extraction")
        seconds = wav_duration(audio)
        if seconds > 20:
            raise ValueError("Split speech into utterances of at most 20 seconds")
        validate_word_duration(utterance, seconds)
        if audio.resolve() != (output / "speech.wav").resolve():
            shutil.copyfile(audio, output / "speech.wav")
        # Rebase external manifests so the complete generation record is portable.
        manifest_record = {"stem": utterance.stem, "audio_path": "speech.wav",
                           "transcript": utterance.transcript, "words": list(utterance.words)}
        (output / "utterance.jsonl").write_text(json.dumps(manifest_record, ensure_ascii=False) + "\n")
        utterances = load_manifest(output / "utterance.jsonl")
        condition_dir = output / "generation/conditions"
        audio_model = args.models / "wav2vec2-large-xlsr-53-english"
        text_model = args.models / "Qwen3.5-4B"
        if not audio_model.is_dir() or not text_model.is_dir():
            raise ValueError("Feature encoders missing; run the download command")
        print("Extracting speech and word-aligned text features...", flush=True)
        extract_audio_conditions(utterances, condition_dir / ".audio", str(audio_model.resolve()), 30, args.device, False, False)
        extract_text_conditions(utterances, condition_dir / ".text", str(text_model.resolve()), -2, 256, args.device, False, False)
        merge_conditions(utterances, condition_dir / ".audio", condition_dir / ".text", condition_dir, False)
        condition = condition_dir / f"{utterance.stem}.pt"
        transcript = utterance.transcript
    else:
        if args.audio is None:
            raise ValueError("--conditions requires the matching --audio WAV")
        condition_dir = output / "generation/conditions"
        condition_dir.mkdir(parents=True)
        condition = condition_dir / args.conditions.name
        shutil.copyfile(args.conditions, condition)
        shutil.copyfile(args.audio, output / "speech.wav")
        transcript = None
    seconds = wav_duration(output / "speech.wav")
    if seconds > 20:
        raise ValueError("Speech must be at most 20 seconds")
    condition_data = torch.load(condition, map_location="cpu", weights_only=True)
    if condition_data.get("source_audio_sha256") != sha256(output / "speech.wav"):
        raise ValueError("Condition features are not bound to the supplied WAV")
    # The upstream dataset validates feature dimensions, token times and provenance.
    print("Generating G1 gesture with the official PyTorch EMA model...", flush=True)
    export_predictions(checkpoint_path=checkpoint, config_path=SOURCE / "configs/sgdit_audio_text.yaml",
                       data_root=None, condition_dir=condition_dir, output_dir=output / "generation/predictions",
                       device_name=args.device, seeds=[args.seed], steps=8, guidance_scale=1.,
                       batch_size=1, num_workers=0)
    prediction = output / "generation/predictions" / f"seed_{args.seed:03d}" / f"{condition.stem}.pt"
    payload = torch.load(prediction, map_location="cpu", weights_only=True)
    if payload.get("representation_units") != "physical":
        raise ValueError("ECHO-G output is not denormalized physical motion")
    reference = convert_motion(payload["robot_repr"].numpy(), float(payload["fps"]))
    np.savez(output / "motion.npz", **reference)
    motion = Motion.load(output / "motion.npz")
    if abs(len(motion.times) / motion.fps - seconds) > .05:
        raise ValueError("Generated motion and audio durations differ by more than 50 ms")
    metadata = {"generator": "ECHO-G", "source_revision": SOURCE_REVISION,
                "checkpoint_sha256": CHECKPOINT_SHA256, "seed": args.seed,
                "reference_type": "g1", "language": args.language, "transcript": transcript,
                "tts_voice": (args.voice or VOICES[args.language]) if text is not None else None,
                "input_text": text,
                "audio_seconds": seconds, "motion_frames": len(motion.times), "motion_fps": motion.fps}
    ReplayBundle.write(output, metadata=metadata)
    print(f"Replay bundle: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="download and verify pinned models")
    download.add_argument("--models", type=Path, default=MODELS)
    download.add_argument("--checkpoint-only", action="store_true")
    download.add_argument("--verify-only", action="store_true")
    generate_parser = commands.add_parser("generate", help="prepare speech and motion; never send robot commands")
    source = generate_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file", type=Path)
    source.add_argument("--manifest", type=Path, help="one timed utterance with an existing WAV")
    source.add_argument("--conditions", type=Path, help="existing upstream condition .pt bound to --audio")
    generate_parser.add_argument("--audio", type=Path, help="matching WAV for --conditions")
    generate_parser.add_argument("--models", type=Path, default=MODELS)
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--language", choices=VOICES, default="zh")
    generate_parser.add_argument("--voice")
    generate_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.command == "download":
            download_models(args.models, checkpoint_only=args.checkpoint_only, verify_only=args.verify_only)
        else:
            if args.threads < 1 or args.seed < 0:
                raise ValueError("threads must be positive and seed nonnegative")
            if args.audio is not None and args.conditions is None:
                raise ValueError("--audio is only used with --conditions; use --manifest for existing speech")
            generate(args)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"ECHO-G: {error}\n")


if __name__ == "__main__":
    main()
