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
from interface.motion_replay.playback import ReplayBundle, sha256, wav_duration
from interface.motion_replay import Motion

SOURCE_REVISION = "60b85fb932b124b1a2ddd0868f4917a73c383901"
MODELS = ROOT / "models/text2motion_echo_g"
CHECKPOINT_SHA256 = "c84f31fce140cdc8c3be5e5dbc8cb8eb3a82ce1865e2d84026a0791d4735928d"
VOICES = {"zh": "zh-CN-XiaoxiaoNeural", "en": "en-US-JennyNeural"}
# ECHO-G canonical joint order -> native G1 semantic joint order.
CANONICAL_TO_POLICY = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
                       16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]


def download_models(directory=MODELS, *, checkpoint_only=False, verify_only=False):
    """Fetch pinned runtime files only; no training data or exported ONNX model."""
    directory = Path(directory)
    manifest = json.loads((MODELS / "encoders.json").read_text())
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
    import torch
    from .features import load_manifest, extract_features
    from .model import predict
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requires a CUDA-enabled PyTorch environment")
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
        source_record = json.loads(manifest.read_text(encoding="utf-8").strip())
        utterance = load_manifest(manifest)
        if utterance["audio_path"].resolve() != (output / "speech.wav").resolve():
            shutil.copyfile(utterance["audio_path"], output / "speech.wav")
        record = {**utterance, "audio_path": "speech.wav"}
        (output / "utterance.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n")
        utterance["audio_path"] = output / "speech.wav"
        print("Extracting speech and word-aligned text features...", flush=True)
        condition_data = extract_features(utterance, args.models, args.device)
        condition_dir = output / "generation/conditions"
        condition_dir.mkdir(parents=True)
        condition = condition_dir / f"{utterance['stem']}.pt"
        torch.save(condition_data, condition)
        transcript = utterance["transcript"]
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
    if condition_data["stem"] != condition.stem:
        raise ValueError("Condition filename must match its recorded utterance stem")
    print("Generating G1 gesture from the ECHO-G EMA checkpoint...", flush=True)
    physical = predict(checkpoint, condition_data, seed=args.seed, device=args.device)
    prediction = output / "generation/predictions" / f"seed_{args.seed:03d}" / f"{condition.stem}.pt"
    prediction.parent.mkdir(parents=True)
    torch.save({"robot_repr": physical, "representation_units": "physical", "fps": 30.,
                "stem": condition.stem, "seed": args.seed, "integration_steps": 8,
                "guidance_scale": 1., "source_checkpoint_sha256": CHECKPOINT_SHA256,
                "source_condition_sha256": sha256(condition)}, prediction)
    reference = convert_motion(physical.numpy())
    np.savez(output / "motion.npz", **reference)
    motion = Motion.load(output / "motion.npz")
    if abs(len(motion.times) / motion.fps - seconds) > .05:
        raise ValueError("Generated motion and audio durations differ by more than 50 ms")
    metadata = {"generator": "ECHO-G", "source_revision": SOURCE_REVISION,
                "checkpoint_sha256": CHECKPOINT_SHA256, "seed": args.seed,
                "reference_type": "g1", "language": args.language, "transcript": transcript,
                "tts_voice": ((args.voice or VOICES[args.language]) if text is not None
                              else source_record.get("tts_voice") if manifest else None),
                "input_text": text if text is not None else source_record.get("transcript") if manifest else None,
                "audio_seconds": seconds, "motion_frames": len(motion.times), "motion_fps": motion.fps}
    ReplayBundle.write(output, metadata=metadata)
    print(f"Replay bundle: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(prog="python -m text2motion_offline.echo_g", description=__doc__)
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
    script = commands.add_parser("generate-script", help="generate paragraphs and compose one complete replay bundle")
    script.add_argument("--text-file", type=Path, required=True, help="UTF-8 script, one utterance per blank-line-separated paragraph")
    script.add_argument("--models", type=Path, default=MODELS)
    script.add_argument("--output", type=Path, required=True)
    script.add_argument("--language", choices=VOICES, default="zh")
    script.add_argument("--voice")
    script.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    script.add_argument("--seed", type=int, default=0, help="first paragraph seed; incremented for each subsequent paragraph")
    script.add_argument("--threads", type=int, default=4)
    script.add_argument("--pause", type=float, default=1.0, help="silence and motion transition between paragraphs, seconds")
    script.add_argument("--body-mode", choices=["full-body", "standing"], default="full-body",
                        help="keep generated full-body motion, or preserve waist/arms with a fixed standing leg reference")
    args = parser.parse_args()
    try:
        if args.command == "download":
            download_models(args.models, checkpoint_only=args.checkpoint_only, verify_only=args.verify_only)
        else:
            if args.threads < 1 or args.seed < 0:
                raise ValueError("threads must be positive and seed nonnegative")
            if args.command == "generate-script":
                from .sequence import generate_script
                generate_script(args)
                return
            if args.audio is not None and args.conditions is None:
                raise ValueError("--audio is only used with --conditions; use --manifest for existing speech")
            generate(args)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"ECHO-G: {error}\n")


if __name__ == "__main__":
    main()
