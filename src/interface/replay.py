"""Stream an offline G1, H2 or SMPL motion to the policy runtime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from interface.commands import RuntimeClient, g1_reference, h2_reference, smpl_reference
from interface.commands.protocol import G1_OFFSETS, SMPL_OFFSETS
from interface.playback import AudioPlayer, ReplayBundle, wav_duration

# Native H2 MuJoCo CSV -> body policy order. Head joints 15/16 are excluded.
# This is a permutation only; physical H2 angles are never projected into G1.
H2_POLICY_FROM_MUJOCO = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 17, 24, 5, 11,
    18, 25, 4, 10, 19, 26, 20, 27, 21, 28, 22, 29, 23, 30,
])
REFERENCE_TYPES = {"g1", "h2", "smpl"}


def _frames(value, shape, name):
    data = np.asarray(value, dtype=np.float64)
    if data.ndim != len(shape) + 1 or data.shape[1:] != shape or len(data) < 2:
        raise ValueError(f"{name} requires at least two frames with shape {shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"{name} contains non-finite data")
    return data


def _csv(path):
    # SONIC clips may have a column-name header. Do not discard a malformed data row.
    with path.open() as stream:
        first = stream.readline().strip()
    try:
        [float(cell) for cell in first.split(",")]
        skip = 0
    except ValueError:
        if not all(re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", cell.strip())
                   for cell in first.split(",")):
            raise ValueError(f"Invalid CSV header: {path}")
        skip = 1
    return np.loadtxt(path, delimiter=",", skiprows=skip, ndmin=2)


def _metadata(path):
    """Read portable release manifests and the original SONIC declarations."""
    records = []
    for filename in ("manifest.json", "h2_lora_reference_manifest.json"):
        manifest = path.parent / filename
        if manifest.is_file():
            data = json.loads(manifest.read_text())
            entries = [entry for entry in data.get("motions", [])
                       if entry["motion_name"] == path.name]
            if len(entries) != 1:
                raise ValueError(f"Expected one entry for {path.name} in {manifest}")
            records.extend((entries[0], data))
    for filename in ("reference.json", "h2_lora_reference.json"):
        if (path / filename).is_file():
            records.append(json.loads((path / filename).read_text()))
    metadata = path / "metadata.txt"
    if metadata.is_file():
        contents = metadata.read_text()
        record = {}
        for field, pattern in (("reference_type", r"^Reference type:\s*(\S+)"),
                               ("fps", r"^FPS:\s*(\S+)")):
            match = re.search(pattern, contents, re.MULTILINE | re.IGNORECASE)
            if match:
                record[field] = match[1]
        records.append(record)
        # Some old H2 exports have 29D projected files but no type declaration.
        if "g1_semantic_29d_from_h2" in contents and not (path / "h2_joint_pos.csv").exists():
            if not any(r.get("reference_type") == "smpl" for r in records):
                raise ValueError("H2-derived joint_pos is not native G1 data; native H2 files are missing")
    return records


def _reference_type(requested, records, candidates):
    declared = {str(r["reference_type"]).lower() for r in records if "reference_type" in r}
    if requested is not None:
        declared.add(requested)
    if declared - REFERENCE_TYPES or len(declared) > 1:
        raise ValueError(f"Unsupported or conflicting reference types: {sorted(declared)}")
    if declared:
        return declared.pop()
    if len(candidates) != 1:
        raise ValueError("Ambiguous motion input; declare reference_type or use --type g1|h2|smpl")
    return candidates.pop()


class Motion:
    """G1 or H2 body trajectory, already in its respective 29D policy order."""

    def __init__(self, joint_pos, root_quat, fps: float, joint_vel=None, *, reference_type="g1"):
        if reference_type not in {"g1", "h2"}:
            raise ValueError("Joint trajectories must be g1 or h2")
        self.reference_type = reference_type
        self.q = _frames(joint_pos, (29,), "joint_pos")
        self._timeline(root_quat, fps, len(self.q))
        self.dq = (np.gradient(self.q, 1 / self.fps, axis=0) if joint_vel is None
                   else _frames(joint_vel, (29,), "joint_vel"))
        if self.dq.shape != self.q.shape or not np.isfinite(self.dq).all():
            raise ValueError("joint_vel must match joint_pos frames and contain finite values")

    def _timeline(self, root_quat, fps, count):
        self.root = _frames(root_quat, (4,), "root_quat")
        self.fps = float(fps)
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive")
        if len(self.root) != count:
            raise ValueError("Root and motion frame counts differ")
        norm = np.linalg.norm(self.root, axis=1, keepdims=True)
        if not np.isfinite(norm).all() or np.any(norm < 1e-6):
            raise ValueError("motion contains an invalid root quaternion")
        self.root = self.root / norm
        self.times = np.arange(count) / self.fps
        self.duration = self.times[-1]
        self.rotation = Slerp(self.times, Rotation.from_quat(self.root[:, [1, 2, 3, 0]]))

    @classmethod
    def load(cls, path: Path, fps: float | None = None, reference_type: str | None = None):
        path = Path(path)
        if path.is_dir():
            if any((path / name).is_file() for name in ("manifest.json", "h2_lora_reference_manifest.json")):
                raise ValueError("Pass one clip directory, not the collection root")
            records = _metadata(path)
            candidates = set()
            if (path / "h2_joint_pos.csv").is_file():
                candidates.add("h2")
            if (path / "smpl_joint.csv").is_file():
                candidates.add("smpl")
            if not candidates and (path / "joint_pos.csv").is_file():
                candidates.add("g1")
            kind = _reference_type(reference_type, records, candidates)
            rate = fps if fps is not None else next((r["fps"] for r in records if "fps" in r), 50)
            if kind == "smpl":
                joints = _frames(_csv(path / "smpl_joint.csv"), (72,), "smpl_joint").reshape(-1, 24, 3)
                root = _csv(path / "smpl_root_quat.csv")
                # Original SONIC uses semantic wrist indices 23..28, not SMPL pose angles.
                wrists = _frames(_csv(path / "joint_pos.csv"), (29,), "joint_pos")[:, 23:29]
                return SmplMotion(joints, root, wrists, rate)
            prefix = "h2_" if kind == "h2" else ""
            q = _csv(path / f"{prefix}joint_pos.csv")
            velocity_path = path / f"{prefix}joint_vel.csv"
            dq = _csv(velocity_path) if velocity_path.exists() else None
            root = _csv(path / "body_quat.csv")[:, :4]
        else:
            with np.load(path, allow_pickle=False) as data:
                records = [{"reference_type": str(data["reference_type"].item())}] if "reference_type" in data else []
                candidates = {kind for key, kind in (("h2_joint_pos", "h2"), ("joints", "smpl")) if key in data}
                if not candidates and "joint_pos" in data:
                    candidates.add("g1")
                kind = _reference_type(reference_type, records, candidates)
                rate = fps if fps is not None else float(data["fps"])
                root = data["root_quat"]
                if kind == "smpl":
                    return SmplMotion(data["joints"], root, data["wrist_joints"], rate)
                prefix = "h2_" if kind == "h2" else ""
                q = data[f"{prefix}joint_pos"]
                dq = data[f"{prefix}joint_vel"] if f"{prefix}joint_vel" in data else None
        if kind == "h2":
            q = _frames(q, (31,), "h2_joint_pos")[:, H2_POLICY_FROM_MUJOCO]
            if dq is not None:
                dq = _frames(dq, (31,), "h2_joint_vel")[:, H2_POLICY_FROM_MUJOCO]
        return cls(q, root, rate, dq, reference_type=kind)

    def _sample_times(self, elapsed, speed, offsets):
        if not np.isfinite(elapsed) or not np.isfinite(speed) or speed <= 0:
            raise ValueError("elapsed must be finite and speed positive")
        requested = elapsed + offsets * speed
        return requested, np.clip(requested, 0, self.duration)

    def _interpolate(self, value, times):
        flat = value.reshape(len(value), -1)
        result = np.column_stack([np.interp(times, self.times, col) for col in flat.T])
        return result.reshape((len(times),) + value.shape[1:])

    def _root_at(self, times):
        return self.rotation(times).as_quat()[:, [3, 0, 1, 2]]

    def sample(self, elapsed: float, speed: float = 1):
        requested, times = self._sample_times(elapsed, speed, G1_OFFSETS)
        q = self._interpolate(self.q, times)
        dq = self._interpolate(self.dq, times) * speed
        dq[requested >= self.duration] = 0
        encode = h2_reference if self.reference_type == "h2" else g1_reference
        return encode(q, dq, self._root_at(times))


class SmplMotion(Motion):
    """Root-centered SMPL joints in world axes, with semantic wrist angles."""

    def __init__(self, joints, root_quat, wrist_joints, fps):
        self.reference_type = "smpl"
        self.joints = _frames(joints, (24, 3), "joints")
        self.wrists = _frames(wrist_joints, (6,), "wrist_joints")
        if len(self.wrists) != len(self.joints):
            raise ValueError("SMPL joint and wrist frame counts differ")
        self._timeline(root_quat, fps, len(self.joints))

    def sample(self, elapsed: float, speed: float = 1):
        _, times = self._sample_times(elapsed, speed, SMPL_OFFSETS)
        return smpl_reference(self._interpolate(self.joints, times), self._root_at(times),
                              self._interpolate(self.wrists, times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path, help="CSV clip, NPZ, or directory containing replay.json")
    parser.add_argument("--type", choices=sorted(REFERENCE_TYPES), dest="reference_type",
                        help="type for unlabelled input; must agree with existing metadata")
    parser.add_argument("--fps", type=float, help="override source fps (CSV fallback: 50)")
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5558")
    parser.add_argument("--takeover", action="store_true")
    parser.add_argument("--loop", action="store_true", help="clip must have a continuous loop boundary")
    parser.add_argument("--inspect", action="store_true", help="validate and print metadata without connecting")
    parser.add_argument("--mute", action="store_true", help="replay bundle motion without audio")
    parser.add_argument("--audio-command", default="aplay -q", help="player argv (no shell); WAV path is appended")
    parser.add_argument("--audio-ready", action="store_true", help="player supports READY/PLAY handshake")
    parser.add_argument("--audio-delay", type=float, default=0, help="delay audio relative to motion, seconds")
    args = parser.parse_args()
    try:
        bundle = ReplayBundle.load(args.motion) if (args.motion / "replay.json").is_file() else None
        motion = Motion.load(bundle.motion if bundle else args.motion, args.fps, args.reference_type)
        audio = bundle.audio if bundle and not args.mute else None
        if not np.isfinite(args.speed) or args.speed <= 0:
            raise ValueError("--speed must be positive")
        if not np.isfinite(args.audio_delay) or args.audio_delay < 0:
            raise ValueError("--audio-delay must be finite and nonnegative")
        audio_seconds = wav_duration(audio) if audio else 0
        if audio and abs(audio_seconds - (motion.duration + 1 / motion.fps)) > .05:
            raise ValueError("Bundle audio and motion durations differ by more than 50 ms")
        if audio and (args.speed != 1 or args.loop or args.fps is not None):
            raise ValueError("Audio replay requires original fps, --speed 1 and no --loop; use --mute for motion-only playback")
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    spacing = .02 if motion.reference_type == "smpl" else .1
    print(f"{motion.reference_type.upper()}: {len(motion.times)} frames, {motion.fps:g} Hz, "
          f"{motion.duration:.2f} s; 10 samples at {spacing:g} s intervals")
    if args.inspect:
        if bundle:
            print(f"Bundle verified; audio: {bundle.audio.name if bundle.audio else 'none'}")
        return
    player = AudioPlayer(audio, args.audio_command, ready=args.audio_ready) if audio else None
    try:
        if player:
            player.prepare()
        with RuntimeClient("motion_replay", args.endpoint, takeover=args.takeover) as client:
            if player:
                # Queue the first frame before starting audio. A successful send is
                # transport readiness, not acknowledgement of policy execution.
                deadline = time.monotonic() + 3
                while not client.send(motion.reference_type, motion.sample(0)):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Runtime is not connected; audio was not started")
                    time.sleep(.02)
            started = time.monotonic()
            audio_started = False
            duration = max(motion.duration, audio_seconds + args.audio_delay) if audio else motion.duration
            while True:
                elapsed = (time.monotonic() - started) * args.speed
                if elapsed >= duration:
                    if not args.loop:
                        break
                    elapsed %= motion.duration
                if player:
                    player.check()
                    if not audio_started and elapsed >= args.audio_delay:
                        player.start()
                        audio_started = True
                queued = client.send(motion.reference_type, motion.sample(elapsed, args.speed))
                if player and not queued:
                    raise RuntimeError("Motion transport disconnected during audio replay")
                time.sleep(.02)
            if player:
                player.finish()
    except KeyboardInterrupt:
        pass
    finally:
        if player:
            player.close()


if __name__ == "__main__":
    main()
