"""Stream motion references and optional speech to the policy runtime and audio player."""
import argparse
from pathlib import Path
import time

import numpy as np
from interface.commands import RuntimeClient
from .motion import Motion, REFERENCE_TYPES
from .playback import AudioPlayer, ReplayBundle, wav_duration


def main():
    parser = argparse.ArgumentParser(prog="python -m interface.motion_replay", description=__doc__)
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
