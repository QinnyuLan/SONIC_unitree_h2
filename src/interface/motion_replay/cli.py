"""Stream G1, H2 and SMPL motion references to the policy runtime."""
import argparse
from pathlib import Path
import time

import numpy as np
from interface.commands import RuntimeClient
from interface.paths import ROOT
from .motion import Motion, REFERENCE_TYPES


def main():
    parser = argparse.ArgumentParser(prog="python -m interface.motion_replay", description=__doc__)
    parser.add_argument("motion", type=Path, nargs="?", help="CSV clip directory or NPZ motion file")
    parser.add_argument("--keyboard", action="store_true", help="interactive collection browser and playback")
    parser.add_argument("--state", default="tcp://127.0.0.1:5559", help="runtime status for keyboard control")
    parser.add_argument("--motions-root", type=Path, default=ROOT / "motions")
    parser.add_argument("--type", choices=sorted(REFERENCE_TYPES), dest="reference_type",
                        help="type for unlabelled input; must agree with existing metadata")
    parser.add_argument("--fps", type=float, help="override source fps (CSV fallback: 50)")
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5558")
    parser.add_argument("--takeover", action="store_true")
    parser.add_argument("--loop", action="store_true", help="clip must have a continuous loop boundary")
    parser.add_argument("--inspect", action="store_true", help="validate and print metadata without connecting")
    args = parser.parse_args()
    if args.keyboard:
        if args.motion is not None or args.inspect or args.fps is not None or args.reference_type is not None:
            parser.error("--keyboard selects typed clips from collections; omit motion, --inspect, --fps and --type")
        if not np.isfinite(args.speed) or args.speed <= 0:
            parser.error("--speed must be positive")
        from .keyboard import run
        try:
            run(args)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        return
    if args.motion is None:
        parser.error("provide a motion path or use --keyboard")
    try:
        motion = Motion.load(args.motion, args.fps, args.reference_type)
        if not np.isfinite(args.speed) or args.speed <= 0:
            raise ValueError("--speed must be positive")
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    spacing = .02 if motion.reference_type == "smpl" else .1
    print(f"{motion.reference_type.upper()}: {len(motion.times)} frames, {motion.fps:g} Hz, "
          f"{motion.duration:.2f} s; 10 samples at {spacing:g} s intervals")
    if args.inspect:
        return
    try:
        with RuntimeClient("motion_replay", args.endpoint, takeover=args.takeover) as client:
            started = time.monotonic()
            while True:
                elapsed = (time.monotonic() - started) * args.speed
                if elapsed >= motion.duration:
                    if not args.loop:
                        break
                    elapsed %= motion.duration
                client.send(motion.reference_type, motion.sample(elapsed, args.speed))
                time.sleep(.02)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
