"""Send independent head or hand targets; no body policy input is changed."""
import argparse
import json
import math
import time
from gear_sonic.commands import RuntimeClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["head", "hands"])
    parser.add_argument("--pitch", type=float, default=0, help="radians")
    parser.add_argument("--yaw", type=float, default=0, help="radians")
    parser.add_argument("--left", type=json.loads, help="JSON array of seven Dex3-compatible targets")
    parser.add_argument("--right", type=json.loads)
    parser.add_argument("--endpoint", help="default head :5562; hands :5563")
    parser.add_argument("--seconds", type=float, default=0, help="0: until interrupted")
    parser.add_argument("--takeover", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error("invalid duration")
    data = {"pitch": args.pitch, "yaw": args.yaw} if args.kind == "head" else {
        k: v for k, v in {"left": args.left, "right": args.right}.items() if v is not None}
    endpoint = args.endpoint or f"tcp://127.0.0.1:{5562 if args.kind == 'head' else 5563}"
    with RuntimeClient(args.kind, endpoint, takeover=args.takeover) as client:
        started = time.monotonic()
        try:
            while args.seconds == 0 or time.monotonic() - started < args.seconds:
                client.send(args.kind, data)
                time.sleep(.02)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
