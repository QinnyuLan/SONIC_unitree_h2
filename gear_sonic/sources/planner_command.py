"""Send a locomotion request to the independent G1 planner service."""
import argparse
import time
from gear_sonic.commands import RuntimeClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=int, default=0, help="SONIC locomotion mode: 0 idle, 1 slow walk, 2 walk")
    parser.add_argument("--speed", type=float, default=-1)
    parser.add_argument("--height", type=float, default=-1)
    parser.add_argument("--movement", nargs=3, type=float, default=[0, 0, 0])
    parser.add_argument("--facing", nargs=3, type=float, default=[1, 0, 0])
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--takeover", action="store_true")
    args = parser.parse_args()
    with RuntimeClient("planner_control", args.endpoint, ttl_ms=500, takeover=args.takeover) as client:
        try:
            while True:
                client.send("planner", {k: getattr(args, k) for k in ("mode", "speed", "height", "movement", "facing")})
                time.sleep(.05)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
