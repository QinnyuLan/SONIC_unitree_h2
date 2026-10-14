#!/usr/bin/env python3
"""Send one explicit position command to a serial Inspire RH56 hand."""

from __future__ import annotations

import argparse
import sys

from gear_sonic.utils.teleop.inspire_hand_serial import InspireSerialHand


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Serial device path")
    parser.add_argument("--device-id", type=int, default=1, help="Hand device ID (default: 1)")
    parser.add_argument(
        "--position",
        type=float,
        required=True,
        help="Uniform position for all six actuators: 0=closed, 1=open",
    )
    parser.add_argument(
        "--serial-timeout-ms",
        type=float,
        default=30.0,
        help="Write/ACK timeout (default: 30 ms)",
    )
    parser.add_argument(
        "--no-validate-ack",
        action="store_true",
        help="Require nine ACK bytes but skip ACK header/id/checksum validation",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.device_id <= 0xFE:
        parser.error("--device-id must be in [1, 254]")
    if not 0.0 <= args.position <= 1.0:
        parser.error("--position must be in [0, 1]")
    if args.serial_timeout_ms <= 0.0:
        parser.error("--serial-timeout-ms must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        with InspireSerialHand(
            args.port,
            device_id=args.device_id,
            timeout_s=args.serial_timeout_ms / 1000.0,
            validate_ack=not args.no_validate_ack,
        ) as hand:
            response = hand.set_position((args.position,) * 6)
    except Exception as exc:
        print(f"Inspire command failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Inspire command accepted: port={args.port} id={args.device_id} "
        f"q={args.position:.3f} ack={response.hex()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
