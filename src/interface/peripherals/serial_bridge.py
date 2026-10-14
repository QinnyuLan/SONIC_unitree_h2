#!/usr/bin/env python3
"""Drive locally attached Inspire RH56 hands from PICO Trigger messages.

This process is deliberately separate from the 500 Hz H2 deployment process:
serial writes wait for a device ACK and must never block body LowCmd publishing.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

import zmq

from interface.peripherals.inspire_serial import (
    MANAGER_STATE_TOPIC,
    InspireSerialHand,
    InspireTriggerController,
    decode_manager_trigger_state,
)

LOGGER = logging.getLogger("h2_inspire_hand_serial_bridge")


class DryRunHand:
    def __init__(self, side: str, path: str, device_id: int) -> None:
        self.side = side
        self.path = path
        self.device_id = device_id

    def set_position(self, positions) -> bytes:
        LOGGER.info(
            "DRY-RUN %s port=%s id=%d q=%s",
            self.side,
            self.path,
            self.device_id,
            ",".join(f"{value:.3f}" for value in positions),
        )
        return b""

    def close(self) -> None:
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bridge PICO manager_state Trigger values to locally attached Inspire RH56 "
            "hands using G1-style independent serial ports"
        )
    )
    parser.add_argument("--left-port", default="", help="Left-hand serial path (device ID 1)")
    parser.add_argument("--right-port", default="", help="Right-hand serial path (device ID 1)")
    parser.add_argument("--left-id", type=int, default=1, help="Left-hand device ID (default: 1)")
    parser.add_argument("--right-id", type=int, default=1, help="Right-hand device ID (default: 1)")
    parser.add_argument("--zmq-host", default="localhost", help="PICO manager host")
    parser.add_argument("--zmq-port", type=int, default=5556, help="PICO manager port")
    parser.add_argument(
        "--open-threshold",
        type=float,
        default=0.4,
        help="Trigger value at or below which the hand opens/arms (default: 0.4)",
    )
    parser.add_argument(
        "--close-threshold",
        type=float,
        default=0.6,
        help="Trigger value at or above which the hand closes (default: 0.6)",
    )
    parser.add_argument(
        "--max-close-ratio",
        type=float,
        default=1.0,
        help="Maximum closure in [0,1]; 1 is fully closed (default: 1)",
    )
    parser.add_argument(
        "--serial-timeout-ms",
        type=float,
        default=30.0,
        help="Timeout for one serial write/ACK transaction (default: 30 ms)",
    )
    parser.add_argument(
        "--message-timeout",
        type=float,
        default=0.5,
        help="Stop issuing commands after manager_state is stale (default: 0.5 s)",
    )
    parser.add_argument(
        "--control-while-off",
        action="store_true",
        help="Allow Trigger control while the PICO manager stream mode is OFF (bench use only)",
    )
    parser.add_argument(
        "--no-validate-ack",
        action="store_true",
        help="Require nine ACK bytes but skip ACK header/id/checksum validation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Decode Trigger messages and log commands without opening serial ports",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    if not args.left_port and not args.right_port:
        parser.error("at least one of --left-port or --right-port is required")
    if args.left_port and args.right_port and not args.dry_run:
        if os.path.realpath(args.left_port) == os.path.realpath(args.right_port):
            parser.error("G1 wiring requires different serial ports for left and right hands")
    if not 1 <= args.left_id <= 0xFE or not 1 <= args.right_id <= 0xFE:
        parser.error("--left-id and --right-id must be in [1, 254]")
    if not 0.0 <= args.open_threshold < args.close_threshold <= 1.0:
        parser.error("thresholds must satisfy 0 <= open-threshold < close-threshold <= 1")
    if not 0.0 <= args.max_close_ratio <= 1.0:
        parser.error("--max-close-ratio must be in [0, 1]")
    if args.serial_timeout_ms <= 0.0:
        parser.error("--serial-timeout-ms must be positive")
    if args.message_timeout <= 0.0:
        parser.error("--message-timeout must be positive")
    if not 1 <= args.zmq_port <= 65535:
        parser.error("--zmq-port must be in [1, 65535]")
    return args


def create_controller(args: argparse.Namespace) -> InspireTriggerController:
    hands = {}
    try:
        for side, path, device_id in (
            ("left", args.left_port, args.left_id),
            ("right", args.right_port, args.right_id),
        ):
            if not path:
                continue
            if args.dry_run:
                hands[side] = DryRunHand(side, path, device_id)
            else:
                hands[side] = InspireSerialHand(
                    path,
                    device_id=device_id,
                    timeout_s=args.serial_timeout_ms / 1000.0,
                    validate_ack=not args.no_validate_ack,
                )
    except Exception:
        for hand in hands.values():
            hand.close()
        raise

    return InspireTriggerController(
        hands,
        open_threshold=args.open_threshold,
        close_threshold=args.close_threshold,
        max_close_ratio=args.max_close_ratio,
    )


def run(args: argparse.Namespace) -> int:
    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    controller = create_controller(args)
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.SUBSCRIBE, MANAGER_STATE_TOPIC)
    endpoint = f"tcp://{args.zmq_host}:{args.zmq_port}"
    socket.connect(endpoint)

    LOGGER.info(
        "listening on %s for %s hand(s); G1 wiring uses device ID 1 per serial port",
        endpoint,
        ", ".join(controller.configured_sides),
    )
    LOGGER.info(
        "Trigger mapping: release<=%.2f opens/arms, press>=%.2f closes, max_close_ratio=%.2f",
        args.open_threshold,
        args.close_threshold,
        args.max_close_ratio,
    )

    last_message_at: float | None = None
    input_enabled = False
    last_decode_warning_at = 0.0
    try:
        while not stop_requested:
            if socket.poll(timeout=50):
                raw = socket.recv()
                try:
                    state = decode_manager_trigger_state(raw)
                except ValueError as exc:
                    now = time.monotonic()
                    if now - last_decode_warning_at >= 2.0:
                        LOGGER.warning("ignored invalid manager_state: %s", exc)
                        last_decode_warning_at = now
                    continue

                last_message_at = time.monotonic()
                should_enable = args.control_while_off or state.stream_mode != 0
                if should_enable != input_enabled:
                    controller.set_enabled(should_enable)
                    input_enabled = should_enable
                    if should_enable:
                        LOGGER.info("Trigger input enabled; release each Trigger once to arm its hand")
                    else:
                        LOGGER.info("Trigger input disabled in OFF mode; hands retain their last target")

                for event in controller.update(
                    left_trigger=state.left_trigger,
                    right_trigger=state.right_trigger,
                ):
                    LOGGER.info(
                        "%s hand -> %s (Trigger=%.3f, q=%.3f)",
                        event.side,
                        "CLOSED" if event.closed else "OPEN",
                        event.trigger,
                        event.positions[0],
                    )
            elif (
                input_enabled
                and last_message_at is not None
                and time.monotonic() - last_message_at >= args.message_timeout
            ):
                controller.set_enabled(False)
                input_enabled = False
                LOGGER.warning(
                    "manager_state timed out; serial commands stopped and hands retain last targets"
                )
    finally:
        controller.close()
        socket.close()
        context.term()
        LOGGER.info("bridge stopped; no shutdown pose was sent")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        return run(args)
    except Exception as exc:
        LOGGER.error("bridge failed: %s", exc)
        if args.verbose:
            LOGGER.exception("failure details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
