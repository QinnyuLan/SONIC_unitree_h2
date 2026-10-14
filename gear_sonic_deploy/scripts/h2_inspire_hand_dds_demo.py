#!/usr/bin/env python3
"""Inspect and command Inspire RH56 hands through the PC2 DDS service.

The command/state layout is 12 normalized positions: right hand first, then
left hand. Each hand is ordered as pinky, ring, middle, index, thumb bend,
and thumb rotation. Position 0 is closed and position 1 is open.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import sys
import threading
import time
from typing import Sequence

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

HAND_DOF = 6
TOTAL_DOF = 12
RIGHT_OFFSET = 0
LEFT_OFFSET = HAND_DOF
JOINT_ORDER = ("pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotate")


def normalized_position(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a number: {text}") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("position must be finite and in [0, 1]")
    return value


def joint_positions(text: str) -> tuple[float, ...]:
    fields = text.replace(",", " ").split()
    if len(fields) != HAND_DOF:
        raise argparse.ArgumentTypeError(
            f"--joints requires {HAND_DOF} values in this order: {', '.join(JOINT_ORDER)}"
        )
    try:
        values = tuple(float(field) for field in fields)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--joints contains a non-numeric value") from exc
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise argparse.ArgumentTypeError("every --joints value must be finite and in [0, 1]")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default="eno1", help="DDS interface (default: eno1)")
    parser.add_argument("--namespace", default="inspire", help="DDS namespace (default: inspire)")
    parser.add_argument(
        "--side",
        choices=("right", "left", "both"),
        default="both",
        help="hand receiving the explicit target (default: both)",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--position",
        type=normalized_position,
        help="uniform six-actuator target; 0=closed, 1=open",
    )
    target.add_argument(
        "--joints",
        type=joint_positions,
        metavar="Q0,Q1,Q2,Q3,Q4,Q5",
        help=f"six actuator targets in order: {', '.join(JOINT_ORDER)}",
    )
    parser.add_argument(
        "--allow-full-close",
        action="store_true",
        help="allow an explicit target below 0.5; required to reduce pinch risk",
    )
    parser.add_argument(
        "--state-timeout",
        type=float,
        default=3.0,
        help="seconds to wait for an online target hand (default: 3)",
    )
    parser.add_argument(
        "--state-max-age",
        type=float,
        default=0.5,
        help="maximum command-time state age in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--monitor-seconds",
        type=float,
        default=5.0,
        help="state-only runtime when no target is supplied; 0 runs until Ctrl-C",
    )
    parser.add_argument(
        "--publish-seconds",
        type=float,
        default=1.0,
        help="command publication duration (default: 1)",
    )
    parser.add_argument("--publish-hz", type=float, default=20.0, help="command rate (default: 20)")
    parser.add_argument(
        "--discovery-timeout",
        type=float,
        default=3.0,
        help="seconds to wait for the PC2 command subscriber (default: 3)",
    )
    parser.add_argument(
        "--post-monitor-seconds",
        type=float,
        default=2.0,
        help="state observation time after a command (default: 2)",
    )
    args = parser.parse_args(argv)

    if not args.namespace:
        parser.error("--namespace must not be empty")
    for name in (
        "state_timeout",
        "state_max_age",
        "publish_seconds",
        "publish_hz",
        "discovery_timeout",
        "post_monitor_seconds",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.monitor_seconds) or args.monitor_seconds < 0.0:
        parser.error("--monitor-seconds must be non-negative")

    pattern = requested_pattern(args)
    if pattern is not None and min(pattern) < 0.5 and not args.allow_full_close:
        parser.error("targets below 0.5 require --allow-full-close")
    return args


def requested_pattern(args: argparse.Namespace) -> tuple[float, ...] | None:
    if args.position is not None:
        return (args.position,) * HAND_DOF
    if args.joints is not None:
        return tuple(args.joints)
    return None


def selected_offsets(side: str) -> tuple[int, ...]:
    if side == "right":
        return (RIGHT_OFFSET,)
    if side == "left":
        return (LEFT_OFFSET,)
    if side == "both":
        return (RIGHT_OFFSET, LEFT_OFFSET)
    raise ValueError(f"invalid side: {side}")


@dataclass(frozen=True)
class StateSnapshot:
    q: tuple[float, ...]
    lost: tuple[int, ...]
    received_at: float
    sequence: int


def side_online(snapshot: StateSnapshot, offset: int) -> bool:
    values = snapshot.q[offset : offset + HAND_DOF]
    lost = snapshot.lost[offset : offset + HAND_DOF]
    return (
        len(values) == HAND_DOF
        and all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
        and all(value == 0 for value in lost)
    )


def build_target(
    snapshot: StateSnapshot,
    side: str,
    pattern: Sequence[float],
) -> tuple[float, ...]:
    if len(snapshot.q) != TOTAL_DOF:
        raise ValueError(f"state must contain {TOTAL_DOF} positions")
    if len(pattern) != HAND_DOF:
        raise ValueError(f"target pattern must contain {HAND_DOF} positions")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in pattern):
        raise ValueError("target pattern must be finite and in [0, 1]")

    # An offline unselected side is ignored by a single-hand PC2 service. Use an
    # open placeholder so the mandatory 12-value DDS message remains valid.
    target = [value if math.isfinite(value) and 0.0 <= value <= 1.0 else 1.0 for value in snapshot.q]
    for offset in selected_offsets(side):
        target[offset : offset + HAND_DOF] = pattern
    return tuple(target)


def make_command(target: Sequence[float]) -> MotorCmds_:
    if len(target) != TOTAL_DOF:
        raise ValueError(f"command must contain {TOTAL_DOF} positions")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in target):
        raise ValueError("command positions must be finite and in [0, 1]")
    command = MotorCmds_([unitree_go_msg_dds__MotorCmd_() for _ in range(TOTAL_DOF)])
    for motor, value in zip(command.cmds, target, strict=True):
        motor.q = float(value)
    return command


def format_side(snapshot: StateSnapshot, offset: int) -> str:
    values = ", ".join(f"{value:.3f}" for value in snapshot.q[offset : offset + HAND_DOF])
    lost = max(snapshot.lost[offset : offset + HAND_DOF], default=1)
    status = "ONLINE" if side_online(snapshot, offset) else "OFFLINE"
    return f"{status} q=[{values}] lost={lost}"


def format_snapshot(snapshot: StateSnapshot) -> str:
    age_ms = max(0.0, time.monotonic() - snapshot.received_at) * 1000.0
    return (
        f"right {format_side(snapshot, RIGHT_OFFSET)} | "
        f"left {format_side(snapshot, LEFT_OFFSET)} | age={age_ms:.0f} ms"
    )


class InspireDdsClient:
    def __init__(self, network: str, namespace: str, enable_commands: bool) -> None:
        ChannelFactoryInitialize(0, network)
        self._condition = threading.Condition()
        self._snapshot: StateSnapshot | None = None
        self._sequence = 0
        self._state_topic = f"rt/{namespace}/state"
        self._command_topic = f"rt/{namespace}/cmd"
        self._subscriber = ChannelSubscriber(self._state_topic, MotorStates_)
        self._subscriber.Init(self._on_state, 1)
        self._publisher: ChannelPublisher | None = None
        if enable_commands:
            self._publisher = ChannelPublisher(self._command_topic, MotorCmds_)
            self._publisher.Init()

    def _on_state(self, message: MotorStates_) -> None:
        if len(message.states) != TOTAL_DOF:
            return
        with self._condition:
            self._sequence += 1
            self._snapshot = StateSnapshot(
                q=tuple(float(state.q) for state in message.states),
                lost=tuple(int(state.lost) for state in message.states),
                received_at=time.monotonic(),
                sequence=self._sequence,
            )
            self._condition.notify_all()

    def wait_for_state(self, timeout: float, after_sequence: int = -1) -> StateSnapshot | None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._snapshot is None or self._snapshot.sequence <= after_sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
            return self._snapshot

    def wait_for_online(
        self,
        side: str,
        timeout: float,
        max_age: float,
    ) -> StateSnapshot | None:
        deadline = time.monotonic() + timeout
        offsets = selected_offsets(side)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None
            snapshot = self.wait_for_state(remaining)
            if snapshot is None:
                return None
            if time.monotonic() - snapshot.received_at <= max_age and all(
                side_online(snapshot, offset) for offset in offsets
            ):
                return snapshot
            time.sleep(min(0.02, remaining))

    def publish(self, command: MotorCmds_, timeout: float | None = None) -> bool:
        if self._publisher is None:
            raise RuntimeError("command publisher is disabled")
        return bool(self._publisher.Write(command, timeout))

    def collect_latest(self, duration: float, after_sequence: int) -> StateSnapshot | None:
        deadline = time.monotonic() + duration
        latest: StateSnapshot | None = None
        sequence = after_sequence
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return latest
            snapshot = self.wait_for_state(remaining, after_sequence=sequence)
            if snapshot is None:
                return latest
            latest = snapshot
            sequence = snapshot.sequence

    def close(self) -> None:
        if self._publisher is not None:
            self._publisher.Close()
        self._subscriber.Close()


def monitor(client: InspireDdsClient, duration: float, print_hz: float = 2.0) -> int:
    deadline = None if duration == 0.0 else time.monotonic() + duration
    sequence = -1
    received_any = False
    wait_period = 1.0 / print_hz
    while deadline is None or time.monotonic() < deadline:
        timeout = wait_period
        if deadline is not None:
            timeout = min(timeout, max(0.0, deadline - time.monotonic()))
        if timeout <= 0.0:
            break
        snapshot = client.wait_for_state(timeout, after_sequence=sequence)
        if snapshot is None:
            continue
        received_any = True
        sequence = snapshot.sequence
        print(format_snapshot(snapshot))
        time.sleep(wait_period)
    return 0 if received_any else 2


def run_command(client: InspireDdsClient, args: argparse.Namespace) -> int:
    pattern = requested_pattern(args)
    assert pattern is not None
    snapshot = client.wait_for_online(args.side, args.state_timeout, args.state_max_age)
    if snapshot is None:
        print(
            f"Refusing to send: no fresh ONLINE state for {args.side} hand(s) on rt/{args.namespace}/state",
            file=sys.stderr,
        )
        return 2

    target = build_target(snapshot, args.side, pattern)
    command = make_command(target)
    print("Current:", format_snapshot(snapshot))
    print(
        f"Publishing {args.side} target {tuple(round(value, 3) for value in pattern)} to rt/{args.namespace}/cmd"
    )
    if not client.publish(command, timeout=args.discovery_timeout):
        print("Refusing to continue: no PC2 DDS command subscriber matched", file=sys.stderr)
        return 3

    period = 1.0 / args.publish_hz
    deadline = time.monotonic() + args.publish_seconds
    while time.monotonic() < deadline:
        client.publish(command)
        time.sleep(period)
    print("Command publication stopped; the hand service retains the last target.")

    final = client.collect_latest(args.post_monitor_seconds, after_sequence=snapshot.sequence)
    if final is not None:
        print("Final:  ", format_snapshot(final))
        errors = []
        for offset in selected_offsets(args.side):
            errors.extend(abs(final.q[offset + index] - pattern[index]) for index in range(HAND_DOF))
        print(f"Selected-side max target error: {max(errors):.3f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    pattern = requested_pattern(args)
    command_mode = pattern is not None
    print(
        f"DDS domain=0 interface={args.network} topics=rt/{args.namespace}/cmd,state\n"
        f"Joint order: {', '.join(JOINT_ORDER)}; 0=closed, 1=open"
    )
    if not command_mode:
        print("Monitor-only mode: no hand command will be published.")

    client: InspireDdsClient | None = None
    try:
        client = InspireDdsClient(args.network, args.namespace, enable_commands=command_mode)
        if command_mode:
            return run_command(client, args)
        result = monitor(client, args.monitor_seconds)
        if result != 0:
            print(f"No 12-value state received on rt/{args.namespace}/state", file=sys.stderr)
        return result
    except KeyboardInterrupt:
        print("\nStopped; no shutdown hand command was sent.")
        return 130
    except Exception as exc:
        print(f"Inspire DDS demo failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    sys.exit(main())
