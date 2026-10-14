"""Direct serial control primitives for Inspire RH56 dexterous hands.

The packet layout follows the local ``dfx_inspire_service`` implementation.  A
hand has six normalized position commands in this order: pinky, ring, middle,
index, thumb bend, and thumb rotation.  Zero is closed and one is open.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import select
import struct
import termios
import threading
import time
from typing import Protocol, Sequence

INSPIRE_HAND_DOF = 6
MANAGER_STATE_TOPIC = b"manager_state"
PACKED_HEADER_SIZE = 1280
OPEN_POSITION = (1.0,) * INSPIRE_HAND_DOF


class InspireSerialError(RuntimeError):
    """Raised when a serial hand transaction cannot be completed safely."""


def _checksum(data: bytes | bytearray) -> int:
    return sum(data[2:-1]) & 0xFF


def build_set_position_packet(device_id: int, positions: Sequence[float]) -> bytes:
    """Build the RH56 20-byte position command used by ``SetPosition``."""
    if not 1 <= int(device_id) <= 0xFE:
        raise ValueError(f"device_id must be in [1, 254], got {device_id}")
    if len(positions) != INSPIRE_HAND_DOF:
        raise ValueError(
            f"Inspire position must contain {INSPIRE_HAND_DOF} values, got {len(positions)}"
        )

    encoded: list[int] = []
    for index, raw_value in enumerate(positions):
        value = float(raw_value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"position[{index}] must be finite and in [0, 1], got {raw_value}")
        # Match Eigen's cast<int16_t> in dfx_inspire_service (truncate toward zero).
        encoded.append(int(value * 1000.0))

    packet = bytearray((0xEB, 0x90, int(device_id), 0x0F, 0x12, 0xCE, 0x05))
    packet.extend(struct.pack("<6H", *encoded))
    packet.append(0)
    packet[-1] = _checksum(packet)
    return bytes(packet)


def validate_write_ack(response: bytes, device_id: int) -> None:
    """Validate the nine-byte response returned after a position write."""
    if len(response) != 9:
        raise InspireSerialError(f"expected 9-byte Inspire ACK, received {len(response)} bytes")
    if response[:2] != b"\xEB\x90":
        raise InspireSerialError(f"invalid Inspire ACK header: {response[:2].hex()}")
    if response[2] != device_id:
        raise InspireSerialError(
            f"Inspire ACK device id {response[2]} does not match command id {device_id}"
        )
    if response[-1] != _checksum(response):
        raise InspireSerialError("invalid Inspire ACK checksum")


class PosixSerialPort:
    """Small 115200-8N1 serial transport with complete read/write handling."""

    def __init__(self, path: str, timeout_s: float = 0.03) -> None:
        if not path:
            raise ValueError("serial path must not be empty")
        if timeout_s <= 0.0:
            raise ValueError("serial timeout must be positive")
        self.path = path
        self.timeout_s = float(timeout_s)
        self._fd = -1
        self._lock = threading.Lock()

        try:
            self._fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            self._configure()
        except PermissionError as exc:
            self.close()
            raise InspireSerialError(
                f"permission denied opening {path}; grant access to the serial device "
                "or add the user to the dialout group"
            ) from exc
        except OSError as exc:
            self.close()
            raise InspireSerialError(f"failed to open Inspire serial port {path}: {exc}") from exc

    def _configure(self) -> None:
        attrs = termios.tcgetattr(self._fd)
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = termios.B115200 | termios.CS8 | termios.CLOCAL | termios.CREAD
        attrs[3] = 0
        attrs[4] = termios.B115200
        attrs[5] = termios.B115200
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(self._fd, termios.TCSANOW, attrs)
        termios.tcflush(self._fd, termios.TCIOFLUSH)

    def _write_all(self, data: bytes, deadline: float) -> None:
        view = memoryview(data)
        written = 0
        while written < len(view):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise InspireSerialError(f"timed out writing to {self.path}")
            _, writable, _ = select.select([], [self._fd], [], remaining)
            if not writable:
                raise InspireSerialError(f"timed out writing to {self.path}")
            try:
                count = os.write(self._fd, view[written:])
            except BlockingIOError:
                continue
            if count <= 0:
                raise InspireSerialError(f"serial write returned {count} for {self.path}")
            written += count

    def _read_exact(self, size: int, deadline: float) -> bytes:
        response = bytearray()
        while len(response) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            readable, _, _ = select.select([self._fd], [], [], remaining)
            if not readable:
                break
            try:
                chunk = os.read(self._fd, size - len(response))
            except BlockingIOError:
                continue
            if not chunk:
                continue
            response.extend(chunk)
        if len(response) != size:
            raise InspireSerialError(
                f"timed out reading Inspire ACK from {self.path}: {len(response)}/{size} bytes"
            )
        return bytes(response)

    def transact(self, request: bytes, response_size: int = 9) -> bytes:
        """Write one complete command and read its complete response."""
        with self._lock:
            if self._fd < 0:
                raise InspireSerialError(f"serial port {self.path} is closed")
            try:
                termios.tcflush(self._fd, termios.TCIFLUSH)
                deadline = time.monotonic() + self.timeout_s
                self._write_all(request, deadline)
                termios.tcdrain(self._fd)
                return self._read_exact(response_size, deadline)
            except OSError as exc:
                raise InspireSerialError(f"serial transaction failed on {self.path}: {exc}") from exc

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> "PosixSerialPort":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class SerialTransaction(Protocol):
    def transact(self, request: bytes, response_size: int = 9) -> bytes: ...

    def close(self) -> None: ...


class HandWriter(Protocol):
    def set_position(self, positions: Sequence[float]) -> bytes: ...

    def close(self) -> None: ...


class InspireSerialHand:
    """One RH56 hand connected through one serial port (G1-style wiring)."""

    def __init__(
        self,
        path: str,
        *,
        device_id: int = 1,
        timeout_s: float = 0.03,
        validate_ack: bool = True,
        transport: SerialTransaction | None = None,
    ) -> None:
        if not 1 <= int(device_id) <= 0xFE:
            raise ValueError(f"device_id must be in [1, 254], got {device_id}")
        self.path = path
        self.device_id = int(device_id)
        self.validate_ack = bool(validate_ack)
        self._transport = transport or PosixSerialPort(path, timeout_s=timeout_s)

    def set_position(self, positions: Sequence[float]) -> bytes:
        packet = build_set_position_packet(self.device_id, positions)
        response = self._transport.transact(packet, response_size=9)
        if self.validate_ack:
            validate_write_ack(response, self.device_id)
        return response

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> "InspireSerialHand":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


@dataclass(frozen=True)
class ManagerTriggerState:
    stream_mode: int
    left_trigger: float
    right_trigger: float
    timestamp_monotonic: float | None = None


def decode_manager_trigger_state(message: bytes) -> ManagerTriggerState:
    """Decode Trigger values from a packed ``manager_state`` ZMQ message."""
    if not message.startswith(MANAGER_STATE_TOPIC):
        raise ValueError("message does not start with the manager_state topic")
    header_start = len(MANAGER_STATE_TOPIC)
    payload_start = header_start + PACKED_HEADER_SIZE
    if len(message) < payload_start:
        raise ValueError("manager_state message is shorter than its packed header")

    raw_header = message[header_start:payload_start].split(b"\0", 1)[0]
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("manager_state contains an invalid JSON header") from exc

    endian = header.get("endian", "le")
    if endian not in {"le", "be"}:
        raise ValueError(f"unsupported packed-message endianness: {endian}")
    endian_prefix = "<" if endian == "le" else ">"
    formats = {
        "bool": ("?", 1),
        "u8": ("B", 1),
        "i32": ("i", 4),
        "i64": ("q", 8),
        "f32": ("f", 4),
        "f64": ("d", 8),
    }

    wanted = {"stream_mode", "left_trigger", "right_trigger", "timestamp_monotonic"}
    values: dict[str, int | float | bool] = {}
    offset = payload_start
    for field in header.get("fields", []):
        dtype = field.get("dtype")
        if dtype not in formats:
            raise ValueError(f"unsupported packed-message dtype: {dtype}")
        shape = field.get("shape")
        if not isinstance(shape, list) or not shape or any(int(dim) < 0 for dim in shape):
            raise ValueError(f"invalid shape for manager_state field {field.get('name')}")
        count = math.prod(int(dim) for dim in shape)
        format_char, item_size = formats[dtype]
        field_size = count * item_size
        if offset + field_size > len(message):
            raise ValueError(f"truncated manager_state field {field.get('name')}")
        name = field.get("name")
        if name in wanted:
            if count != 1:
                raise ValueError(f"manager_state field {name} must be scalar")
            values[name] = struct.unpack_from(endian_prefix + format_char, message, offset)[0]
        offset += field_size

    missing = {"stream_mode", "left_trigger", "right_trigger"} - values.keys()
    if missing:
        raise ValueError(f"manager_state is missing fields: {', '.join(sorted(missing))}")
    left_trigger = float(values["left_trigger"])
    right_trigger = float(values["right_trigger"])
    if not math.isfinite(left_trigger) or not math.isfinite(right_trigger):
        raise ValueError("manager_state Trigger values must be finite")
    return ManagerTriggerState(
        stream_mode=int(values["stream_mode"]),
        left_trigger=left_trigger,
        right_trigger=right_trigger,
        timestamp_monotonic=(
            float(values["timestamp_monotonic"])
            if "timestamp_monotonic" in values
            else None
        ),
    )


class TriggerLatch:
    """Binary Trigger state with release-to-arm and open/close hysteresis."""

    def __init__(self, *, open_threshold: float = 0.4, close_threshold: float = 0.6) -> None:
        if not 0.0 <= open_threshold < close_threshold <= 1.0:
            raise ValueError("Trigger thresholds must satisfy 0 <= open < close <= 1")
        self.open_threshold = float(open_threshold)
        self.close_threshold = float(close_threshold)
        self.armed = False
        self.closed = False

    def reset(self) -> None:
        self.armed = False
        self.closed = False

    def update(self, value: float) -> bool | None:
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Trigger value must be finite and in [0, 1], got {value}")
        if not self.armed:
            if value <= self.open_threshold:
                self.armed = True
                self.closed = False
                return False
            return None
        if self.closed and value <= self.open_threshold:
            self.closed = False
            return False
        if not self.closed and value >= self.close_threshold:
            self.closed = True
            return True
        return None


@dataclass(frozen=True)
class HandCommandEvent:
    side: str
    closed: bool
    trigger: float
    positions: tuple[float, ...]


class InspireTriggerController:
    """Apply left/right Trigger edges to independent G1-wired serial hands."""

    def __init__(
        self,
        hands: dict[str, HandWriter],
        *,
        open_threshold: float = 0.4,
        close_threshold: float = 0.6,
        max_close_ratio: float = 1.0,
    ) -> None:
        invalid_sides = set(hands) - {"left", "right"}
        if invalid_sides:
            raise ValueError(f"unsupported hand sides: {sorted(invalid_sides)}")
        if not hands:
            raise ValueError("at least one Inspire hand must be configured")
        if not 0.0 <= max_close_ratio <= 1.0:
            raise ValueError("max_close_ratio must be in [0, 1]")
        self.hands = dict(hands)
        self.max_close_ratio = float(max_close_ratio)
        self._latches = {
            side: TriggerLatch(
                open_threshold=open_threshold,
                close_threshold=close_threshold,
            )
            for side in hands
        }
        self._enabled = False

    @property
    def configured_sides(self) -> tuple[str, ...]:
        return tuple(side for side in ("left", "right") if side in self.hands)

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._enabled and not enabled:
            for latch in self._latches.values():
                latch.reset()
        self._enabled = enabled

    def update(self, *, left_trigger: float, right_trigger: float) -> list[HandCommandEvent]:
        if not self._enabled:
            return []
        trigger_by_side = {"left": left_trigger, "right": right_trigger}
        events: list[HandCommandEvent] = []
        for side in self.configured_sides:
            trigger = float(trigger_by_side[side])
            closed = self._latches[side].update(trigger)
            if closed is None:
                continue
            positions = (
                (1.0 - self.max_close_ratio,) * INSPIRE_HAND_DOF
                if closed
                else OPEN_POSITION
            )
            self.hands[side].set_position(positions)
            events.append(
                HandCommandEvent(
                    side=side,
                    closed=closed,
                    trigger=trigger,
                    positions=positions,
                )
            )
        return events

    def close(self) -> None:
        for hand in self.hands.values():
            hand.close()

    def __enter__(self) -> "InspireTriggerController":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
