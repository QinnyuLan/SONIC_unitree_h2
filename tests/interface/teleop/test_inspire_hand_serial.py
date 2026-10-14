import os
import pty
import select
import struct
import threading
import time
import unittest

import numpy as np

from interface.peripherals.inspire_serial import (
    InspireSerialError,
    InspireTriggerController,
    PosixSerialPort,
    TriggerLatch,
    build_set_position_packet,
    decode_manager_trigger_state,
    validate_write_ack,
)
from interface.teleop.transport.packets import pack_pose_message


def _ack(device_id: int = 1) -> bytes:
    response = bytearray((0xEB, 0x90, device_id, 0x04, 0x12, 0xCE, 0x05, 0x01, 0x00))
    response[-1] = sum(response[2:-1]) & 0xFF
    return bytes(response)


class FakeHand:
    def __init__(self) -> None:
        self.positions = []
        self.closed = False

    def set_position(self, positions) -> bytes:
        self.positions.append(tuple(positions))
        return _ack()

    def close(self) -> None:
        self.closed = True


class InspirePacketTest(unittest.TestCase):
    def test_set_position_packet_matches_reference_layout(self):
        packet = build_set_position_packet(1, [0.0, 0.25, 0.5, 0.75, 1.0, 0.9999])
        self.assertEqual(len(packet), 20)
        self.assertEqual(packet[:7], bytes((0xEB, 0x90, 1, 0x0F, 0x12, 0xCE, 0x05)))
        self.assertEqual(struct.unpack("<6H", packet[7:19]), (0, 250, 500, 750, 1000, 999))
        self.assertEqual(packet[-1], sum(packet[2:-1]) & 0xFF)

    def test_rejects_invalid_position_instead_of_silently_clipping(self):
        with self.assertRaisesRegex(ValueError, r"position\[2\]"):
            build_set_position_packet(1, [1.0, 1.0, -1.0, 1.0, 1.0, 1.0])

    def test_ack_validation(self):
        validate_write_ack(_ack(), 1)
        bad = bytearray(_ack())
        bad[-1] ^= 0xFF
        with self.assertRaisesRegex(InspireSerialError, "checksum"):
            validate_write_ack(bytes(bad), 1)


class TriggerLatchTest(unittest.TestCase):
    def test_requires_release_then_uses_hysteresis(self):
        latch = TriggerLatch(open_threshold=0.4, close_threshold=0.6)
        self.assertIsNone(latch.update(0.9))
        self.assertEqual(latch.update(0.2), False)
        self.assertIsNone(latch.update(0.5))
        self.assertEqual(latch.update(0.7), True)
        self.assertIsNone(latch.update(0.5))
        self.assertEqual(latch.update(0.3), False)

    def test_controller_is_disabled_until_explicitly_enabled(self):
        left = FakeHand()
        right = FakeHand()
        controller = InspireTriggerController(
            {"left": left, "right": right}, max_close_ratio=0.6
        )
        self.assertEqual(controller.update(left_trigger=0.0, right_trigger=0.0), [])

        controller.set_enabled(True)
        events = controller.update(left_trigger=0.0, right_trigger=0.0)
        self.assertEqual([event.side for event in events], ["left", "right"])
        self.assertEqual(left.positions[-1], (1.0,) * 6)
        self.assertEqual(right.positions[-1], (1.0,) * 6)

        events = controller.update(left_trigger=0.8, right_trigger=0.2)
        self.assertEqual([event.side for event in events], ["left"])
        self.assertEqual(left.positions[-1], (0.4,) * 6)
        controller.close()
        self.assertTrue(left.closed)
        self.assertTrue(right.closed)

    def test_disabling_requires_release_to_arm_again(self):
        left = FakeHand()
        controller = InspireTriggerController({"left": left})
        controller.set_enabled(True)
        controller.update(left_trigger=0.0, right_trigger=0.0)
        controller.update(left_trigger=1.0, right_trigger=0.0)
        controller.set_enabled(False)
        controller.set_enabled(True)
        self.assertEqual(controller.update(left_trigger=1.0, right_trigger=0.0), [])


class ManagerStateDecodeTest(unittest.TestCase):
    def test_decodes_mode_independent_trigger_message(self):
        message = pack_pose_message(
            {
                "stream_mode": np.array([3], dtype=np.int32),
                "left_trigger": np.array([0.25], dtype=np.float32),
                "right_trigger": np.array([0.75], dtype=np.float32),
                "timestamp_monotonic": np.array([12.5], dtype=np.float64),
            },
            topic="manager_state",
        )
        state = decode_manager_trigger_state(message)
        self.assertEqual(state.stream_mode, 3)
        self.assertAlmostEqual(state.left_trigger, 0.25)
        self.assertAlmostEqual(state.right_trigger, 0.75)
        self.assertAlmostEqual(state.timestamp_monotonic, 12.5)

    def test_rejects_old_manager_message_without_triggers(self):
        message = pack_pose_message(
            {"stream_mode": np.array([1], dtype=np.int32)}, topic="manager_state"
        )
        with self.assertRaisesRegex(ValueError, "left_trigger"):
            decode_manager_trigger_state(message)


class PosixSerialPortTest(unittest.TestCase):
    def test_handles_fragmented_serial_ack(self):
        master_fd, slave_fd = pty.openpty()
        slave_path = os.ttyname(slave_fd)
        port = PosixSerialPort(slave_path, timeout_s=0.2)
        os.close(slave_fd)
        packet = build_set_position_packet(1, [1.0] * 6)
        received = bytearray()
        emulator_error = []

        def emulate_hand():
            try:
                deadline = time.monotonic() + 0.2
                while len(received) < len(packet):
                    remaining = deadline - time.monotonic()
                    readable, _, _ = select.select([master_fd], [], [], max(0.0, remaining))
                    if not readable:
                        raise TimeoutError("test hand did not receive the request")
                    received.extend(os.read(master_fd, len(packet) - len(received)))
                response = _ack()
                os.write(master_fd, response[:3])
                time.sleep(0.005)
                os.write(master_fd, response[3:])
            except Exception as exc:
                emulator_error.append(exc)

        thread = threading.Thread(target=emulate_hand)
        thread.start()
        try:
            response = port.transact(packet)
        finally:
            port.close()
            thread.join(timeout=1.0)
            os.close(master_fd)

        self.assertEqual(emulator_error, [])
        self.assertEqual(bytes(received), packet)
        self.assertEqual(response, _ack())


if __name__ == "__main__":
    unittest.main()
