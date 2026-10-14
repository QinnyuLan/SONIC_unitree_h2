#!/usr/bin/env python3
"""End-to-end DDS-to-serial test using a pseudo Inspire hand."""

import argparse
import os
from pathlib import Path
import pty
import select
import signal
import struct
import subprocess
import tempfile
import threading
import time
import unittest


def checksum(frame: bytes | bytearray) -> int:
    return sum(frame[2:-1]) & 0xFF


def position_response(device_id: int, positions: tuple[int, ...]) -> bytes:
    frame = bytearray((0x90, 0xEB, device_id, 0x0F, 0x11, 0x0A, 0x06))
    frame.extend(struct.pack("<6H", *positions))
    frame.append(0)
    frame[-1] = checksum(frame)
    return bytes(frame)


def write_ack(device_id: int) -> bytes:
    frame = bytearray((0x90, 0xEB, device_id, 0x04, 0x12, 0xCE, 0x05, 0x01, 0))
    frame[-1] = checksum(frame)
    return bytes(frame)


class PseudoHand:
    def __init__(self, master_fd: int):
        self.master_fd = master_fd
        self.positions = (100, 200, 300, 400, 500, 600)
        self.set_commands: list[tuple[int, ...]] = []
        self.position_queries = 0
        self.errors: list[Exception] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def _run(self):
        pending = bytearray()
        try:
            while not self.stop_event.is_set():
                readable, _, _ = select.select([self.master_fd], [], [], 0.1)
                if not readable:
                    continue
                try:
                    chunk = os.read(self.master_fd, 1024)
                except OSError as error:
                    if self.stop_event.is_set() or error.errno == 5:
                        continue
                    raise
                pending.extend(chunk)

                while True:
                    header = pending.find(b"\xEB\x90")
                    if header < 0:
                        pending.clear()
                        break
                    if header:
                        del pending[:header]
                    if len(pending) < 4:
                        break
                    frame_length = pending[3] + 5
                    if len(pending) < frame_length:
                        break
                    frame = bytes(pending[:frame_length])
                    del pending[:frame_length]
                    self._handle_frame(frame)
        except Exception as error:  # pragma: no cover - reported by assertion
            self.errors.append(error)

    def _handle_frame(self, frame: bytes):
        if checksum(frame) != frame[-1]:
            raise AssertionError(f"bad request checksum: {frame.hex()}")
        device_id = frame[2]
        if frame[4] == 0x11 and frame[5:7] == b"\x0A\x06":
            self.position_queries += 1
            os.write(self.master_fd, position_response(device_id, self.positions))
            return
        if frame[4] == 0x12 and frame[5:7] == b"\xCE\x05":
            values = struct.unpack("<6H", frame[7:19])
            self.set_commands.append(values)
            self.positions = values
            os.write(self.master_fd, write_ack(device_id))
            return
        raise AssertionError(f"unexpected request: {frame.hex()}")


class Native485DdsTest(unittest.TestCase):
    service_binary = str(Path(__file__).resolve().parents[4] / "build/inspire_hands/bin/inspire_hand_service")
    example_binary = str(Path(__file__).resolve().parents[4] / "build/inspire_hands/bin/inspire_hand_client")
    network = "lo"

    @classmethod
    def setUpClass(cls):
        if not Path(cls.service_binary).is_file():
            raise unittest.SkipTest("Build tools/robot_services/inspire_hands before running these checks")

    def run_example(self, namespace: str, position: float, duration_ms: int):
        return subprocess.run(
            [
                self.example_binary,
                "--domain",
                "218",
                "--network",
                self.network,
                "--namespace",
                namespace,
                "--position",
                str(position),
                "--duration-ms",
                str(duration_ms),
                "--publish-hz",
                "20",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=duration_ms / 1000.0 + 4.0,
        )

    def test_dds_position_reaches_right_serial_port(self):
        master_fd, slave_fd = pty.openpty()
        slave_path = os.ttyname(slave_fd)
        namespace = f"inspire_native485_test_{os.getpid()}"
        hand = PseudoHand(master_fd)
        hand.start()
        service = subprocess.Popen(
            [
                self.service_binary,
                "--domain",
                "218",
                "--network",
                self.network,
                "--namespace",
                namespace,
                "--hands",
                "right",
                "--right-port",
                slave_path,
                "--probe-attempts",
                "1",
                "--serial-timeout-ms",
                "100",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        service_output = ""
        try:
            deadline = time.monotonic() + 5.0
            while "DDS ready:" not in service_output and time.monotonic() < deadline:
                readable, _, _ = select.select([service.stdout], [], [], 0.2)
                if readable:
                    service_output += service.stdout.readline()
                if service.poll() is not None:
                    break
            self.assertIn("DDS ready:", service_output)

            monitor = subprocess.run([
                self.example_binary, "--network", self.network, "--domain", "218",
                "--namespace", namespace, "--duration-ms", "600",
            ], capture_output=True, text=True, timeout=4)
            self.assertEqual(monitor.returncode, 0, monitor.stderr)
            self.assertIn("State right q=", monitor.stdout)
            self.assertEqual(hand.set_commands, [], "read-only client sent a position target")

            example = self.run_example(namespace, 0.73, 2500)
            self.assertEqual(example.returncode, 0, example.stderr)

            deadline = time.monotonic() + 2.0
            while not hand.set_commands and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(hand.set_commands, "no DDS command reached pseudo hand")
            self.assertEqual(hand.set_commands[-1], (730,) * 6)
        finally:
            if service.poll() is None:
                service.send_signal(signal.SIGTERM)
            try:
                remaining, _ = service.communicate(timeout=3.0)
                service_output += remaining
            except subprocess.TimeoutExpired:
                service.kill()
                service.communicate()
            hand.stop()
            os.close(slave_fd)
            os.close(master_fd)

        self.assertFalse(hand.thread.is_alive())
        self.assertEqual(hand.errors, [], service_output)

    def test_command_during_disconnect_is_not_replayed_after_usb_reconnect(self):
        namespace = f"inspire_reconnect_test_{os.getpid()}_{time.monotonic_ns()}"
        with tempfile.TemporaryDirectory() as temp_dir:
            port = os.path.join(temp_dir, "right-hand")
            service = subprocess.Popen(
                [
                    self.service_binary,
                    "--domain",
                    "218",
                    "--network",
                    self.network,
                    "--namespace",
                    namespace,
                    "--hands",
                    "right",
                    "--right-port",
                    port,
                    "--probe-attempts",
                    "1",
                    "--serial-timeout-ms",
                    "50",
                    "--command-timeout-ms",
                    "5000",
                    "--usb-retry-ms",
                    "100",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            service_output = ""
            master_fd = None
            slave_fd = None
            hand = None
            try:
                deadline = time.monotonic() + 5.0
                while "DDS ready:" not in service_output and time.monotonic() < deadline:
                    readable, _, _ = select.select([service.stdout], [], [], 0.2)
                    if readable:
                        service_output += service.stdout.readline()
                    if service.poll() is not None:
                        break
                self.assertIn("DDS ready:", service_output)

                disconnected_command = self.run_example(namespace, 0.22, 1500)
                self.assertEqual(
                    disconnected_command.returncode, 0, disconnected_command.stderr
                )
                self.assertIn("State right q=[", disconnected_command.stdout)

                master_fd, slave_fd = pty.openpty()
                hand = PseudoHand(master_fd)
                hand.start()
                os.symlink(os.ttyname(slave_fd), port)

                deadline = time.monotonic() + 2.0
                while hand.position_queries < 2 and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertGreaterEqual(
                    hand.position_queries, 2, "service did not recover the serial hand"
                )
                time.sleep(0.2)
                self.assertEqual(
                    hand.set_commands,
                    [],
                    "DDS command received while disconnected was replayed after reconnect",
                )

                fresh_command = self.run_example(namespace, 0.81, 1500)
                self.assertEqual(fresh_command.returncode, 0, fresh_command.stderr)
                deadline = time.monotonic() + 2.0
                while (810,) * 6 not in hand.set_commands and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertNotIn((220,) * 6, hand.set_commands)
                self.assertIn(
                    (810,) * 6,
                    hand.set_commands,
                    "fresh post-recovery DDS command did not reach the pseudo hand",
                )
            finally:
                if service.poll() is None:
                    service.send_signal(signal.SIGTERM)
                try:
                    remaining, _ = service.communicate(timeout=3.0)
                    service_output += remaining
                except subprocess.TimeoutExpired:
                    service.kill()
                    service.communicate()
                if hand is not None:
                    hand.stop()
                if slave_fd is not None:
                    os.close(slave_fd)
                if master_fd is not None:
                    os.close(master_fd)

        self.assertIsNotNone(hand)
        self.assertFalse(hand.thread.is_alive())
        self.assertEqual(hand.errors, [], service_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("service_binary")
    parser.add_argument("example_binary")
    parser.add_argument("--network", default="lo")
    args, remaining = parser.parse_known_args()
    Native485DdsTest.service_binary = os.path.abspath(args.service_binary)
    Native485DdsTest.example_binary = os.path.abspath(args.example_binary)
    Native485DdsTest.network = args.network
    unittest.main(argv=[__file__, *remaining])
