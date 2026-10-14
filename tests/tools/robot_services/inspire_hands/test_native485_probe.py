#!/usr/bin/env python3
"""Pseudo-terminal tests for H2 native RS485 probing and daemon recovery."""

import argparse
import os
from pathlib import Path
import pty
import select
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


def take_position_request(buffered: bytearray):
    while True:
        start = buffered.find(b"\xEB\x90")
        if start < 0:
            if buffered[-1:] == b"\xEB":
                del buffered[:-1]
            else:
                buffered.clear()
            return None
        if start > 0:
            del buffered[:start]
        if len(buffered) < 9:
            return None
        frame = bytes(buffered[:9])
        del buffered[:9]
        if frame[4] == 0x11 and frame[5:7] == b"\x0A\x06":
            return frame


class Native485ProbeTest(unittest.TestCase):
    binary = str(Path(__file__).resolve().parents[4] / "build/inspire_hands/bin/inspire_hand_service")

    @classmethod
    def setUpClass(cls):
        if not Path(cls.binary).is_file():
            raise unittest.SkipTest("Build tools/robot_services/inspire_hands before running these checks")

    def daemon_command(self, port: str):
        return [
            self.binary,
            "--domain",
            "218",
            "--network",
            "lo",
            "--namespace",
            f"inspire_probe_test_{os.getpid()}_{time.monotonic_ns()}",
            "--hands",
            "right",
            "--right-port",
            port,
            "--probe-attempts",
            "1",
            "--serial-timeout-ms",
            "10",
            "--worker-period-ms",
            "20",
            "--offline-poll-ms",
            "500",
            "--usb-retry-ms",
            "100",
        ]

    def start_silent_daemon(self):
        master_fd, slave_fd = pty.openpty()
        slave_path = os.ttyname(slave_fd)
        service = subprocess.Popen(
            self.daemon_command(slave_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return service, master_fd, slave_fd

    @staticmethod
    def collect_position_request_times(master_fd: int, duration: float):
        deadline = time.monotonic() + duration
        request_times = []
        buffered = bytearray()
        while time.monotonic() < deadline:
            readable, _, _ = select.select(
                [master_fd], [], [], max(0.0, deadline - time.monotonic())
            )
            if not readable:
                continue
            buffered.extend(os.read(master_fd, 4096))
            while True:
                if take_position_request(buffered) is None:
                    break
                request_times.append(time.monotonic())
        return request_times

    def run_probe(self, respond: bool):
        master_fd, slave_fd = pty.openpty()
        slave_path = os.ttyname(slave_fd)
        request = bytearray()
        emulator_errors = []

        def emulate_hand():
            try:
                deadline = time.monotonic() + 1.0
                while len(request) < 9:
                    timeout = max(0.0, deadline - time.monotonic())
                    readable, _, _ = select.select([master_fd], [], [], timeout)
                    if not readable:
                        raise TimeoutError("probe request was not received")
                    request.extend(os.read(master_fd, 9 - len(request)))
                if respond:
                    response = position_response(1, (100, 200, 300, 400, 500, 600))
                    os.write(master_fd, response[:4])
                    time.sleep(0.005)
                    os.write(master_fd, response[4:])
            except Exception as error:  # pragma: no cover - reported by assertion
                emulator_errors.append(error)

        thread = threading.Thread(target=emulate_hand)
        thread.start()
        try:
            result = subprocess.run(
                [
                    self.binary,
                    "--probe-only",
                    "--hands",
                    "right",
                    "--right-port",
                    slave_path,
                    "--probe-attempts",
                    "1",
                    "--serial-timeout-ms",
                    "50",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=3.0,
            )
        finally:
            thread.join(timeout=1.0)
            os.close(slave_fd)
            os.close(master_fd)

        self.assertFalse(thread.is_alive())
        self.assertEqual(emulator_errors, [])
        self.assertEqual(bytes(request[:2]), b"\xEB\x90")
        self.assertEqual(request[2], 1)
        self.assertEqual(request[4], 0x11)
        self.assertEqual(bytes(request[5:7]), b"\x0A\x06")
        self.assertEqual(request[-1], checksum(request))
        return result

    def test_fragmented_position_response_is_accepted(self):
        result = self.run_probe(respond=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[right] ONLINE", result.stdout)
        self.assertIn("0.1 0.2 0.3 0.4 0.5 0.6", result.stdout)

    def test_missing_hand_fails_without_sending_position_command(self):
        result = self.run_probe(respond=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("[right] NO RESPONSE", result.stderr)

    def test_silent_rs485_enters_two_hz_backoff_without_stopping_dds(self):
        service, master_fd, slave_fd = self.start_silent_daemon()
        try:
            request_times = self.collect_position_request_times(master_fd, 2.4)
            self.assertIsNone(service.poll(), "RS485 silence must not stop the DDS daemon")
            service.terminate()
            output, _ = service.communicate(timeout=3.0)
        finally:
            if service.poll() is None:
                service.kill()
                service.communicate()
            os.close(slave_fd)
            os.close(master_fd)

        self.assertEqual(service.returncode, 0, output)
        self.assertGreaterEqual(len(request_times), 5, request_times)
        self.assertLessEqual(
            len(request_times),
            20,
            f"silent hand was still queried near the online rate: {request_times}",
        )
        slow_gaps = [
            later - earlier
            for earlier, later in zip(request_times, request_times[1:])
            if later - earlier >= 0.35
        ]
        self.assertGreaterEqual(
            len(slow_gaps),
            2,
            f"expected repeated ~500 ms offline polling gaps: {request_times}",
        )
        self.assertIn("[right] NO RESPONSE", output)
        self.assertIn("DDS ready:", output)
        self.assertNotIn("[usb-watchdog]", output)

    def test_daemon_keeps_dds_alive_when_the_tty_device_disappears(self):
        service, master_fd, slave_fd = self.start_silent_daemon()
        output = ""
        master_open = True
        original_pid = service.pid
        try:
            time.sleep(0.4)
            self.assertIsNone(service.poll(), "daemon did not reach its running state")
            os.close(master_fd)
            master_open = False
            time.sleep(1.2)
            self.assertIsNone(service.poll(), "USB loss must not stop the DDS daemon")
            self.assertEqual(service.pid, original_pid)
            service.terminate()
            output, _ = service.communicate(timeout=3.0)
        finally:
            if service.poll() is None:
                service.kill()
                remaining, _ = service.communicate()
                output += remaining
            os.close(slave_fd)
            if master_open:
                os.close(master_fd)

        self.assertEqual(service.returncode, 0, output)
        self.assertIn("DDS ready:", output)
        self.assertNotIn("[usb-watchdog]", output)

    def test_missing_usb_at_start_recovers_through_symlink_in_same_process(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            port = os.path.join(temp_dir, "right-hand")
            service = subprocess.Popen(
                self.daemon_command(port),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            original_pid = service.pid
            master_fd = None
            slave_fd = None
            stop_emulator = threading.Event()
            response_sent = threading.Event()
            response_times = []
            emulator_errors = []

            try:
                time.sleep(0.35)
                self.assertIsNone(
                    service.poll(), "daemon must start even when the USB path is absent"
                )

                master_fd, slave_fd = pty.openpty()
                os.symlink(os.ttyname(slave_fd), port)

                def emulate_hand():
                    buffered = bytearray()
                    try:
                        while not stop_emulator.is_set():
                            readable, _, _ = select.select([master_fd], [], [], 0.1)
                            if not readable:
                                continue
                            buffered.extend(os.read(master_fd, 4096))
                            while True:
                                request = take_position_request(buffered)
                                if request is None:
                                    break
                                os.write(
                                    master_fd,
                                    position_response(
                                        request[2], (100, 200, 300, 400, 500, 600)
                                    ),
                                )
                                response_times.append(time.monotonic())
                                response_sent.set()
                    except OSError as error:
                        if not stop_emulator.is_set():
                            emulator_errors.append(error)
                    except Exception as error:  # pragma: no cover - assertion reports it
                        emulator_errors.append(error)

                thread = threading.Thread(target=emulate_hand)
                thread.start()
                self.assertTrue(
                    response_sent.wait(timeout=2.0),
                    "daemon did not reopen the newly attached USB serial path",
                )
                time.sleep(0.25)
                self.assertGreaterEqual(
                    len(response_times),
                    5,
                    f"recovered hand did not resume online polling: {response_times}",
                )
                self.assertIsNone(service.poll(), "daemon exited after USB recovery")
                self.assertEqual(service.pid, original_pid)
                service.terminate()
                output, _ = service.communicate(timeout=3.0)
            finally:
                stop_emulator.set()
                if "thread" in locals():
                    thread.join(timeout=1.0)
                if service.poll() is None:
                    service.kill()
                    remaining, _ = service.communicate()
                    output = locals().get("output", "") + remaining
                if slave_fd is not None:
                    os.close(slave_fd)
                if master_fd is not None:
                    os.close(master_fd)

        self.assertFalse(thread.is_alive())
        self.assertEqual(emulator_errors, [])
        self.assertEqual(service.returncode, 0, output)
        self.assertIn("DDS ready:", output)
        self.assertIn("[right] ONLINE", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("binary")
    args, remaining = parser.parse_known_args()
    Native485ProbeTest.binary = os.path.abspath(args.binary)
    unittest.main(argv=[__file__, *remaining])
