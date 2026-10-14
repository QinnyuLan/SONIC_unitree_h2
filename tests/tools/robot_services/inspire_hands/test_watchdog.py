"""Exercise the real service's readiness/heartbeat and a blocked serial worker."""
import argparse
import os
from pathlib import Path
import pty
import signal
import socket
import subprocess
import tempfile
import time
import unittest


class WatchdogTest(unittest.TestCase):
    build = Path(__file__).resolve().parents[4] / "build/inspire_hands"
    binary = str(build / "bin/inspire_hand_service")
    stall_library = str(build / "libstall_serial.so")

    @classmethod
    def setUpClass(cls):
        if not Path(cls.binary).is_file() or not Path(cls.stall_library).is_file():
            raise unittest.SkipTest("Build the hand service with -DBUILD_TESTING=ON")

    def collect(self, receiver, seconds):
        messages = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            receiver.settimeout(max(0.001, deadline - time.monotonic()))
            try:
                messages.append(receiver.recv(4096).decode())
            except socket.timeout:
                break
        return messages

    def exercise(self, *, abstract=False, stall=False):
        with tempfile.TemporaryDirectory(prefix="inspire-watchdog-") as tmp:
            receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            address = f"{tmp}/notify"
            if abstract:
                address = f"\0inspire-watchdog-{os.getpid()}-{time.monotonic_ns()}"
            receiver.bind(address)
            env = {k: v for k, v in os.environ.items()
                   if k not in {"NOTIFY_SOCKET", "WATCHDOG_USEC", "WATCHDOG_PID", "LD_PRELOAD"}}
            env.update(NOTIFY_SOCKET=address.replace("\0", "@"), WATCHDOG_USEC="600000")
            master_fd, slave_fd = pty.openpty()
            if stall:
                env["LD_PRELOAD"] = self.stall_library
            with open(f"{tmp}/service.log", "w+") as log:
                service = subprocess.Popen([
                    self.binary, "--network", "lo", "--domain", "218",
                    "--namespace", f"inspire_watchdog_{os.getpid()}_{time.monotonic_ns()}",
                    "--hands", "right", "--right-port",
                    os.ttyname(slave_fd) if stall else f"{tmp}/missing-usb",
                    "--probe-attempts", "1", "--serial-timeout-ms", "10",
                ], env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    first = self.collect(receiver, 4.0)
                    self.assertTrue(any("READY=1" in message for message in first), first)
                    self.assertTrue(any("WATCHDOG=1" in message for message in first), first)
                    if stall:
                        self.assertTrue(any("worker stalled" in message for message in first), first)
                        later = self.collect(receiver, 0.8)
                        self.assertFalse(any("WATCHDOG=1" in message for message in later), later)
                        self.assertIsNone(service.poll())
                    else:
                        later = self.collect(receiver, 0.8)
                        self.assertTrue(any("WATCHDOG=1" in message for message in later), later)
                        service.send_signal(signal.SIGSTOP)
                        self.collect(receiver, 0.3)  # drain an in-flight datagram
                        self.assertFalse(self.collect(receiver, 0.8))
                        service.send_signal(signal.SIGCONT)
                        service.terminate()
                        self.assertTrue(any("STOPPING=1" in message
                                            for message in self.collect(receiver, 1.5)))
                        self.assertEqual(service.wait(timeout=3), 0)
                finally:
                    if service.poll() is None:
                        service.kill()
                    service.wait(timeout=3)
                    receiver.close()
                    os.close(master_fd)
                    os.close(slave_fd)

    def test_missing_usb_keeps_heartbeat_and_shutdown_notifies(self):
        self.exercise()

    def test_abstract_notification_socket(self):
        self.exercise(abstract=True)

    def test_stalled_serial_worker_stops_heartbeat(self):
        self.exercise(stall=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("binary")
    parser.add_argument("stall_library")
    args, remaining = parser.parse_known_args()
    WatchdogTest.binary = str(Path(args.binary).resolve())
    WatchdogTest.stall_library = str(Path(args.stall_library).resolve())
    unittest.main(argv=[__file__, *remaining])
