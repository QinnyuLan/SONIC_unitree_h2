"""Opt-in, private-loopback DDS integration tests. Never accesses a robot NIC."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from h2_ankle_sysid import KD, KP


def fixture(domain, failure):
    sys.path.insert(0, str(ROOT / "external_dependencies/unitree_sdk2_python"))
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_, unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    ChannelFactoryInitialize(domain, "lo")
    publisher = ChannelPublisher("rt/lowstate", LowState_)
    publisher.Init()
    state = unitree_hg_msg_dds__LowState_()
    state.imu_state.quaternion = [1., 0., 0., 0.]
    state.mode_machine = 1
    crc = CRC()
    command, first_command, conflict = None, None, None

    def receive(msg):
        nonlocal command, first_command
        if msg.crc != crc.Crc(msg):
            raise ValueError("invalid command CRC")
        command = msg
        if first_command is None:
            first_command = time.monotonic()

    subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
    subscriber.Init(receive)
    if failure == "preexisting_writer":
        conflict = ChannelPublisher("rt/lowcmd", LowCmd_)
        conflict.Init()
    print("FIXTURE_READY", flush=True)
    deadline = time.monotonic()
    for tick in range(60_000):
        state.tick = tick*2
        if command:
            for i in range(31):
                m = state.motor_state[i]
                bias = math.radians(6.08) if failure == "elbow_bias" and i in (18, 25) else 0.
                m.dq = (command.motor_cmd[i].q+bias-m.q)*10 if command.motor_cmd[i].kp else 0
                m.q += .002*m.dq
        failed = first_command is not None and time.monotonic()-first_command > 1
        if failed and failure == "imu_motion":
            # Rotate around all three axes, independent of the stationary fixture joints.
            half_angle = math.radians(30)
            state.imu_state.quaternion = [math.cos(half_angle)] + [math.sin(half_angle)/math.sqrt(3)]*3
        if failed and failure == "new_writer" and conflict is None:
            conflict = ChannelPublisher("rt/lowcmd", LowCmd_)
            conflict.Init()
        if not (failed and failure == "stale"):
            state.crc = crc.Crc(state)
            publisher.Write(state)
        if conflict:
            message = unitree_hg_msg_dds__LowCmd_()
            message.crc = crc.Crc(message)
            conflict.Write(message)
        deadline += .002
        time.sleep(max(0, deadline-time.monotonic()))


@unittest.skipUnless(os.environ.get("H2_SYSID_DDS_TESTS") == "1", "opt-in local DDS tests")
class DdsIntegration(unittest.TestCase):
    def run_case(self, failure, side="left", monitor=False):
        # Separate domain per case to avoid stale DDS discovery after teardown.
        domain = {"none": 91, "stale": 92, "new_writer": 93, "preexisting_writer": 94, "monitor": 95,
                  "operator_stop": 96}[failure]
        fixture_process = subprocess.Popen([sys.executable, __file__, "--fixture", "--domain", str(domain),
                                            "--failure", failure], stdout=subprocess.PIPE, text=True)
        try:
            while fixture_process.stdout.readline().strip() != "FIXTURE_READY":
                if fixture_process.poll() is not None:
                    self.fail("DDS fixture failed to start")
            with tempfile.TemporaryDirectory() as d:
                output = Path(d) / "run"
                cmd = [sys.executable, str(ROOT / "scripts/h2_ankle_sysid.py"),
                       "--target", "sim", "--interface", "lo", "--domain", str(domain),
                       "--side", side, "--frequencies", "1,2", "--cycles", "3",
                       "--kp-scale", "1.2", "--kd-scale", "1.1", "--output", str(output)]
                cmd += ["--monitor-seconds", "1"] if monitor else ["--execute", "--confirm-supported", "--auto"]
                if failure == "operator_stop":
                    process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    try:
                        until = time.monotonic()+10
                        while time.monotonic() < until:
                            metadata_path = output / "metadata.json"
                            try:
                                if json.loads(metadata_path.read_text()).get("status") == "armed":
                                    break
                            except (FileNotFoundError, json.JSONDecodeError):
                                pass
                            time.sleep(.05)
                        time.sleep(.5)
                        process.send_signal(signal.SIGINT)
                        out, err = process.communicate(timeout=5)
                        result = subprocess.CompletedProcess(cmd, process.returncode, out, err)
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait()
                else:
                    result = subprocess.run(cmd, text=True, capture_output=True, timeout=45)
                meta = json.loads((output / "metadata.json").read_text())
                with (output / "commands.csv").open() as f:
                    commands = list(csv.DictReader(f))
                with (output / "states.csv").open() as f:
                    states = list(csv.DictReader(f))
                return result, meta, commands, states
        finally:
            fixture_process.terminate()
            fixture_process.wait(timeout=5)
            fixture_process.stdout.close()

    def test_right_sweep_commands_and_damping(self):
        result, meta, commands, states = self.run_case("none", side="right")
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual(meta["status"], "completed")
        self.assertGreater(len(states), 1000)
        self.assertEqual({float(r["frequency_hz"]) for r in commands if int(r["measure"])}, {1., 2.})
        for row in commands:
            for i in range(31):
                self.assertEqual(float(row[f"dq_des_{i}"]), 0)
                self.assertEqual(float(row[f"tau_ff_{i}"]), 0)
                if row["phase"] == "damping":
                    self.assertEqual(float(row[f"kp_{i}"]), 0)
                    self.assertEqual(float(row[f"kd_{i}"]), 8)
                else:
                    if i != 11:
                        self.assertEqual(float(row[f"q_des_{i}"]), meta["base_q"][i])
                    self.assertAlmostEqual(float(row[f"kd_{i}"]), KD[i]*(1.1 if i == 11 else 1))
                    if row["phase"] != "gain_ramp":
                        self.assertAlmostEqual(float(row[f"kp_{i}"]), KP[i]*(1.2 if i == 11 else 1))
        self.assertEqual(commands[-1]["phase"], "damping")

    def test_state_stale_stops_and_damps(self):
        result, meta, commands, _ = self.run_case("stale")
        self.assertEqual(result.returncode, 1)
        self.assertIn("stale", meta["status"])
        self.assertEqual(commands[-1]["phase"], "damping")
        self.assertNotIn("excitation", {r["phase"] for r in commands})

    def test_other_writer_before_arm_never_publishes(self):
        result, _, commands, _ = self.run_case("preexisting_writer")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(commands, [])

    def test_other_writer_during_hold_stops_without_fighting(self):
        result, meta, commands, _ = self.run_case("new_writer")
        self.assertEqual(result.returncode, 1)
        self.assertIn("publisher", meta["status"])
        self.assertNotIn("excitation", {r["phase"] for r in commands})
        self.assertNotIn("damping", {r["phase"] for r in commands})

    def test_monitor_never_publishes(self):
        result, meta, commands, states = self.run_case("monitor", monitor=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual(meta["status"], "monitor_only")
        self.assertEqual(commands, [])
        self.assertGreater(len(states), 100)

    def test_operator_interrupt_damps_and_exits(self):
        result, meta, commands, _ = self.run_case("operator_stop")
        self.assertEqual(result.returncode, 1)
        self.assertIn("signal", meta["status"])
        self.assertEqual(commands[-1]["phase"], "damping")
        self.assertNotIn("excitation", {r["phase"] for r in commands})


if __name__ == "__main__":
    if "--fixture" in sys.argv:
        p = argparse.ArgumentParser()
        p.add_argument("--fixture", action="store_true")
        p.add_argument("--domain", type=int, required=True)
        p.add_argument("--failure", required=True)
        args = p.parse_args()
        if not 90 <= args.domain <= 99:
            p.error("test fixture only allows private domains 90-99 on lo")
        fixture(args.domain, args.failure)
    else:
        unittest.main()
