import argparse
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import importlib.util
import hashlib
import io
import math
import queue
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


s = module("h2_ankle_sysid")
a = module("analyze_h2_ankle_sysid")


def state():
    # A supported pose with no motors moving, not a commanded stand-up trajectory.
    return s.State(1_000_000_000, 0, 10, 0, 1, (0.,)*31, (0.,)*31, (0.,)*31,
                   (25,)*62, (0,)*31, (1., 0., 0., 0.), (0.,)*3, (0.,)*3, 0, True)


class Arguments(unittest.TestCase):
    def rejected(self, flags):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            s.parse_args(["--side", "left"] + flags)

    def test_preview_never_connects(self):
        with patch.object(s, "Transport", side_effect=AssertionError("must not connect")), redirect_stdout(io.StringIO()):
            self.assertEqual(s.run(s.parse_args(["--side", "left"])), 0)

    def test_no_automatic_real_control(self):
        self.rejected(["--execute"])
        self.rejected(["--execute", "--confirm-supported", "--auto"])
        self.rejected(["--execute", "--confirm-supported", "--monitor-seconds", "5"])
        self.rejected(["--target", "sim", "--interface", "eno1", "--domain", "73"])
        self.rejected(["--target", "sim", "--interface", "lo", "--domain", "0"])

    def test_invalid_numbers_and_frequency_order(self):
        for flag in ("--amplitude-deg", "--kp-scale", "--kd-scale", "--frequencies"):
            for value in ("nan", "inf", "-1", "0", "999"):
                self.rejected([flag, value])
        self.rejected(["--frequencies", "1,0.5"])
        self.rejected(["--frequencies", "1,1"])
        self.rejected(["--frequencies", "3", "--amplitude-deg", "2"])


class WaveAndGains(unittest.TestCase):
    def test_smooth_bounded_envelope(self):
        amplitude = math.radians(.5)
        for frequency in (.25, 1, 2):
            duration = 9/frequency
            times = np.linspace(0, duration, 10001)
            values = np.array([s.wave(t, frequency, amplitude, 5)[0] for t in times])
            self.assertEqual(values[0], 0)
            self.assertEqual(values[-1], 0)
            self.assertLessEqual(max(abs(values)), amplitude + 1e-12)
            self.assertLess(abs(values[1] / times[1]), 1e-5)
            self.assertFalse(s.wave(1/frequency, frequency, amplitude, 5)[1])
            self.assertTrue(s.wave(3/frequency, frequency, amplitude, 5)[1])

    def test_exact_dds_names_and_only_selected_pitch_scaled(self):
        self.assertEqual(len(s.NAMES), 31)
        self.assertEqual(s.NAMES[5], "left_ankle_pitch_joint")
        self.assertEqual(s.NAMES[11], "right_ankle_pitch_joint")
        for index in (5, 11):
            kp, kd = s.gains(index, 1.2, 1.1)
            for i in range(31):
                self.assertAlmostEqual(kp[i], s.KP[i]*(1.2 if i == index else 1))
                self.assertAlmostEqual(kd[i], s.KD[i]*(1.1 if i == index else 1))
            self.assertEqual(s.gains(index, 1.2, 1.1, damping=True), ([0.]*31, [8.]*31))


class Guards(unittest.TestCase):
    def setUp(self):
        self.limits = s.limits_from_asset()
        self.state = state()

    def test_valid_state(self):
        s.check_state(self.state, 1_000_000_000, self.limits)

    def test_stale_nan_modes_faults_temperature_remote(self):
        bad = [None, replace(self.state, receive_ns=0), replace(self.state, mode_pr=1),
               replace(self.state, errors=(1,)+(0,)*30), replace(self.state, temperature=(65,)*62),
               replace(self.state, q=(math.nan,)+(0.,)*30), replace(self.state, remote_keys=8),
               replace(self.state, quaternion=(0.,)*4)]
        for item in bad:
            with self.subTest(item=item), self.assertRaises(s.SafetyStop):
                s.check_state(item, 1_000_000_000, self.limits)

    def test_ankle_speed_is_symmetric(self):
        for speed in (-1.1, 1.1):
            dq = list(self.state.dq)
            dq[5] = speed
            with self.assertRaises(s.SafetyStop):
                s.check_state(replace(self.state, dq=tuple(dq)), 1_000_000_000, self.limits)

    def test_position_error_torque_and_orientation(self):
        q, tau = [0.]*31, [0.]*31
        q[5] = .09
        tau[5] = 13
        for item in (replace(self.state, q=tuple(q)), replace(self.state, tau=tuple(tau)),
                     replace(self.state, quaternion=(math.cos(.2), math.sin(.2), 0., 0.))):
            with self.assertRaises(s.SafetyStop):
                s.check_state(item, 1_000_000_000, self.limits, base=(0.,)*31,
                              target=(0.,)*31, initial_quat=self.state.quaternion)

    def test_no_orientation_baseline_preserves_other_state_guards(self):
        rotated = replace(self.state, quaternion=(math.cos(.3), math.sin(.3), 0., 0.))
        s.check_state(rotated, 1_000_000_000, self.limits, initial_quat=None)
        with self.assertRaisesRegex(s.SafetyStop, "body orientation"):
            s.check_state(rotated, 1_000_000_000, self.limits, initial_quat=self.state.quaternion)
        q, dq, tau = [0.]*31, [0.]*31, [0.]*31
        q[5], dq[5], tau[5] = .09, 1.1, 13
        bad = [replace(rotated, q=tuple(q)), replace(rotated, dq=tuple(dq)),
               replace(rotated, tau=tuple(tau)), replace(rotated, receive_ns=0),
               replace(rotated, mode_pr=1), replace(rotated, errors=(1,)+(0,)*30),
               replace(rotated, temperature=(65,)*62), replace(rotated, remote_keys=8),
               replace(rotated, quaternion=(0.,)*4), replace(rotated, gyro=(math.nan, 0., 0.))]
        for item in bad:
            with self.subTest(item=item), self.assertRaises(s.SafetyStop):
                s.check_state(item, 1_000_000_000, self.limits, target=(0.,)*31, initial_quat=None)

    def test_frozen_tick_and_duplicate_publishers(self):
        t = object.__new__(s.Transport)
        t.error = None
        t.recorder = argparse.Namespace(error=None)
        t.args = argparse.Namespace(target="real")
        t.tick_changed_ns = 0
        t.state_writers, t.command_writers, t.own_writer_created = 1, 0, False
        with self.assertRaises(s.SafetyStop):
            t.healthy()
        t.args.target = "sim"
        t.healthy()
        t.command_writers = 1
        with self.assertRaises(s.SafetyStop):
            t.healthy()
        t.own_writer_created = True
        t.healthy()
        t._command_match(None, argparse.Namespace(current_count=2))
        with self.assertRaises(s.SafetyStop):
            t.healthy()


class LogsAndAnalysis(unittest.TestCase):
    def test_recorder_flush_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "run"
            rec = s.Recorder(path)
            rec.event("test")
            rec.close()
            self.assertIsNone(rec.error)
            self.assertEqual(rec.counts["events"], 1)
            with self.assertRaises(FileExistsError):
                s.Recorder(path)

    def test_sine_fit_recovers_gain_phase_and_offset(self):
        t = np.linspace(0, 10, 5001)
        x = .01*np.sin(2*np.pi*t)+.2
        y = .008*np.sin(2*np.pi*t-.3)-.5+.001*t
        u, _ = a.fit_sine(t, x, 1)
        v, residual = a.fit_sine(t, y, 1)
        self.assertAlmostEqual(abs(v/u), .8)
        self.assertAlmostEqual(np.angle(v/u), -.3)
        self.assertLess(residual, 1e-12)

    def test_log_queue_overflow_is_fatal(self):
        rec = object.__new__(s.Recorder)
        rec.queue = queue.Queue(maxsize=1)
        rec.error = None
        rec.put("events", {})
        rec.put("events", {})
        self.assertIn("overflow", rec.error)

    def test_dds_command_has_valid_crc_and_unused_motors_disabled(self):
        sys.path.insert(0, str(ROOT / "external_dependencies/unitree_sdk2_python"))
        from unitree_sdk2py.utils.crc import CRC
        t = object.__new__(s.Transport)
        t.crc = CRC()
        captured, logged = [], []
        t.writer = SimpleNamespace(write=captured.append)
        t.recorder = SimpleNamespace(put=lambda kind, row: logged.append((kind, row)))
        t.get = state
        target = [0.]*31
        target[5] = .01
        kp, kd = s.gains(5, 1.2, 1.1)
        t.send(target, kp, kd, 1, "excitation", .25, 3., False, 7)
        cmd = captured[0]
        self.assertEqual(cmd.crc, t.crc.Crc(cmd))
        self.assertEqual(cmd.mode_pr, 0)
        self.assertEqual(cmd.mode_machine, 1)
        for i, motor in enumerate(cmd.motor_cmd):
            self.assertEqual(motor.mode, int(i < 31))
            self.assertEqual(motor.dq, 0)
            self.assertEqual(motor.tau, 0)
            self.assertEqual(motor.kp, kp[i] if i < 31 else 0)
            self.assertEqual(motor.kd, kd[i] if i < 31 else 0)
        self.assertEqual(len(logged[0][1]), len(s.CMD_FIELDS))

    def test_hardware_crc_failure_is_not_usable_state(self):
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
        t = object.__new__(s.Transport)
        msg = unitree_hg_msg_dds__LowState_()
        t.crc = SimpleNamespace(Crc=lambda msg: 123)
        rows = []
        t.recorder = SimpleNamespace(put=lambda kind, row: rows.append(row))
        t.args = SimpleNamespace(target="real")
        t.error, t.latest = None, None
        t._read_states(SimpleNamespace(take=lambda n: [msg]))
        self.assertIsNone(t.latest)
        self.assertIn("CRC", t.error)
        self.assertEqual(len(rows[0]), len(s.STATE_FIELDS))


class OfflineFixture(unittest.TestCase):
    def test_official_ankle_linkages_remain_dynamic_and_asset_is_unchanged(self):
        sim = module("sim_h2_ankle_sysid")
        before = hashlib.sha256(s.ASSET.read_bytes()).hexdigest()
        env = sim.FixtureEnv(sim.make_config(73), onscreen=False)
        self.assertEqual(hashlib.sha256(s.ASSET.read_bytes()).hexdigest(), before)
        self.assertTrue(np.all(env.mj_model.geom_contype == 0))
        self.assertTrue(np.all(env.mj_model.geom_conaffinity == 0))
        equalities = {env.mj_model.equality(i).name for i in range(env.mj_model.neq)}
        self.assertIn("fixture_pelvis", equalities)
        for name in s.NAMES:
            self.assertEqual("fixture_"+name in equalities, "ankle_" not in name)
        self.assertIn("left_ankle_connect", equalities)
        self.assertIn("right_ankle_connect", equalities)
        for group in env.closed_loop_transmissions:
            jac = env._closed_loop_output_to_motor_jacobian(group)
            self.assertTrue(np.isfinite(jac).all())


if __name__ == "__main__":
    unittest.main()
