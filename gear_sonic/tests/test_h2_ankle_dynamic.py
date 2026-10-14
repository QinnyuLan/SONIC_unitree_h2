"""Dynamic ankle protocol/analysis tests; optional DDS tests use only lo/90."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
import math
import os
from pathlib import Path
import pty
import select
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import h2_ankle_range_test as r
import analyze_h2_ankle_sysid as analysis


class DynamicProtocol(unittest.TestCase):
    def test_extended_plan_covers_large_angles_and_high_frequencies_with_explicit_caps(self):
        args = r.parse_args(["--experiment", "dynamic", "--sweep-plan",
                            str(ROOT / "scripts/configs/h2_ankle_dynamic_large_range.json")])
        stages = r.plan(args, r.ready_pose(), r.limits_from_asset())
        self.assertEqual(len(stages), 18)
        self.assertEqual(max(s.amplitude_deg for s in stages), 25)
        self.assertEqual(max(s.frequency_hz for s in stages), 3)
        self.assertEqual(args.kp_scale, 1)
        self.assertEqual(args.kd_scale, 1)
        self.assertAlmostEqual(sum((args.cycles+4)/s.frequency_hz+3 for s in stages), 513.)
        self.assertEqual([s.amplitude_deg for s in stages if s.frequency_hz == .25], [3, 6, 10, 15, 25])
        self.assertEqual(args.target_speed_limit_rad_s, .8)
        self.assertEqual(args.target_accel_limit_rad_s2, 8.)
        self.assertEqual(len({r.command_phase(s, "excitation", args) for s in stages}), 18)
        for stage in stages:
            segment = r.trial_segments(stage, r.ready_pose(), args)[0]
            times = np.linspace(0, segment[3], 10001)
            qs = np.array([r.trial_target(stage, segment, t, args)[0] for t in times])
            velocity = np.gradient(qs[:, stage.index], times)
            self.assertLess(np.max(np.abs(velocity)), args.target_speed_limit_rad_s)
            self.assertLess(np.max(np.abs(np.gradient(velocity, times))), args.target_accel_limit_rad_s2)
            other = [i for i in range(31) if i != stage.index]
            self.assertTrue(np.all(qs[:, other] == np.asarray(r.ready_pose())[other]))

    def test_extended_plan_rejects_unsafe_unknown_or_ambiguous_settings(self):
        valid = dict(schema="h2_ankle_sweep_plan_v1", bands=[dict(amplitude_deg=3, frequencies_hz=[.25])])
        bad = [dict(valid, bands=[dict(amplitude_deg=25, frequencies_hz=[3])]),
               dict(valid, target_speed_limit_rad_s=1.01), dict(valid, target_accel_limit_rad_s2=9),
               dict(valid, target_speed_limit_rad_s=True), dict(valid, unknown_option=1),
               dict(valid, bands=[dict(amplitude_deg=26, frequencies_hz=[.1])]),
               dict(valid, bands=[dict(amplitude_deg=3, frequencies_hz=[])]),
               dict(valid, bands=[dict(amplitude_deg=True, frequencies_hz=[.25])]),
               dict(valid, bands=valid["bands"]*2)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            for data in bad:
                r.save_json(path, data)
                with self.subTest(data=data), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    r.parse_args(["--experiment", "dynamic", "--sweep-plan", str(path)])
            r.save_json(path, valid)
            for extra in (["--amplitude-deg", "3"], ["--frequencies", ".25"], ["--full-range"]):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    r.parse_args(["--experiment", "dynamic", "--sweep-plan", str(path), *extra])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                r.parse_args(["--sweep-plan", str(path)])

    def test_multi_amplitude_trials_are_not_pooled_by_frequency(self):
        specs = [dict(amplitude_deg=a, frequency_hz=.5, excitation_phase=f"trial_{a}") for a in (3, 10)]
        self.assertEqual(analysis.trial_specs(dict(schema="h2_ankle_dynamic_v2", stages=specs)), specs)
        with self.assertRaisesRegex(ValueError, "distinct"):
            analysis.trial_specs(dict(schema="h2_ankle_dynamic_v2", stages=specs*2))
        rows = [dict(command_amplitude_deg=a, frequency_hz=.5, gain=g, phase_deg=-20,
                     output_harmonic_ratio_2_3=.1, quality_flags=[]) for a, g in ((10, .9), (3, .5))]
        comparison = analysis.amplitude_comparisons(rows)[0]
        self.assertEqual(comparison["amplitudes_deg"], [3, 10])
        self.assertEqual(comparison["gains"], [.5, .9])

    def test_defaults_are_fixed_amplitude_local_left_sine(self):
        args = r.parse_args(["--experiment", "dynamic"])
        ready = r.ready_pose()
        stages = r.plan(args, ready, r.limits_from_asset())
        self.assertEqual(args.side, "left")
        self.assertEqual(args.cycles, 6)
        self.assertEqual([s.frequency_hz for s in stages], [.25, .5, .75, 1])
        for s in stages:
            self.assertEqual(s.index, 5)
            self.assertAlmostEqual((s.lower+s.upper)/2, ready[5])
            self.assertAlmostEqual(math.degrees(s.upper-s.lower), 6)

    def test_preview_does_not_create_dds_or_release_control(self):
        output = io.StringIO()
        with patch.object(r, "Transport", side_effect=AssertionError("DDS")), \
             patch.object(r, "create_motion_switcher", side_effect=AssertionError("RPC")), \
             redirect_stdout(output):
            self.assertEqual(r.run(r.parse_args(["--experiment", "dynamic"])), 0)
        self.assertIn("Body IMU orientation stop: DISABLED", output.getvalue())

    def test_range_preview_does_not_disable_body_orientation_stop(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(r.run(r.parse_args([])), 0)
        self.assertNotIn("Body IMU orientation stop: DISABLED", output.getvalue())

    def test_invalid_or_unsafe_combinations_rejected(self):
        invalid = (["--side", "both"], ["--full-range"], ["--speed-deg-s", "5"],
                   ["--amplitude-deg", "nan"], ["--amplitude-deg", "5", "--frequencies", "1"],
                   ["--frequencies", ".5,.25"], ["--frequencies", "nan"],
                   ["--frequencies", "2"], ["--cycles", "2"], ["--cycles", "13"],
                   ["--execute"], ["--execute", "--auto", "--confirm-supported"])
        for extra in invalid:
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                r.parse_args(["--experiment", "dynamic", *extra])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            r.parse_args(["--frequencies", ".25"])

    def test_wave_has_constant_measurement_window_and_holds_other_joints(self):
        args = r.parse_args(["--experiment", "dynamic", "--side", "right"])
        ready = r.ready_pose()
        for stage in r.plan(args, ready, r.limits_from_asset()):
            segments = r.trial_segments(stage, ready, args)
            self.assertEqual(segments[1][3], 3.)
            duration = segments[0][3]
            times = np.linspace(0, duration, 10001)
            samples = [r.trial_target(stage, segments[0], t, args) for t in times]
            qs = np.asarray([v[0] for v in samples])
            np.testing.assert_allclose(qs[0], ready, atol=1e-12)
            np.testing.assert_allclose(qs[-1], ready, atol=1e-12)
            unchanged = [i for i in range(31) if i != stage.index]
            self.assertTrue(np.all(qs[:, unchanged] == np.asarray(ready)[unchanged]))
            self.assertLessEqual(np.max(np.abs(qs[:, 11]-ready[11])), math.radians(3)+1e-12)
            velocity = np.gradient(qs[:, 11], times)
            acceleration = np.gradient(velocity, times)
            self.assertLess(np.max(np.abs(velocity)), r.DYNAMIC_SPEED_LIMIT)
            self.assertLess(np.max(np.abs(acceleration)), r.DYNAMIC_ACCEL_LIMIT)
            for cycles, measure in ((1, False), (3, True), (7, True), (9, False)):
                self.assertEqual(r.trial_target(stage, segments[0], cycles/stage.frequency_hz, args)[1], measure)
            self.assertEqual(r.trial_target(stage, segments[1], 1., args), (ready, False))

    def test_frf_recovers_gain_phase_despite_bias_and_drift(self):
        f = .75
        t = np.arange(0, 6/f, .002)
        phase, gain = math.radians(-37), .72
        u = .052*np.sin(2*np.pi*f*t)-.363
        y = .052*gain*np.sin(2*np.pi*f*t+phase)-.34+.001*t
        h = analysis.fit_sine(t, y, f)[0]/analysis.fit_sine(t, u, f)[0]
        self.assertAlmostEqual(abs(h), gain, places=10)
        self.assertAlmostEqual(np.angle(h), phase, places=10)
        quality = analysis.dynamic_quality(t, u, y, f, h)
        self.assertLess(quality["output_harmonic_ratio_2_3"], 1e-10)
        self.assertLess(quality["cycle_gain_cv"], 1e-10)
        self.assertLess(quality["cycle_phase_std_deg"], 1e-8)
        self.assertEqual(quality["fitted_cycles"], 6)

    def test_harmonic_distortion_and_variable_response_are_visible(self):
        t = np.arange(0, 6, .002)
        u = .05*np.sin(2*np.pi*t)
        y = u+.015*np.sin(4*np.pi*t)
        quality = analysis.dynamic_quality(t, u, y, 1, 1+0j)
        self.assertAlmostEqual(quality["output_harmonic_ratio_2_3"], .3, places=10)
        y = u*np.where(t<3, 1., .25)
        quality = analysis.dynamic_quality(t, u, y, 1, .625+0j)
        self.assertGreater(quality["cycle_gain_cv"], .2)


@unittest.skipUnless(os.environ.get("H2_SYSID_DDS_TESTS") == "1", "private loopback integration")
class DynamicDds(unittest.TestCase):
    def run_with_two_keys(self, command):
        master, slave = pty.openpty()
        process = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave)
        os.close(slave)
        transcript = ""
        sent_y = sent_n = False
        deadline = time.monotonic()+120
        try:
            while process.poll() is None:
                if time.monotonic()>deadline:
                    self.fail("interactive protocol timeout: "+transcript)
                readable, _, _ = select.select([master], [], [], .1)
                if readable:
                    try:
                        data = os.read(master, 65536)
                    except OSError:
                        break
                    transcript += data.decode(errors="replace")
                if not sent_y and "No motor commands yet." in transcript:
                    self.assertNotIn("Armed.", transcript)
                    os.write(master, b"y")
                    sent_y = True
                if not sent_n and "Init Done." in transcript:
                    os.write(master, b"n")
                    sent_n = True
            code = process.wait(timeout=5)
            self.assertEqual(code, 0, transcript)
            self.assertTrue(sent_y and sent_n, transcript)
            return transcript
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            os.close(master)

    def run_case(self, multi_amplitude=False, fixture_failure="elbow_bias", replay=True):
        fixture = subprocess.Popen([sys.executable, str(ROOT / "gear_sonic/tests/test_h2_ankle_sysid_dds.py"),
            "--fixture", "--domain", "90", "--failure", fixture_failure], stdout=subprocess.PIPE, text=True)
        try:
            while fixture.stdout.readline().strip() != "FIXTURE_READY":
                if fixture.poll() is not None:
                    self.fail("fixture failed")
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "run"
                if multi_amplitude:
                    plan_path = Path(tmp) / "plan.json"
                    r.save_json(plan_path, dict(schema="h2_ankle_sweep_plan_v1",
                        target_speed_limit_rad_s=.8, target_accel_limit_rad_s2=8,
                        bands=[dict(amplitude_deg=6, frequencies_hz=[.25, 1]),
                               dict(amplitude_deg=25, frequencies_hz=[.25]),
                               dict(amplitude_deg=1, frequencies_hz=[3])]))
                    excitation = ["--sweep-plan", str(plan_path)]
                    expected = [(6, .25), (6, 1), (25, .25), (1, 3)]
                else:
                    excitation = ["--amplitude-deg", "1", "--frequencies", "1,1.5"]
                    expected = [(1, 1), (1, 1.5)]
                transcript = self.run_with_two_keys([sys.executable, str(ROOT / "scripts/h2_ankle_range_test.py"),
                    "--experiment", "dynamic", "--target", "sim", "--interface", "lo", "--domain", "90",
                    "--side", "right", *excitation, "--cycles", "3",
                    "--execute", "--confirm-supported", "--output", str(out)])
                self.assertNotIn("ARM H2 READY", transcript)
                self.assertNotIn("RELEASE H2 CONTROL", transcript)
                meta = json.loads((out / "metadata.json").read_text())
                self.assertEqual(meta["schema"], "h2_ankle_dynamic_v2" if multi_amplitude else "h2_ankle_dynamic_v1")
                self.assertEqual(meta["active_index"], 11)
                self.assertFalse(meta["imu_orientation_stop_enabled"])
                self.assertEqual([(s["amplitude_deg"], s["frequency_hz"]) for s in meta["completed_stages"]], expected)
                if fixture_failure == "imu_motion":
                    initial = np.asarray(meta["initial_quaternion"])
                    with (out / "states.csv").open() as f:
                        quats = np.asarray([[float(row[f"imu_quat_wxyz_{i}"]) for i in range(4)]
                                            for row in csv.DictReader(f)])
                    dots = np.abs(quats @ initial) / (np.linalg.norm(quats, axis=1)*np.linalg.norm(initial))
                    self.assertGreater(np.degrees(2*np.arccos(np.clip(dots, 0, 1))).max(), 50)
                    self.assertTrue(np.any(np.all(np.abs(quats[:, 1:])>.2, axis=1)))
                events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
                self.assertEqual(next(e for e in events if e["event"] == "start_authorized")["source"], "operator_y")
                with (out / "commands.csv").open() as f:
                    rows = list(csv.DictReader(f))
                measured = [v for v in rows if v["measure"] == "1"]
                self.assertTrue(measured)
                valid_phases = {s["excitation_phase"] for s in meta["stages"]}
                for row in measured:
                    self.assertIn(row["phase"], valid_phases)
                    for i in range(31):
                        self.assertEqual(float(row[f"kp_{i}"]), r.KP[i])
                        self.assertEqual(float(row[f"kd_{i}"]), r.KD[i])
                        if i != 11:
                            self.assertEqual(float(row[f"q_des_{i}"]), meta["ready_q"][i])
                report = analysis.analyze(out)
                self.assertEqual(len(report["frequencies"]), len(expected))
                self.assertEqual(report["unmeasured_trials"], [])
                for row, (amplitude, frequency) in zip(report["frequencies"], expected):
                    self.assertEqual(row["command_amplitude_deg"], amplitude)
                    self.assertEqual(row["frequency_hz"], frequency)
                    self.assertAlmostEqual(row["input_amplitude_deg"], amplitude, delta=.03)
                    self.assertNotIn("incomplete_frequency_window", row["quality_flags"])
                if not replay:
                    return
                replay = subprocess.run([sys.executable, str(ROOT / "scripts/sim_h2_ankle_sysid.py"),
                                         "--replay-log", str(out)], capture_output=True, text=True, timeout=180)
                self.assertEqual(replay.returncode, 0, replay.stdout+replay.stderr)
                sim = analysis.analyze(out / "sim_replay")
                self.assertEqual(len(sim["frequencies"]), len(expected))
                self.assertEqual(sim["status"], "completed")
                for row in sim["frequencies"]:
                    self.assertTrue(math.isfinite(row["gain"]))
                    self.assertNotIn("simulation_not_realtime", row["quality_flags"])
                    self.assertLess(row["peak_speed_deg_s"], math.degrees(1))
                    self.assertLess(row["peak_tracking_error_deg"], math.degrees(.12))
                compared = analysis.compare_replay(report, sim, out / "real_vs_sim.csv")
                self.assertEqual(len(compared), len(expected))
                if multi_amplitude:
                    self.assertEqual(report["amplitude_comparison"][0]["amplitudes_deg"], [6, 25])
                    self.assertEqual(len(sim["amplitude_comparison"]), 1)
                with self.assertRaisesRegex(ValueError, "own offline replay"):
                    analysis.compare_replay(sim, report, out / "invalid.csv")
        finally:
            fixture.terminate()
            fixture.wait(timeout=5)
            fixture.stdout.close()

    def test_sweep_logging_analysis_and_offline_official_h2_replay(self):
        self.run_case()

    def test_large_angle_and_high_frequency_plan_with_two_keys_and_offline_replay(self):
        self.run_case(multi_amplitude=True)

    def test_body_imu_rotation_does_not_interrupt_supported_dynamic_sweep(self):
        self.run_case(fixture_failure="imu_motion", replay=False)


if __name__ == "__main__":
    unittest.main()
