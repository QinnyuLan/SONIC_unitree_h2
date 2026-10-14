from contextlib import redirect_stdout, redirect_stderr
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import h2_ankle_range_test as r


class RangePlan(unittest.TestCase):
    def test_ready_angles_match_release_deploy(self):
        q = r.ready_pose()
        expected = (-.312, 0, 0, .669, 0, -.363)*2 + (0,0,0) + \
                   (.2,.2,0,.6,0,0,0) + (.2,-.2,0,.6,0,0,0) + (0,0)
        np.testing.assert_allclose(q, expected, atol=0, rtol=0)

    def test_preview_never_connects(self):
        with patch.object(r, "Transport", side_effect=AssertionError("DDS")), redirect_stdout(io.StringIO()):
            self.assertEqual(r.run(r.parse_args(["--full-range"])), 0)

    def test_full_range_progressive_and_both_sides(self):
        ready, limits = r.ready_pose(), r.limits_from_asset()
        stages = r.plan(r.parse_args(["--full-range"]), ready, limits)
        self.assertEqual(len(stages), 8)
        self.assertEqual([s.index for s in stages], [5]*4+[11]*4)
        for group in (stages[:4], stages[4:]):
            self.assertAlmostEqual(group[-1].lower, limits[group[-1].index][0]+r.MARGIN)
            self.assertAlmostEqual(group[-1].upper, limits[group[-1].index][1]-r.MARGIN)
            for a,b in zip(group, group[1:]):
                self.assertLessEqual(b.lower, a.lower)
                self.assertGreaterEqual(b.upper, a.upper)

    def test_only_one_target_joint_changes_and_moves_are_smooth(self):
        ready = r.ready_pose()
        stages = r.plan(r.parse_args(["--full-range"]), ready, r.limits_from_asset())
        for stage in stages:
            for label, a, b, duration in r.stage_segments(stage, ready, math.radians(5)):
                ts = np.linspace(0, duration, 501)
                qs = np.asarray([r.smooth_move(a,b,t,duration) for t in ts])
                for i in range(31):
                    if i != stage.index:
                        self.assertTrue(np.all(qs[:,i] == ready[i]))
                self.assertLessEqual(np.max(np.abs(np.gradient(qs[:,stage.index], ts))), math.radians(5)+1e-6)
                np.testing.assert_allclose(qs[0],a,atol=1e-12)
                np.testing.assert_allclose(qs[-1],b,atol=1e-12)

    def test_invalid_modes_and_ranges_fail(self):
        for args in (["--execute"], ["--execute","--auto","--confirm-supported"],
                     ["--target","sim"], ["--speed-deg-s","nan"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                r.parse_args(args)
        for interval in ((-65,35), (0,10), (-30,-40)):
            args = r.parse_args(["--range-deg", *map(str, interval)])
            with self.assertRaises(ValueError):
                r.plan(args, r.ready_pose(), r.limits_from_asset())

    def test_stalled_ankle_cannot_be_forced_through_range(self):
        ready = r.ready_pose()
        state = type("State", (), {"q": ready})()
        target = list(ready)
        target[5]+=.13
        with self.assertRaises(r.SafetyStop):
            r.check_tracking(state, target, ready, 5)

    def test_other_joints_must_hold_ready_and_preparation_must_settle(self):
        ready = r.ready_pose()
        state = type("State", (), {"q": ready, "dq": (0.,)*31})()
        self.assertTrue(r.reached_ready(state, ready))
        q = list(ready)
        q[3] += .16
        state.q = tuple(q)
        self.assertFalse(r.reached_ready(state, ready))
        with self.assertRaises(r.SafetyStop):
            r.check_tracking(state, ready, ready, 5)

    def test_measured_static_elbow_offsets_pass_without_changing_targets(self):
        ready = r.ready_pose()
        q = list(ready)
        for i, error_deg in ((18, 6.0788), (25, 6.0639), (5, 1.8772), (11, 3.0670)):
            q[i] += math.radians(error_deg)
        state = SimpleNamespace(q=tuple(q), dq=(0.,)*31)
        self.assertTrue(r.reached_ready(state, ready))
        self.assertEqual(r.ready_blockers(state, ready), [])
        for active in (5, 11):
            r.check_tracking(state, ready, ready, active)

    def test_ankle_limits_remain_strict_and_other_hold_bounds_remain(self):
        ready = r.ready_pose()
        for i, limit in ((4, .1), (5, .1), (10, .1), (11, .1), (18, .15), (25, .15)):
            for direction in (-1, 1):
                q = list(ready)
                q[i] += direction*(limit-1e-6)
                state = SimpleNamespace(q=tuple(q), dq=(0.,)*31)
                self.assertTrue(r.reached_ready(state, ready))
                q[i] += direction*2e-6
                state.q = tuple(q)
                self.assertFalse(r.reached_ready(state, ready))
                self.assertEqual(r.ready_blockers(state, ready)[0]["index"], i)

    def test_speed_and_nonfinite_values_still_fail_readiness(self):
        ready = r.ready_pose()
        for i in (5, 18, 25):
            for value in (-.151, .151, math.nan, math.inf):
                dq = [0.]*31
                dq[i] = value
                state = SimpleNamespace(q=ready, dq=tuple(dq))
                self.assertFalse(r.reached_ready(state, ready))
                self.assertEqual(r.ready_blockers(state, ready)[0]["kind"], "speed")

    def test_readiness_diagnostic_identifies_joint_and_condition(self):
        ready = r.ready_pose()
        q, dq = list(ready), [0.]*31
        q[11] += .11
        dq[18] = .16
        blockers = r.ready_blockers(SimpleNamespace(q=q, dq=dq), ready)
        message = r.ready_failure_summary(blockers)
        self.assertIn("right_ankle_pitch_joint position +6.30 deg", message)
        self.assertIn("left_elbow_joint speed +9.17 deg/s", message)
        self.assertIn("not stable", r.ready_failure_summary([]))

    def test_range_fixture_and_sampled_closed_loop_workspace(self):
        import sim_h2_ankle_sysid as sim

        with tempfile.TemporaryDirectory() as tmp:
            metadata = Path(tmp) / "metadata.json"
            r.save_json(metadata, dict(schema="h2_ankle_range_v1", joints_dds=r.NAMES,
                                       base_q=r.ready_pose(), initial_quaternion=[1,0,0,0]))
            config = sim.make_config(73, metadata)
        self.assertFalse(config["FIXTURE_LOCK_NON_ANKLE"])
        env = sim.FixtureEnv(config, onscreen=False)
        equalities = {env.mj_model.equality(i).name for i in range(env.mj_model.neq)}
        self.assertEqual({n for n in equalities if n.startswith("fixture_")}, {"fixture_pelvis"})
        self.assertTrue(np.all(env.mj_model.geom_contype == 0))
        ready = env._h2_local_config_values_to_sim_order(
            np.asarray(config["DEFAULT_DOF_ANGLES"]), "DEFAULT_DOF_ANGLES")
        initial = env.mj_data.qpos.copy()
        limited = np.flatnonzero(env.mj_model.jnt_limited)
        bounds = env.mj_model.jnt_range[limited]
        stages = r.plan(r.parse_args(["--full-range"]), r.ready_pose(), r.limits_from_asset())
        for stage in (stages[3], stages[7]):
            for angle in np.linspace(stage.lower, stage.upper, 25):
                target = ready.copy()
                target[env.body_joint_names.index(r.NAMES[stage.index])] = angle
                env.mj_data.qpos[:] = initial
                env._project_closed_loop_default_pose(target)
                values = env.mj_data.qpos[env.mj_model.jnt_qposadr[limited]]
                self.assertTrue(np.all(values >= bounds[:,0]-1e-8))
                self.assertTrue(np.all(values <= bounds[:,1]+1e-8))


@unittest.skipUnless(os.environ.get("H2_SYSID_DDS_TESTS") == "1", "private loopback integration")
class RangeDds(unittest.TestCase):
    def run_case(self, failure, domain):
        import csv
        fixture = subprocess.Popen([sys.executable, str(ROOT / "gear_sonic/tests/test_h2_ankle_sysid_dds.py"),
            "--fixture", "--domain", str(domain), "--failure", failure], stdout=subprocess.PIPE, text=True)
        try:
            while fixture.stdout.readline().strip() != "FIXTURE_READY":
                if fixture.poll() is not None:
                    self.fail("fixture failed")
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "run"
                result = subprocess.run([sys.executable, str(ROOT / "scripts/h2_ankle_range_test.py"),
                    "--target", "sim", "--interface", "lo", "--domain", str(domain), "--side", "both",
                    "--range-deg", "-22", "-20", "--execute", "--confirm-supported", "--auto",
                    "--output", str(out)], capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                meta = json.loads((out / "metadata.json").read_text())
                self.assertEqual([s["side"] for s in meta["completed_stages"]], ["left","right"])
                phases = set()
                with (out / "commands.csv").open() as f:
                    for row in csv.DictReader(f):
                        phase = row["phase"]
                        phases.add(phase)
                        if phase.startswith("range_") or phase == "waiting":
                            active = 5 if phase.startswith("range_left") else 11 if phase.startswith("range_right") else -1
                            for i in range(31):
                                if i != active:
                                    self.assertEqual(float(row[f"q_des_{i}"]), meta["ready_q"][i])
                self.assertIn("prepare", phases)
                self.assertIn("damping", phases)
                if failure == "elbow_bias":
                    with (out / "events.jsonl").open() as f:
                        settled = next(e for line in f if (e := json.loads(line))["event"] == "ready_reached")
                    for i in (18, 25):
                        self.assertGreater(settled["position_error"][i], .10)
                        self.assertLess(settled["position_error"][i], .15)
                import analyze_h2_ankle_range
                report = analyze_h2_ankle_range.analyze(out)
                self.assertEqual(len(report["stages"]), 2)
        finally:
            fixture.terminate()
            fixture.wait(timeout=5)
            fixture.stdout.close()

    def test_prepare_left_then_right_and_analysis(self):
        self.run_case("none", 97)

    def test_static_elbow_bias_does_not_block_scans_or_returns(self):
        self.run_case("elbow_bias", 99)


if __name__ == "__main__":
    unittest.main()
