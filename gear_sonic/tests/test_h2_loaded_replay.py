"""Offline loaded replay contracts: native order, closure, and actual command inputs."""

import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from identify_h2_loaded_actuators import (
    H2_JOINT_NAMES_DDS, Replay, initialize, make_env, motor_groups, sole_points,
)


class LoadedReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("unitree_sdk2py.core.channel.ChannelFactoryInitialize",
                   side_effect=AssertionError("Offline tests must not initialize DDS")):
            cls.env = make_env()
        cls.order = np.array([H2_JOINT_NAMES_DDS.index(n) for n in cls.env.body_joint_names])
        cls.to_dds = np.argsort(cls.order)
        cls.ready = cls.env.mj_data.qpos[cls.env.body_qpos_index][cls.to_dds].copy()
        cls.ready_full_qpos = cls.env.mj_data.qpos.copy()

    def setUp(self):
        self.env.unitree_bridge = None

    def pose(self):
        q = self.ready.copy()
        q[3] += .09
        q[9] += .04
        q[4] += .025
        q[5] -= .045
        q[10] -= .03
        q[11] += .02
        q[13] += .015
        q[14] -= .03
        q[15:] += np.linspace(-.02, .02, 16)
        dq = np.linspace(-.09, .12, 31)
        xyzw = Rotation.from_euler("xyz", [.04, -.03, .35]).as_quat()
        return q, dq, xyzw[[3, 0, 1, 2]], np.array([.07, -.11, .08])

    def test_reconstructs_all_output_joints_and_closure_velocity(self):
        q, dq, quat, gyro = self.pose()
        p, v = np.array([.2, -.4, 1.4]), np.array([.13, -.04, .025])
        info = initialize(self.env, q, dq, quat, gyro, root_position=p, root_velocity=v)
        m, d = self.env.mj_model, self.env.mj_data
        np.testing.assert_allclose(d.qpos[self.env.body_qpos_index][self.to_dds], q,
                                   atol=1e-8, rtol=0)
        np.testing.assert_array_equal(d.qvel[self.env.body_dof_index][self.to_dds], dq)
        np.testing.assert_array_equal(d.qpos[:3], p)
        np.testing.assert_array_equal(d.qvel[:3], v)
        np.testing.assert_allclose(self.env.prepare_obs()["floating_base_vel_local"][3:],
                                   gyro, atol=1e-12, rtol=0)
        rows = np.flatnonzero(d.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
        self.assertGreater(len(rows), 0)
        self.assertLess(np.max(np.abs(d.efc_pos[rows])), 1e-7)
        self.assertLess(np.linalg.norm(self.env._dense_constraint_jacobian()[rows] @ d.qvel), 1e-6)
        self.assertLess(info["equality_dq_error"], 1e-6)
        hidden = np.setdiff1d(np.arange(m.nv), np.r_[np.arange(6), self.env.body_dof_index])
        self.assertGreater(np.linalg.norm(d.qvel[hidden]), .01)

    def test_reconstruction_matches_independent_sim_pose_kinematics(self):
        env, m, d = self.env, self.env.mj_model, self.env.mj_data
        q, dq, quat, gyro = self.pose()
        mujoco.mj_resetData(m, d)
        d.qpos[:7] = [.25, -.3, 1.2, *quat]
        qsim = q[self.order]
        d.qpos[env.body_qpos_index] = qsim
        env._project_closed_loop_default_pose(qsim)
        # Forward kinematics from a separately constructed full simulation state.
        mujoco.mj_forward(m, d)
        expected_xpos, expected_xmat = d.xpos.copy(), d.xmat.copy()
        expected_qpos = d.qpos.copy()
        initialize(env, q, dq, quat, gyro, root_position=[.25, -.3, 1.2],
                   root_velocity=[.1, .2, -.03])
        # Ball-joint twist around a massless linkage axis is not observed by DDS.
        # Compare the physical output/motor axes and robot-link kinematics.
        indices = np.unique(np.concatenate([np.arange(7), env.body_qpos_index]
                            + [g["motor_qpos"] for g in env.closed_loop_transmissions]))
        bodies = m.jnt_bodyid[env.body_joint_index]
        np.testing.assert_allclose(d.qpos[indices], expected_qpos[indices], atol=1e-8, rtol=0)
        np.testing.assert_allclose(d.xpos[bodies], expected_xpos[bodies], atol=1e-8, rtol=0)
        np.testing.assert_allclose(d.xmat[bodies], expected_xmat[bodies], atol=1e-8, rtol=0)

    def test_deep_squat_remains_on_ready_assembly_branch(self):
        env, m, d = self.env, self.env.mj_model, self.env.mj_data
        q, dq, quat, gyro = self.pose()
        q[[0, 6]] = [-1.05, -.98]
        q[[3, 9]] = [2.0, 1.85]
        q[[5, 11]] = [-.43, -.33]
        qsim = q[self.order]
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self.ready_full_qpos
        start = d.qpos[env.body_qpos_index].copy()
        closed = {int(i) for g in env.closed_loop_transmissions for i in g["output_positions"]}
        direct = np.array([i for i in range(31) if i not in closed])
        # Independent finer continuation provides a physical assembly reference.
        for alpha in np.linspace(0, 1, 51)[1:]:
            target = start + alpha * (qsim - start)
            d.qpos[env.body_qpos_index[direct]] = target[direct]
            env._project_closed_loop_default_pose(target)
        motors = np.concatenate([g["motor_qpos"] for g in env.closed_loop_transmissions])
        expected = d.qpos[motors].copy()
        initialize(env, q, dq, quat, gyro, root_position=[0, 0, 1.2], root_velocity=[0, 0, 0])
        np.testing.assert_allclose(d.qpos[motors], expected, atol=1e-7, rtol=0)
        np.testing.assert_allclose(d.qpos[env.body_qpos_index][self.to_dds], q, atol=1e-8, rtol=0)

    def test_height_and_contact_velocity_are_explicit_hypotheses(self):
        q, dq, quat, gyro = self.pose()
        for mode in ("zero", "left", "right", "both"):
            with self.subTest(mode=mode):
                info = initialize(self.env, q, dq, quat, gyro, velocity=mode, clearance=.001)
                self.assertAlmostEqual(min(info["foot_bottom_z"]), .001, places=10)
                points = sole_points(self.env)
                self.assertAlmostEqual(min(p[2] for _, p in points), .001, places=10)
                if mode == "zero":
                    np.testing.assert_array_equal(info["root_velocity"], [0, 0, 0])
                elif mode in ("left", "right"):
                    self.assertLess(info["assumed_contact_point_speed"][mode == "right"], 1e-10)

    def dataset(self, path, duration=.025):
        q, dq, quat, gyro = self.pose()
        state = np.array([[t, 0, 1, *q, *dq, *quat, *gyro] for t in (0., .001, .1)])
        kp = 20 + np.arange(31)
        kd = .5 + np.arange(31) * .03
        rows = []
        for t, shift in ((0., 0.), (.002, .01), (.011, -.015), (.1, .02)):
            target = q + np.linspace(-.02, .03, 31) + shift
            rows.append([t, 0, 1, *target, *np.linspace(.01, -.01, 31),
                         *kp, *kd, *np.linspace(-.2, .2, 31)])
        command = np.asarray(rows)
        np.savez(path, state=state, command=command, starts=[.001], repeats=[1],
                 labels=["synthetic"], duration=duration)
        with patch("identify_h2_loaded_actuators.make_env", return_value=self.env):
            replay = Replay(path)
        self.addCleanup(replay.a.close)
        return replay, state, command

    def test_dataset_shape_native_order_and_causal_zoh(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay, state, command = self.dataset(Path(tmp) / "dataset.npz")
            window = replay.windows[0]
            self.assertEqual(window["target"].shape, (25, 5, 31))
            self.assertEqual(window["measured"].shape, (5, 62))
            self.assertEqual(window["pitch"].shape, (5,))
            expected = command[[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], 3:].reshape(-1, 5, 31)
            np.testing.assert_array_equal(window["target"][:11], expected[:, :, self.order])
            np.testing.assert_array_equal(window["initial"], state[1])
            np.testing.assert_array_equal(window["measured"][:, :31], np.repeat(state[:1, 3:34], 5, axis=0))

    def test_measured_targets_are_not_injected_during_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay, _, _ = self.dataset(Path(tmp) / "dataset.npz", duration=.04)
            window = replay.windows[0]
            first = replay.run({}, window)
            altered = copy.deepcopy(window)
            altered["measured"] += .4
            altered["pitch"] += .2
            second = replay.run({}, altered)
            np.testing.assert_array_equal(first[2], second[2])
            np.testing.assert_array_equal(first[4], second[4])
            self.assertNotEqual(first[0]["score"], second[0]["score"])

    def test_reconstruction_of_dynamic_sim_snapshot_preserves_observables(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay, _, _ = self.dataset(Path(tmp) / "dataset.npz", duration=.04)
            replay.run({}, replay.windows[0])
            env, m, d = self.env, self.env.mj_model, self.env.mj_data
            mujoco.mj_forward(m, d)
            q = d.qpos[env.body_qpos_index][self.to_dds].copy()
            dq = d.qvel[env.body_dof_index][self.to_dds].copy()
            position, quat, velocity = d.qpos[:3].copy(), d.qpos[3:7].copy(), d.qvel[:3].copy()
            gyro = env.prepare_obs()["floating_base_vel_local"][3:].copy()
            bodies = m.jnt_bodyid[env.body_joint_index]
            positions, orientations = d.xpos[bodies].copy(), d.xmat[bodies].copy()
            initialize(env, q, dq, quat, gyro, root_position=position, root_velocity=velocity)
            np.testing.assert_allclose(d.qpos[env.body_qpos_index][self.to_dds], q, atol=1e-8, rtol=0)
            np.testing.assert_array_equal(d.qvel[env.body_dof_index][self.to_dds], dq)
            np.testing.assert_allclose(d.xpos[bodies], positions, atol=1e-8, rtol=0)
            np.testing.assert_allclose(d.xmat[bodies], orientations, atol=1e-8, rtol=0)

    def test_physics_warning_or_clock_reset_invalidates_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay, _, _ = self.dataset(Path(tmp) / "dataset.npz")
            original = mujoco.mj_step
            for failure in ("warning", "reset"):
                with self.subTest(failure=failure):
                    def bad_step(model, data):
                        original(model, data)
                        if failure == "warning":
                            data.warning[0].number += 1
                        else:
                            data.time = 0
                    with patch("identify_h2_loaded_actuators.mujoco.mj_step", side_effect=bad_step):
                        with self.assertRaisesRegex(ValueError, "simulation warning/reset"):
                            replay.run({}, replay.windows[0])

    def test_empty_motor_candidate_matches_standard_sim_step(self):
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        with tempfile.TemporaryDirectory() as tmp:
            replay, _, _ = self.dataset(Path(tmp) / "dataset.npz", duration=.04)
            window = replay.windows[0]
            actual = replay.run({}, window)
            expected_final_qpos = self.env.mj_data.qpos.copy()
            expected_final_qvel = self.env.mj_data.qvel.copy()
            row = window["initial"]
            initialize(self.env, row[3:34], row[34:65], row[65:69], row[69:72])
            cmd = unitree_hg_msg_dds__LowCmd_()
            bridge = SimpleNamespace(low_cmd=cmd, num_body_motor=31, num_hand_motor=0,
                motor_to_sim_joint_index=self.to_dds, joystick=None, use_sensor=False,
                body_low_cmd_available=lambda: True, body_low_cmd_is_fresh=lambda: True,
                PublishLowState=lambda obs: None)
            self.env.set_unitree_bridge(bridge)
            output = []
            for k, fields in enumerate(window["target"]):
                if k % 5 == 0:
                    d = self.env.mj_data
                    pitch = Rotation.from_quat(d.qpos[[4, 5, 6, 3]]).as_euler("xyz")[1]
                    output.append(np.r_[d.qpos[self.env.body_qpos_index][self.to_dds],
                                        d.qvel[self.env.body_dof_index][self.to_dds], pitch])
                for i, values in enumerate(fields[:, self.to_dds].T):
                    motor = cmd.motor_cmd[i]
                    motor.q, motor.dq, motor.kp, motor.kd, motor.tau = values
                self.env.sim_step()
            np.testing.assert_array_equal(actual[2], output)
            np.testing.assert_array_equal(self.env.mj_data.qpos, expected_final_qpos)
            np.testing.assert_array_equal(self.env.mj_data.qvel, expected_final_qvel)


if __name__ == "__main__":
    unittest.main()
