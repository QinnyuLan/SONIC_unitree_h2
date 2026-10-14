import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from gear_sonic.utils.mujoco_sim.motor_response import MotorResponse

class MotorResponseTests(unittest.TestCase):
    def make(self, config=None):
        return MotorResponse(np.array([10., 20., 30., 40.]),
                             {"ankle": [0, 1], "knee": [2]},
                             {} if config is None else config)

    def test_empty_configuration_is_bit_identical_copy(self):
        for dtype in (np.float32, np.float64, np.int32):
            requested = np.array([-0., 999., -999., 7.], dtype=dtype)
            response = self.make()
            result = response.apply(requested, .001)
            self.assertEqual(result.dtype, requested.dtype)
            self.assertEqual(result.tobytes(), requested.tobytes())
            self.assertFalse(np.shares_memory(result, requested))
            self.assertFalse(response.clipped.any())

    def test_constant_input_matches_analytic_response(self):
        response = self.make({"ankle": {"time_constant_ms": 20, "gain": .8}})
        requested = np.array([4., -8., 1., 100.])
        for step in range(1, 51):
            actual = response.apply(requested, .001)
            expected = .8 * requested[:2] * (1 - np.exp(-step * .001 / .02))
            np.testing.assert_allclose(actual[:2], expected, rtol=1e-14, atol=1e-14)
            np.testing.assert_array_equal(actual[2:], requested[2:])

    def test_dt_partition_invariance(self):
        config = {"ankle": {"time_constant_ms": 35}}
        once, split = self.make(config), self.make(config)
        requested = np.array([3., -4., 9., 7.])
        expected = once.apply(requested, .04)
        for dt in (.001, .002, .007, .03):
            result = split.apply(requested, dt)
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)

    def test_gain_cap_and_group_isolation(self):
        response = self.make({"ankle": {"gain": 1.2, "limit_scale": .25}})
        actual = response.apply(np.array([10., -10., 100., -100.]), .001)
        np.testing.assert_array_equal(actual, [2.5, -5., 100., -100.])
        np.testing.assert_array_equal(response.last_prelimit, [12., -12., 100., -100.])
        np.testing.assert_array_equal(response.last_applied, actual)
        np.testing.assert_array_equal(response.clipped, [True, True, False, False])

    def test_selected_defaults_have_native_cap(self):
        response = self.make({"knee": {}})
        np.testing.assert_array_equal(response.apply(np.array([100., 100., 100., 100.]), .001),
                                      [100., 100., 30., 100.])

    def test_reset_clears_lag_and_initial_sets_effort_without_gain(self):
        response = self.make({"ankle": {"gain": .5, "time_constant_ms": 50}})
        request = np.full(4, 4.)
        first = response.apply(request, .01)
        response.apply(request, .01)
        response.reset()
        np.testing.assert_array_equal(response.apply(request, .01), first)
        initial = np.array([2., 2., 7., 8.])
        response.reset(initial)
        initial[:] = 999.
        np.testing.assert_allclose(response.apply(request, .01)[:2], [2., 2.], atol=0)

    def test_saturation_does_not_modify_prelimit_state(self):
        response = self.make({"ankle": {"time_constant_ms": 10, "limit_scale": .25}})
        response.reset(np.array([10., -10., 0., 0.]))
        np.testing.assert_array_equal(response.last_applied[:2], [2.5, -5.])
        response.apply(np.zeros(4), .001)
        np.testing.assert_allclose(response.last_prelimit[:2], [10 * np.exp(-.1), -10 * np.exp(-.1)])

    def test_inputs_and_diagnostics_are_not_aliased(self):
        limits = np.full(4, 10.)
        groups = {"left": [0, 1]}
        config = {"left": {"gain": .5}}
        response = MotorResponse(limits, groups, config)
        limits[:] = 999
        groups["left"][0] = 3
        config["left"]["gain"] = 1.2
        request = np.array([2., -2., 3., 4.])
        original = request.copy()
        result = response.apply(request, .001)
        np.testing.assert_array_equal(request, original)
        np.testing.assert_array_equal(result, [1., -1., 3., 4.])
        result[:] = 999
        np.testing.assert_array_equal(response.last_applied, [1., -1., 3., 4.])

    def test_invalid_limits(self):
        for limits in ([], [[1]], [0], [-1], [np.inf], [np.nan], [1j], [True], ["1"]):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                MotorResponse(limits, {}, {})

    def test_invalid_groups_and_unknown_configuration(self):
        for groups in ({"a": [0, 0]}, {"a": [0], "b": [0]}, {"a": [-1]}, {"a": [4]},
                       {"a": [True]}, {"a": [1.]}, {"a": [[0]]}, {"a": []},
                       {"a": None}, {"a": np.array(0)}, {"a": np.array([[0]])},
                       {0: [0]}, {"": [0]}, []):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                MotorResponse(np.ones(4), groups, {})
        with self.assertRaises(ValueError):
            self.make({"unknown": {}})

    def test_invalid_parameters(self):
        for field, value in (("gain", .39), ("gain", 1.21), ("time_constant_ms", -1),
                             ("time_constant_ms", 51), ("limit_scale", .24),
                             ("limit_scale", 1.01), ("gain", np.nan), ("gain", np.inf),
                             ("gain", True), ("gain", "1"), ("gain", [1]), ("typo", 1)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.make({"ankle": {field: value}})
        for config in ([], None, {"ankle": None}, {"ankle": []}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                MotorResponse(np.ones(4), {"ankle": [0, 1]}, config)

    def test_invalid_input_does_not_advance_state_or_diagnostics(self):
        response = self.make({"ankle": {"time_constant_ms": 10}})
        request = np.array([1., 2., 3., 4.])
        response.apply(request, .001)
        before = [getattr(response, name).copy()
                  for name in ("_state", "last_prelimit", "last_applied", "clipped")]
        bad_requests = ([1], [[1, 2, 3, 4]], [1, 2, np.nan, 4], [1, 2, np.inf, 4],
                        [1, 2, 3, 4j], ["1", "2", "3", "4"], [True] * 4)
        calls = [(value, .001) for value in bad_requests]
        calls += [(request, value) for value in (0, -1, np.inf, np.nan, True, "1", [1])]
        for value, dt in calls:
            with self.subTest(value=value, dt=dt), self.assertRaises(ValueError):
                response.apply(value, dt)
            for name, expected in zip(("_state", "last_prelimit", "last_applied", "clipped"), before):
                np.testing.assert_array_equal(getattr(response, name), expected)
        with self.assertRaises(ValueError):
            response.reset(np.array([np.nan, 0, 0, 0]))
        np.testing.assert_array_equal(response._state, before[0])

    def test_overflow_rejected_without_advancing(self):
        response = self.make({"ankle": {"gain": 1.2}})
        before = response._state.copy()
        with self.assertRaises(ValueError):
            response.apply(np.full(4, np.finfo(float).max), .001)
        np.testing.assert_array_equal(response._state, before)

    def test_cached_scalar_response_is_bit_identical_to_vector_formula(self):
        response = self.make({"ankle": {"gain": .8, "time_constant_ms": 21, "limit_scale": .7},
                              "knee": {"gain": 1.1, "time_constant_ms": 0}})
        rng = np.random.default_rng(21)
        initial = rng.normal(size=4)
        response.reset(initial)
        state = initial.copy()
        for step in range(300):
            dt = (.001, .001, .001, .002, .0005)[step % 5]
            request = rng.normal(size=4) * 100
            prelimit = request.copy()
            selected = response.selected
            target = prelimit[selected] * response.gain[selected]
            tau = response.time_constant_s[selected]
            alpha = np.ones(tau.size)
            lagged = tau > 0
            alpha[lagged] = -np.expm1(-dt / tau[lagged])
            prelimit[selected] = ((1 - alpha) * state[selected] + alpha * target)
            expected = prelimit.copy()
            expected[selected] = np.clip(prelimit[selected], -response.limits[selected],
                                          response.limits[selected])
            actual = response.apply(request, dt)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(response.last_prelimit, prelimit)
            np.testing.assert_array_equal(response.clipped, expected != prelimit)
            state = prelimit


class MotorResponseIntegrationTests(unittest.TestCase):
    def make_env(self, response=None, directory=None):
        from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
        from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
        from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import H2_JOINT_NAMES_DDS
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

        with patch("gear_sonic.utils.mujoco_sim.configs.resolve_interface", return_value=("lo", "sim")):
            config = SimLoopConfig(wbc_version="h2_native_unitree_ros_loop").load_wbc_yaml()
        config.update(ENABLE_ELASTIC_BAND=False, PRINT_SCENE_INFORMATION=False)
        if response is not None:
            config["H2_MOTOR_RESPONSE"] = response
        if directory is not None:
            config["DIAGNOSTICS_DIR"] = str(directory)
        env = DefaultEnv(config, onscreen=False)
        order = np.array([env.body_joint_names.index(n) for n in H2_JOINT_NAMES_DDS])
        command = unitree_hg_msg_dds__LowCmd_()
        q = env.mj_data.qpos[env.body_qpos_index][order]
        for i in range(31):
            command.motor_cmd[i].q = q[i] + .04
            command.motor_cmd[i].kp = 40
            command.motor_cmd[i].kd = 2
        env.set_unitree_bridge(SimpleNamespace(
            low_cmd=command, num_body_motor=31, num_hand_motor=0,
            motor_to_sim_joint_index=order, joystick=None, use_sensor=False,
            body_low_cmd_available=lambda: True, body_low_cmd_is_fresh=lambda: True,
            PublishLowState=lambda obs: None, low_cmd_received=True, low_cmd_last_time=None))
        return env

    def test_missing_and_empty_config_are_bit_identical(self):
        missing, empty = self.make_env(), self.make_env({})
        self.assertIsNone(missing.motor_response)
        self.assertIsNone(empty.motor_response)
        for _ in range(10):
            missing.sim_step()
            empty.sim_step()
            np.testing.assert_array_equal(missing.mj_data.qpos, empty.mj_data.qpos)
            np.testing.assert_array_equal(missing.mj_data.qvel, empty.mj_data.qvel)
            np.testing.assert_array_equal(missing.mj_data.ctrl, empty.mj_data.ctrl)

    def test_actual_motor_groups_and_post_mapping_response(self):
        config = {"left_ankle": {"gain": .8}, "left_knee": {"gain": .7}}
        env = self.make_env(config)
        for name, positions in env.motor_response_groups.items():
            group = next(g for g in env.closed_loop_transmissions if g["name"] == name + "_loop")
            np.testing.assert_array_equal(positions, group["motor_positions"])
        self.assertEqual(len(env.motor_response_groups["left_ankle"]), 2)
        pd = env.compute_body_torques()
        mapped = env._map_body_torques_to_actuators(pd)
        expected = mapped.copy()
        expected[env.motor_response_groups["left_ankle"]] *= .8
        expected[env.motor_response_groups["left_knee"]] *= .7
        expected = np.clip(expected, -env.body_torque_limit, env.body_torque_limit)
        env.sim_step()
        np.testing.assert_allclose(env.mj_data.ctrl[env.body_actuator_index], expected, atol=1e-12)
        untouched = ~env.motor_response.selected
        np.testing.assert_array_equal(env.motor_response.last_applied[untouched], mapped[untouched])

    def test_reset_clears_response_state(self):
        env = self.make_env({"left_knee": {"time_constant_ms": 25}})
        env.sim_step()
        self.assertTrue(np.any(env.motor_response.last_prelimit))
        env.reset()
        np.testing.assert_array_equal(env.motor_response.last_prelimit, np.zeros(env.num_body_dof))
        np.testing.assert_array_equal(env.motor_response.last_applied, np.zeros(env.num_body_dof))
        self.assertFalse(env.motor_response.clipped.any())

    def test_observation_inverse_matches_pinv_and_preserves_500_step_physics(self):
        candidate = {"left_knee": {"gain": .7}}
        old, new = self.make_env(candidate), self.make_env(candidate)

        def legacy_inverse(motor_torques):
            result = motor_torques.copy()
            for group in old.closed_loop_transmissions:
                jacobian = old._closed_loop_output_to_motor_jacobian(group)
                result[group["output_positions"]] = (np.linalg.pinv(jacobian.T)
                                                      @ motor_torques[group["motor_positions"]])
            return result

        old._map_actuator_torques_to_body = legacy_inverse
        for _ in range(500):
            old.sim_step()
            new.sim_step()
            np.testing.assert_array_equal(old.mj_data.qpos, new.mj_data.qpos)
            np.testing.assert_array_equal(old.mj_data.qvel, new.mj_data.qvel)
            np.testing.assert_array_equal(old.mj_data.ctrl, new.mj_data.ctrl)
            np.testing.assert_allclose(old.obs["body_tau_est"], new.obs["body_tau_est"],
                                       atol=1e-10, rtol=1e-12)

    def test_gain_change_is_not_recorded_as_native_clipping(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "log"
            env = self.make_env({"left_knee": {"gain": .7}}, directory)
            env.sim_step()
            env.diagnostics.close()
            with (directory / "truth.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertNotEqual(float(row["motor_requested_3"]), float(row["motor_response_requested_3"]))
            self.assertAlmostEqual(float(row["motor_response_requested_3"]),
                                   .7 * float(row["motor_requested_3"]), places=12)
            self.assertEqual(float(row["motor_response_clipped_3"]), 0)
            self.assertEqual(float(row["motor_clipped_3"]), 0)
            self.assertAlmostEqual(float(row["motor_ctrl_clipped_3"]),
                                   float(row["motor_response_requested_3"]), places=12)
            metadata = json.loads((directory / "metadata.json").read_text())
            self.assertEqual(metadata["motor_response_candidate"]["config"], {"left_knee": {"gain": .7}})

    def test_candidate_cap_recorded_separately_from_native_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "log"
            env = self.make_env({"left_knee": {"limit_scale": .25}}, directory)
            env.unitree_bridge.low_cmd.motor_cmd[3].q += 100
            env.sim_step()
            env.diagnostics.close()
            with (directory / "truth.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(float(row["motor_response_clipped_3"]), 1)
            self.assertEqual(float(row["motor_clipped_3"]), 0)
            self.assertGreater(abs(float(row["motor_response_prelimit_3"])),
                               abs(float(row["motor_response_requested_3"])))



if __name__ == "__main__":
    unittest.main()
