from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from gear_sonic.utils.mujoco_sim.ankle_response import (
    AnkleResponse, BristleFriction, BristleParameters, MODEL_TEXT_KEY, model_response_config,
)


PARAMETERS = BristleParameters(2.5, 1.1, .1, 900., 8.)


def test_model(config=None):
    root = ET.fromstring('<mujoco><option timestep=".001"/><worldbody/></mujoco>')
    for name in ("left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_ankle_roll_joint"):
        body = ET.SubElement(root.find("worldbody"), "body")
        ET.SubElement(body, "joint", name=name)
        ET.SubElement(body, "geom", size=".1")
    if config is not None:
        ET.SubElement(ET.SubElement(root, "custom"), "text", name=MODEL_TEXT_KEY,
                      data=json.dumps(config))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class BristleModelTests(unittest.TestCase):
    def test_invalid_parameters(self):
        for change in (dict(static_nm=-1), dict(dynamic_nm=3), dict(velocity_rad_s=0),
                       dict(stiffness_nm_rad=float("nan")), dict(position_slope_nm_rad2=-1)):
            with self.assertRaises(ValueError):
                BristleParameters(**(asdict(PARAMETERS) | change))

    def test_constant_velocity_integration_is_step_independent(self):
        def run(dt):
            f = BristleFriction(PARAMETERS)
            for _ in range(round(.1/dt)):
                force = f.force(-.363, .04, dt)
            return force
        self.assertAlmostEqual(run(.001), run(.0005), places=12)
        self.assertGreater(run(.001), 0)
        self.assertLess(run(.001), PARAMETERS.static_nm)

    def test_zero_velocity_retains_presliding_state_and_reset_clears(self):
        f = BristleFriction(PARAMETERS)
        before = f.force(-.363, .001, .001)
        self.assertGreater(before, 0)
        self.assertEqual(f.force(-.363, 0, 1), before)
        f.reset()
        self.assertEqual(f.force(-.363, 0, .001), 0)

    def test_cycle_is_bounded_and_net_dissipative(self):
        f = BristleFriction(PARAMETERS)
        dt, work = .001, 0.
        for t in np.arange(0, 8, dt):
            q = -.363 + .1*np.sin(2*np.pi*t)
            velocity = .2*np.pi*np.cos(2*np.pi*t)
            force = f.force(q, velocity, dt)
            work += force*velocity*dt
            self.assertLessEqual(abs(force), 2.5+8*.1**2+1e-10)
        self.assertGreater(work, 0)

    def test_embedded_metadata_and_explicit_disable(self):
        config = {"left": asdict(PARAMETERS)}
        m = test_model(config)
        self.assertEqual(model_response_config(m, {}), config)
        self.assertEqual(model_response_config(m, {"ANKLE_PITCH_RESPONSE": {}}), {})
        self.assertEqual(model_response_config(test_model(), {}), {})

    def test_only_selected_pitch_and_no_double_application(self):
        m = test_model()
        d = mujoco.MjData(m)
        d.qpos[:] = -.363
        d.qvel[:] = .02
        d.qfrc_applied[:] = [0, 0, 7]
        response = AnkleResponse(m, {"left": asdict(PARAMETERS)})
        response.apply(m, d)
        self.assertLess(d.qfrc_applied[0], 0)
        np.testing.assert_array_equal(d.qfrc_applied[1:], [0, 7])
        force = d.qfrc_applied.copy()
        response.apply(m, d)
        np.testing.assert_array_equal(force, d.qfrc_applied)
        response.reset()
        response.apply(m, d)
        np.testing.assert_array_equal(force, d.qfrc_applied)

    def test_native_profile_default_is_unchanged(self):
        from gear_sonic.utils.mujoco_sim.configs import BaseConfig
        with patch("gear_sonic.utils.mujoco_sim.configs.resolve_interface", return_value=("lo", "sim")):
            baseline = BaseConfig(wbc_version="h2_native_unitree_ros_loop").load_wbc_yaml()
            candidate = BaseConfig(wbc_version="h2_native_unitree_ros_loop_ankle_sysid").load_wbc_yaml()
        differences = {k for k in baseline if baseline[k] != candidate[k]}
        self.assertEqual(differences, {"ROBOT_SCENE", "VERSION"})
        self.assertEqual(candidate["SIMULATE_DT"], .001)

    def test_generated_mjcf_preserves_official_physical_parameters(self):
        from build_h2_ankle_sysid_model import build, SOURCE
        before = SOURCE.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            fit = path/"fit.json"
            fit.write_text(json.dumps({"parameters": asdict(PARAMETERS)}))
            build(fit, fit, path/"model")
            original = mujoco.MjModel.from_xml_path(str(SOURCE))
            derived = mujoco.MjModel.from_xml_path(str(path/"model/H2_loop_ankle_sysid.xml"))
            for name in ("body_mass", "body_inertia", "dof_damping", "dof_frictionloss", "dof_armature",
                         "eq_data", "jnt_range", "actuator_ctrlrange", "geom_friction", "mesh_vert"):
                np.testing.assert_array_equal(getattr(original, name), getattr(derived, name))
            self.assertEqual(set(model_response_config(derived, {})), {"left", "right"})
        self.assertEqual(SOURCE.read_bytes(), before)

    def test_full_closed_loop_fast_path_matches_runtime_with_friction(self):
        from sim_h2_ankle_sysid import FixtureEnv, make_config, NAMES
        from analyze_h2_ankle_friction_hypothesis import simulate
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from types import SimpleNamespace
        config = make_config(73)
        config["ANKLE_PITCH_RESPONSE"] = {"left": asdict(PARAMETERS)}
        env = FixtureEnv(config, onscreen=False)
        initial = env.mj_data.qpos.copy()
        command = unitree_hg_msg_dds__LowCmd_()
        order = np.array([env.body_joint_names.index(n) for n in NAMES])
        env.set_unitree_bridge(SimpleNamespace(low_cmd=command, num_body_motor=31, num_hand_motor=0,
            motor_to_sim_joint_index=order, joystick=None, use_sensor=False,
            body_low_cmd_available=lambda: True, body_low_cmd_is_fresh=lambda: True,
            PublishLowState=lambda obs: None))
        q = env.mj_data.qpos[env.body_qpos_index][order].copy()
        q[5] += .05
        values = np.array([q, np.zeros(31), np.full(31, 40), np.full(31, 2), np.zeros(31)])
        for j in range(31):
            motor = command.motor_cmd[j]
            motor.q, motor.dq, motor.kp, motor.kd, motor.tau = values[:, j]
        expected = []
        for k in range(60):
            expected.append([k*.001, env.mj_data.qpos[env.body_qpos_index[5]],
                             env.mj_data.qvel[env.body_dof_index[5]]])
            env.sim_step()
        commands = np.zeros((2, 158))
        commands[:, 0], commands[:, 1], commands[:, 3:] = [0, .06], 1, values.ravel()
        predicted = simulate(env, initial, dict(commands=commands, spec=dict(frequency_hz=1)), 5)
        np.testing.assert_allclose(predicted, expected, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
