from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from sim_h2_ankle_sysid import apply_pitch_response, FixtureEnv, make_config
from analyze_h2_ankle_friction_hypothesis import metrics, simulate


class PitchFriction(unittest.TestCase):
    def setUp(self):
        xml = '<mujoco><default><joint frictionloss=".2"/></default><worldbody>'
        for name in ("left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_ankle_roll_joint"):
            xml += f'<body><joint name="{name}"/><geom size=".1"/></body>'
        xml += '</worldbody></mujoco>'
        self.env = SimpleNamespace(mj_model=mujoco.MjModel.from_xml_string(xml), sim_dt=.001)

    def test_default_is_noop(self):
        m = self.env.mj_model
        original = [x.copy() for x in (m.dof_frictionloss, m.dof_solref, m.dof_solimp)]
        self.assertEqual(apply_pitch_response(self.env, {}), {})
        for before, after in zip(original, (m.dof_frictionloss, m.dof_solref, m.dof_solimp)):
            np.testing.assert_array_equal(before, after)

    def test_only_selected_pitch_changes(self):
        apply_pitch_response(self.env, {"right": {"frictionloss_add_nm": 2., "hard_friction": True}})
        np.testing.assert_allclose(self.env.mj_model.dof_frictionloss, [.2, 2.2, .2])
        np.testing.assert_allclose(self.env.mj_model.dof_solref[:, 0], [.02, .002, .02])

    def test_validates_all_before_mutation(self):
        for parameters in ({"left": 1, "right": -1}, {"left": float("nan")},
                           {"both": 1}, {"left": {"frictionloss_add_nm": 1, "hard_friction": "false"}},
                           {"left": {"frictionloss_add_nm": 1, "bad": 2}}):
            with self.assertRaises(ValueError):
                apply_pitch_response(self.env, parameters)
            np.testing.assert_allclose(self.env.mj_model.dof_frictionloss, .2)

    def test_frequency_metric_uses_actual_command(self):
        t = np.arange(0, 8, .002)
        c = np.zeros((len(t), 3+155))
        c[:, 0], c[:, 1], c[:, 2] = t, 1, .5
        c[:, 3+5] = .1*np.sin(np.pi*t)
        y = .07*np.sin(np.pi*t-.3)
        trial = dict(commands=c, spec=dict(frequency_hz=.5))
        result = metrics(trial, np.column_stack((t, y, .07*np.pi*np.cos(np.pi*t-.3))), 5)
        self.assertAlmostEqual(result["gain"], .7, places=6)
        self.assertAlmostEqual(result["phase_deg"], np.rad2deg(-.3), places=6)

    def test_fast_replay_matches_existing_pd_and_physics_step(self):
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import H2_JOINT_NAMES_DDS
        env = FixtureEnv(make_config(73), onscreen=False)
        initial = env.mj_data.qpos.copy()
        cmd = unitree_hg_msg_dds__LowCmd_()
        order = np.array([env.body_joint_names.index(n) for n in H2_JOINT_NAMES_DDS])
        env.set_unitree_bridge(SimpleNamespace(low_cmd=cmd, num_body_motor=31, num_hand_motor=0,
            motor_to_sim_joint_index=order, joystick=None, use_sensor=False,
            body_low_cmd_available=lambda: True, body_low_cmd_is_fresh=lambda: True,
            PublishLowState=lambda obs: None))
        q = env.mj_data.qpos[env.body_qpos_index][order].copy()
        q[5] += .01
        values = np.array([q, np.zeros(31), np.full(31, 40), np.full(31, 2), np.zeros(31)])
        for i in range(31):
            cmd.motor_cmd[i].q, cmd.motor_cmd[i].dq, cmd.motor_cmd[i].kp, cmd.motor_cmd[i].kd, cmd.motor_cmd[i].tau = values[:, i]
        expected = []
        for k in range(20):
            expected.append([k*.001, env.mj_data.qpos[env.body_qpos_index[5]], env.mj_data.qvel[env.body_dof_index[5]]])
            env.sim_step()
        commands = np.zeros((2, 158))
        commands[:, 0], commands[:, 1], commands[:, 3:] = [0, .02], 1, values.ravel()
        trial = dict(commands=commands, spec=dict(frequency_hz=1))
        predicted = simulate(env, initial, trial, 5)
        np.testing.assert_allclose(predicted, expected, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
