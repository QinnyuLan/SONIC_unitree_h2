import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gear_sonic.utils.diagnostic_log import DiagnosticLog
from gear_sonic.utils.mujoco_sim.squat_diagnostics import ground_wrench
from h2_squat_compare import launch_args
from record_h2_squat_dds import HEADERS, create_reader, message_row


class DiagnosticTests(unittest.TestCase):
    def test_reader_is_initialized_before_callbacks_and_drains_pending_data(self):
        from cyclonedds.core import Listener

        events = []
        listener = Listener(on_data_available=lambda reader: events.append(('receive', reader.initialized)),
                            on_subscription_matched=lambda reader, status: events.append(('matched', reader)))

        class StartupReader:
            def __init__(self, participant, topic, qos=None, listener=None):
                self.initialized = False
                if listener is not None:
                    listener.on_data_available(None)
                    listener.on_subscription_matched(None, None)
                events.append(('construct', self.initialized))
                self.initialized = True

            def set_listener(self, attached):
                events.append(('attach', self.initialized))
                attached.on_data_available(self)

        with patch('cyclonedds.sub.DataReader', StartupReader):
            reader = create_reader(None, None, None, listener)
        self.assertTrue(reader.initialized)
        self.assertEqual(events, [('matched', None), ('construct', False), ('attach', True),
                                  ('receive', True), ('receive', True)])

    def test_capture_failure_marks_flushed_log_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = DiagnosticLog(Path(tmp)/'log', {'x': ['a']}, {})
            log.put('x', [1])
            log.close(incomplete=True)
            meta = json.loads((log.directory/'metadata.json').read_text())
            self.assertEqual(meta['status'], 'incomplete')
            self.assertEqual(meta['rows']['x'], 1)

    def test_csv_flush_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"log"
            log = DiagnosticLog(path, {"x": ["a", "b"]}, {"source": "test"})
            log.put("x", [1, 2])
            log.close()
            log.close()
            self.assertEqual(json.loads((path/"metadata.json").read_text())['rows']['x'], 1)
            with (path/"x.csv").open() as f:
                self.assertEqual(list(csv.reader(f)), [["a", "b"], ["1", "2"]])
            with self.assertRaises(FileExistsError):
                DiagnosticLog(path, {}, {})

    def test_bad_row_is_explicitly_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = DiagnosticLog(Path(tmp)/"log", {"x": ["a"]}, {})
            log.put("x", [1, 2])
            log.close()
            self.assertIsNotNone(log.error)
            self.assertEqual(json.loads((log.directory/"metadata.json").read_text())['status'], "incomplete")

    def test_ground_wrench_and_cop_sign(self):
        m = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".001"/>
          <worldbody><geom type="plane" size="0 0 .1"/>
          <body name="foot" pos=".2 .1 .05"><freejoint/>
          <geom type="box" size=".04 .03 .05" mass="2"/></body></worldbody></mujoco>''')
        d = mujoco.MjData(m)
        for _ in range(500):
            mujoco.mj_step(m, d)
        n, force, moment, valid, cop = ground_wrench(m, d, {m.body("foot").id})
        self.assertGreater(n, 0)
        self.assertTrue(valid)
        self.assertAlmostEqual(force[2], 2*9.81, delta=.1)
        np.testing.assert_allclose(cop, [.2, .1, 0], atol=.001)
        d.qpos[2] = 1
        mujoco.mj_forward(m, d)
        n, force, moment, valid, cop = ground_wrench(m, d, {m.body("foot").id})
        self.assertFalse(valid)
        self.assertTrue(np.isnan(cop).all())
        np.testing.assert_array_equal(force, [0, 0, 0])

    def test_message_native_order_crc_and_headers(self):
        from unitree_sdk2py.idl.default import (unitree_hg_msg_dds__LowState_,
            unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__IMUState_, unitree_hg_msg_dds__OdoState_)
        from unitree_sdk2py.utils.crc import CRC
        crc = CRC()
        for name, factory in (("lowstate", unitree_hg_msg_dds__LowState_),
                               ("lowcmd", unitree_hg_msg_dds__LowCmd_),
                               ("secondary_imu", unitree_hg_msg_dds__IMUState_),
                               ("odometry_unverified", unitree_hg_msg_dds__OdoState_)):
            msg = factory()
            if name == 'lowstate':
                for i in range(31):
                    msg.motor_state[i].q = i / 10
                msg.crc = crc.Crc(msg)
            if name == 'lowcmd':
                for i in range(31):
                    msg.motor_cmd[i].q = i / 10
                msg.crc = crc.Crc(msg)
            row = message_row(name, msg, crc, clock=123)
            self.assertEqual(len(row), len(HEADERS[name]), name)
            values = dict(zip(HEADERS[name], row))
            self.assertEqual(values['time_monotonic_ms'], 123)
            if name in ('lowcmd', 'lowstate'):
                self.assertEqual(values['crc_ok'], 1)
                self.assertEqual(values['q_29' if name == 'lowstate' else 'q_target_29'], 2.9)

    def test_no_empty_hand_publications_without_hand_motors(self):
        from unittest.mock import Mock
        from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge
        from unitree_sdk2py.idl.default import (unitree_hg_msg_dds__LowState_,
            unitree_hg_msg_dds__IMUState_, unitree_hg_msg_dds__OdoState_)
        bridge = UnitreeSdk2Bridge.__new__(UnitreeSdk2Bridge)
        bridge.use_sensor = False
        bridge.num_body_motor = bridge.num_hand_motor = 0
        bridge.have_frame_sensor_ = False
        bridge.use_local_base_gyro = True
        bridge.pelvis_imu_pitch_bias_rad = 0
        bridge.low_state = unitree_hg_msg_dds__LowState_()
        bridge.odo_state = unitree_hg_msg_dds__OdoState_()
        bridge.torso_imu_state = unitree_hg_msg_dds__IMUState_()
        for name in ('low_state', 'odo_state', 'torso_imu', 'left_hand_state', 'right_hand_state'):
            setattr(bridge, name+'_puber', Mock())
        bridge.PublishLowState(dict(floating_base_pose=np.array([0, 0, 1, 1, 0, 0, 0]),
            floating_base_vel=np.zeros(6), floating_base_vel_local=np.zeros(6),
            floating_base_acc=np.zeros(6), secondary_imu_quat=np.array([1, 0, 0, 0]),
            secondary_imu_vel=np.zeros(6), time=.1))
        for name in ('low_state', 'odo_state', 'torso_imu'):
            getattr(bridge, name+'_puber').Write.assert_called_once()
        bridge.left_hand_state_puber.Write.assert_not_called()
        bridge.right_hand_state_puber.Write.assert_not_called()

    def test_launch_options_and_no_inherited_automatic_control(self):
        for target, interface in [('sim', 'lo'), ('real', 'eno1')]:
            args = argparse.Namespace(command='deploy', run='test', target=target,
                                      interface=interface, variant='identified')
            with patch.dict('os.environ', {'GEAR_SONIC_AUTO_START_AFTER_SEC': '0',
                                          'GEAR_SONIC_SKIP_MOTION_SWITCHER': '1'}):
                path, env, cmd = launch_args(args)
            self.assertNotIn('GEAR_SONIC_AUTO_START_AFTER_SEC', env)
            self.assertNotIn('GEAR_SONIC_SKIP_MOTION_SWITCHER', env)
            self.assertIn('--no-planner', cmd)
            self.assertIn('keyboard', cmd)
            self.assertTrue(any(s.endswith('/model_encoder_full.onnx') for s in cmd))
            self.assertNotIn('--disable-crc-check', cmd)
            self.assertEqual(cmd.count('1.0'), 5)
            self.assertEqual(env['GEAR_SONIC_INITIAL_MOTION_NAME'], 'squat_001__A359')

    def test_instrumentation_does_not_change_physics(self):
        from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
        from gear_sonic.utils.mujoco_sim.configs import BaseConfig
        from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import H2_JOINT_NAMES_DDS
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        with patch('gear_sonic.utils.mujoco_sim.configs.resolve_interface', return_value=('lo', 'sim')):
            config = BaseConfig(wbc_version='h2_native_unitree_ros_loop_ankle_sysid').load_wbc_yaml()
        config.update(ENABLE_ELASTIC_BAND=False, PRINT_SCENE_INFORMATION=False)
        with tempfile.TemporaryDirectory() as tmp:
            plain = DefaultEnv(dict(config), onscreen=False)
            logged = DefaultEnv(dict(config, DIAGNOSTICS_DIR=str(Path(tmp)/'log')), onscreen=False)
            # Retain the former sparse extraction as an independent baseline.
            def legacy_block(rows, dofs, **kwargs):
                block = np.zeros((len(rows), len(dofs)))
                columns = {int(dof): i for i, dof in enumerate(dofs)}
                d = plain.mj_data
                for i, row in enumerate(rows):
                    start = int(d.efc_J_rowadr[row])
                    for k in range(start, start + int(d.efc_J_rownnz[row])):
                        column = columns.get(int(d.efc_J_colind[k]))
                        if column is not None:
                            block[i, column] = d.efc_J[k]
                return block
            plain._constraint_jacobian_block = legacy_block
            def legacy_pd():
                values = np.zeros(plain.num_body_dof)
                bridge = plain.unitree_bridge
                for motor_i in range(bridge.num_body_motor):
                    sim_i = int(bridge.motor_to_sim_joint_index[motor_i])
                    c = bridge.low_cmd.motor_cmd[motor_i]
                    q = plain.mj_data.qpos[plain.body_qpos_index[sim_i]]
                    dq = plain.mj_data.qvel[plain.body_dof_index[sim_i]]
                    values[sim_i] = c.tau + c.kp * (c.q - q) + c.kd * (c.dq - dq)
                return values
            plain.compute_body_torques = legacy_pd
            for env in (plain, logged):
                order = np.array([env.body_joint_names.index(n) for n in H2_JOINT_NAMES_DDS])
                command = unitree_hg_msg_dds__LowCmd_()
                q = env.mj_data.qpos[env.body_qpos_index][order]
                for j in range(31):
                    command.motor_cmd[j].q = q[j]
                    command.motor_cmd[j].kp = 40
                    command.motor_cmd[j].kd = 2
                env.set_unitree_bridge(SimpleNamespace(low_cmd=command, num_body_motor=31,
                    num_hand_motor=0, motor_to_sim_joint_index=order, joystick=None, use_sensor=False,
                    body_low_cmd_available=lambda: True, body_low_cmd_is_fresh=lambda: True,
                    PublishLowState=lambda obs: None, low_cmd_received=True, low_cmd_last_time=None))
            before = logged.mj_data.qpos[logged.body_qpos_index][order].copy()
            for _ in range(500):
                plain.sim_step()
                logged.sim_step()
                np.testing.assert_array_equal(plain.mj_data.qpos, logged.mj_data.qpos)
                np.testing.assert_array_equal(plain.mj_data.qvel, logged.mj_data.qvel)
                np.testing.assert_array_equal(plain.mj_data.ctrl, logged.mj_data.ctrl)
            logged.diagnostics.close()
            with (Path(tmp)/'log/truth.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 100)
            first = rows[0]
            self.assertEqual(float(first['sim_time_s']), 0)
            self.assertAlmostEqual(float(first['sim_step_end_s']), .001)
            for i in range(31):
                self.assertEqual(float(first[f'q_{i}']), before[i])
            self.assertEqual(first['elastic_enabled'], '0')


if __name__ == '__main__':
    unittest.main()
