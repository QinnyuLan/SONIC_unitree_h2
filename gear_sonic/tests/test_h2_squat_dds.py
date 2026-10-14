"""Synthetic DDS test on loopback, domain 187 only; never uses a robot NIC."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external_dependencies/unitree_sdk2_python"))


class ReadOnlyRecorderIntegration(unittest.TestCase):
    def test_private_loopback_capture(self):
        from cyclonedds.domain import Domain, DomainParticipant
        from cyclonedds.pub import DataWriter
        from cyclonedds.topic import Topic
        from unitree_sdk2py.core.channel_config import ChannelConfigHasInterface
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_, IMUState_, OdoState_
        from unitree_sdk2py.idl.default import (unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_,
                                               unitree_hg_msg_dds__IMUState_, unitree_hg_msg_dds__OdoState_)
        from unitree_sdk2py.utils.crc import CRC
        domain = Domain(187, ChannelConfigHasInterface.replace("$__IF_NAME__$", "lo"))
        participant = DomainParticipant(187)
        command_topic = Topic(participant, 'rt/lowcmd', LowCmd_)
        state_topic = Topic(participant, 'rt/lowstate', LowState_)
        writer_command = DataWriter(participant, command_topic)
        writer_state = DataWriter(participant, state_topic)
        imu_topic = Topic(participant, 'rt/secondary_imu', IMUState_)
        odo_topic = Topic(participant, 'rt/odostate', OdoState_)
        writer_imu = DataWriter(participant, imu_topic)
        writer_odo = DataWriter(participant, odo_topic)
        crc = CRC()
        command = unitree_hg_msg_dds__LowCmd_()
        state = unitree_hg_msg_dds__LowState_()
        imu = unitree_hg_msg_dds__IMUState_()
        odo = unitree_hg_msg_dds__OdoState_()
        command.motor_cmd[5].q = .12
        state.motor_state[5].q = .1
        command.crc = crc.Crc(command)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'dds'
            proc = subprocess.Popen([sys.executable, str(ROOT/'scripts/record_h2_squat_dds.py'),
                '--target', 'sim', '--interface', 'lo', '--domain', '187', '--seconds', '2',
                '--output', str(output)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                deadline = time.monotonic()+6
                while proc.poll() is None and time.monotonic() < deadline:
                    state.tick += 1
                    state.crc = crc.Crc(state)
                    writer_command.write(command)
                    writer_state.write(state)
                    writer_imu.write(imu)
                    writer_odo.write(odo)
                    time.sleep(.002)
                stdout, _ = proc.communicate(timeout=3)
                self.assertEqual(proc.returncode, 0, stdout)
                metadata = json.loads((output/'metadata.json').read_text())
                self.assertGreater(metadata['rows']['lowstate'], 10, stdout)
                self.assertGreater(metadata['rows']['lowcmd'], 10, stdout)
                self.assertGreater(metadata['rows']['secondary_imu'], 10, stdout)
                self.assertGreater(metadata['rows']['odometry_unverified'], 10, stdout)
                self.assertEqual(metadata['quality']['callback_errors'], [])
                self.assertEqual(metadata['quality']['max_matched_publishers']['lowcmd'], 1)
                self.assertEqual(metadata['quality']['invalid_crc']['lowstate'], 0)
                with (output/'lowstate.csv').open() as f:
                    rows = list(csv.DictReader(f))
                self.assertAlmostEqual(float(rows[0]['q_5']), .1)
                self.assertGreater(float(rows[0]['dds_source_timestamp_ns']), 0)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.communicate(timeout=5)


if __name__ == '__main__':
    unittest.main()
