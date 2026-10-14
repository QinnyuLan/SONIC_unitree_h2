import contextlib
import io
import math
import unittest
from unittest.mock import Mock, patch

from gear_sonic.utils.teleop.head_pose_command import HeadPitchJoystick, FixedHeadPose
from gear_sonic.scripts.pico_manager_thread_server import HeadPoseStreamer


class HeadPitchJoystickTest(unittest.TestCase):
    def test_fixed_40_degrees_ignores_stick_and_headset(self):
        control = FixedHeadPose(pitch_deg=40, yaw_deg=0)
        reader, socket = Mock(), Mock()
        stream = HeadPoseStreamer(socket, reader, pitch_joystick=control)
        with patch('gear_sonic.scripts.pico_manager_thread_server.time.monotonic') as clock, \
             patch('gear_sonic.scripts.pico_manager_thread_server.build_head_pose_message') as pack, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            for i, axis in enumerate((1, -1, 0, .5, float('nan'))):
                clock.return_value = 1 + i * .05
                reader.get_latest.return_value = dict(timestamp_ns=i+1,
                    timestamp_monotonic=clock.return_value, right_stick_y=axis,
                    headset_pose=[0, 0, 0, 1, 0, 0, 0])
                stream.run_once()
                self.assertAlmostEqual(pack.call_args.args[0], math.radians(40))
                self.assertEqual(pack.call_args.args[1], 0)
                stream.calibrate_now({})
            self.assertEqual(socket.send.call_count, 5)
            self.assertIn('[HeadFixed] pitch_target=+40.00 deg', output.getvalue())

    def test_rate_direction_and_release_hold(self):
        control = HeadPitchJoystick(yaw_deg=12)
        for _ in range(20):
            control.update(1, .05)
        self.assertAlmostEqual(math.degrees(control.pitch), 5)
        self.assertAlmostEqual(math.degrees(control.yaw), 12)
        held = control.pitch
        for axis in (0, .1, -.15):
            self.assertEqual(control.update(axis, .1)[0], held)
            self.assertFalse(control.moving)
        for _ in range(20):
            control.update(-1, .05)
        self.assertAlmostEqual(control.pitch, 0)

    def test_limits_and_stalls(self):
        control = HeadPitchJoystick(pitch_deg=47.9)
        control.update(1, 10)
        self.assertEqual(control.pitch, control.PITCH_LIMITS[1])
        control = HeadPitchJoystick(pitch_deg=-29.9)
        control.update(-1, .1)
        self.assertEqual(control.pitch, control.PITCH_LIMITS[0])
        control = HeadPitchJoystick()
        control.update(1, 10)
        self.assertAlmostEqual(math.degrees(control.pitch), .5)
        before = control.pitch
        for value, dt in ((float('nan'), .1), (float('inf'), .1), (1, -1), (1, float('nan'))):
            self.assertEqual(control.update(value, dt)[0], before)

    def test_invalid_settings(self):
        for kwargs in ({'pitch_deg':49}, {'yaw_deg':101}, {'rate_deg_s':0},
                       {'rate_deg_s':31}, {'deadzone':1}, {'pitch_deg':float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HeadPitchJoystick(**kwargs)

    def test_stream_freshness_hold_print_and_calibration(self):
        control = HeadPitchJoystick(pitch_deg=10, yaw_deg=3)
        reader, socket = Mock(), Mock()
        stream = HeadPoseStreamer(socket, reader, pitch_joystick=control)
        output = io.StringIO()
        def sample(stamp, received, axis):
            reader.get_latest.return_value = dict(timestamp_ns=stamp,
                timestamp_monotonic=received, right_stick_y=axis)
        with patch('gear_sonic.scripts.pico_manager_thread_server.time.monotonic') as clock, \
             patch('gear_sonic.scripts.pico_manager_thread_server.build_head_pose_message') as pack, \
             contextlib.redirect_stdout(output):
            clock.return_value = 1
            sample(1, 1, 1)
            stream.run_once()
            self.assertIn('pitch_target=+10.00 deg', output.getvalue())
            clock.return_value = 1.05
            sample(2, 1.05, 1)
            stream.run_once()
            self.assertAlmostEqual(math.degrees(control.pitch), 10.25)
            self.assertEqual(pack.call_args.args[1], math.radians(3))
            clock.return_value = 1.1
            sample(3, 1.1, 0)
            stream.run_once()
            self.assertIn('pitch_target=+10.25 deg', output.getvalue())
            self.assertIn('[hold]', output.getvalue())
            calls = socket.send.call_count
            stream.run_once()  # duplicate sample must not keep stale input alive
            self.assertEqual(socket.send.call_count, calls)
            clock.return_value = 2
            sample(4, 1.1, 1)  # stale sample
            stream.run_once()
            self.assertEqual(socket.send.call_count, calls)
            sample(5, 2, 1)
            stream.run_once()  # first fresh sample after outage does not integrate gap
            self.assertAlmostEqual(math.degrees(control.pitch), 10.25)
            self.assertTrue(stream.calibrate_now({}))
            self.assertAlmostEqual(math.degrees(control.pitch), 10.25)


if __name__ == '__main__':
    unittest.main()
