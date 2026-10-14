import argparse
from contextlib import redirect_stderr
import importlib.util
import io
import math
from pathlib import Path
import sys
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "gear_sonic_deploy" / "scripts" / "h2_inspire_hand_dds_demo.py"
SPEC = importlib.util.spec_from_file_location("h2_inspire_hand_dds_demo", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
demo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = demo
SPEC.loader.exec_module(demo)


def snapshot(q, lost=None):
    return demo.StateSnapshot(
        q=tuple(q),
        lost=tuple([0] * demo.TOTAL_DOF if lost is None else lost),
        received_at=0.0,
        sequence=1,
    )


class InspireDdsArgumentTest(unittest.TestCase):
    def test_parses_joint_values_in_hardware_order(self):
        self.assertEqual(
            demo.joint_positions("1, .9, .8, .7, .6, .5"),
            (1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
        )

    def test_rejects_wrong_joint_count_and_range(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            demo.joint_positions("1,1,1")
        with self.assertRaises(argparse.ArgumentTypeError):
            demo.joint_positions("1,1,1,1,1,-0.1")

    def test_no_target_means_monitor_only(self):
        args = demo.parse_args([])
        self.assertIsNone(demo.requested_pattern(args))
        self.assertEqual(args.network, "eno1")

    def test_strong_close_requires_explicit_override(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                demo.parse_args(["--position", "0.2"])
        args = demo.parse_args(["--position", "0.2", "--allow-full-close"])
        self.assertEqual(demo.requested_pattern(args), (0.2,) * demo.HAND_DOF)


class InspireDdsCommandTest(unittest.TestCase):
    def test_right_target_preserves_left_state(self):
        current = tuple(index / 20.0 for index in range(demo.TOTAL_DOF))
        target = demo.build_target(snapshot(current), "right", (0.8,) * demo.HAND_DOF)
        self.assertEqual(target[: demo.HAND_DOF], (0.8,) * demo.HAND_DOF)
        self.assertEqual(target[demo.HAND_DOF :], current[demo.HAND_DOF :])

    def test_both_target_sets_exactly_twelve_values(self):
        target = demo.build_target(
            snapshot((1.0,) * demo.TOTAL_DOF),
            "both",
            (1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
        )
        self.assertEqual(target[: demo.HAND_DOF], target[demo.HAND_DOF :])
        message = demo.make_command(target)
        self.assertEqual(len(message.cmds), demo.TOTAL_DOF)
        self.assertEqual(tuple(motor.q for motor in message.cmds), target)
        self.assertTrue(all(motor.mode == 0 for motor in message.cmds))

    def test_unavailable_unselected_side_uses_open_placeholder(self):
        current = (0.9,) * demo.HAND_DOF + (math.nan,) * demo.HAND_DOF
        lost = (0,) * demo.HAND_DOF + (1,) * demo.HAND_DOF
        state = snapshot(current, lost)
        target = demo.build_target(state, "right", (0.8,) * demo.HAND_DOF)
        self.assertEqual(target[demo.HAND_DOF :], (1.0,) * demo.HAND_DOF)
        self.assertTrue(demo.side_online(state, demo.RIGHT_OFFSET))
        self.assertFalse(demo.side_online(state, demo.LEFT_OFFSET))


if __name__ == "__main__":
    unittest.main()
