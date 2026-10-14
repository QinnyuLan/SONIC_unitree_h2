import unittest

from gear_sonic.scripts.pico_manager_thread_server import (
    DEFAULT_PLANNER_MAX_SPEED,
    DEFAULT_PLANNER_SPEED_SCALE,
    LocomotionMode,
    planner_speed_command,
)


class PlannerSpeedLimitTest(unittest.TestCase):
    def test_default_limiter_reduces_slow_walk(self):
        speed = planner_speed_command(
            LocomotionMode.SLOW_WALK,
            1.0,
            DEFAULT_PLANNER_SPEED_SCALE,
            DEFAULT_PLANNER_MAX_SPEED,
        )
        self.assertAlmostEqual(speed, 0.45)

    def test_walk_uses_explicit_speed_instead_of_model_default(self):
        speed = planner_speed_command(
            LocomotionMode.WALK,
            0.5,
            DEFAULT_PLANNER_SPEED_SCALE,
            DEFAULT_PLANNER_MAX_SPEED,
        )
        self.assertAlmostEqual(speed, 0.42)

    def test_run_is_capped_for_teleop_safety(self):
        speed = planner_speed_command(
            LocomotionMode.RUN,
            1.0,
            DEFAULT_PLANNER_SPEED_SCALE,
            DEFAULT_PLANNER_MAX_SPEED,
        )
        self.assertEqual(speed, DEFAULT_PLANNER_MAX_SPEED)

    def test_invalid_limits_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "scale"):
            planner_speed_command(LocomotionMode.WALK, 1.0, 0.0, 0.35)
        with self.assertRaisesRegex(ValueError, "max speed"):
            planner_speed_command(LocomotionMode.WALK, 1.0, 0.35, 0.0)


if __name__ == "__main__":
    unittest.main()
