import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sim.realtime import RealtimeOverrun, RealtimePacer


class FakeClock:
    def __init__(self, oversleep=0):
        self.now = 100.0
        self.oversleep = oversleep
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds + self.oversleep


class RealtimeTests(unittest.TestCase):
    def test_launcher_propagates_simulation_failure(self):
        from sim.factory import SimulatorFactory
        from unittest.mock import Mock

        simulator = Mock()
        simulator.start.side_effect = RealtimeOverrun("simulation fell behind wall clock")
        with self.assertRaisesRegex(RealtimeOverrun, "fell behind"):
            SimulatorFactory.start_simulator(simulator, as_thread=False)
        simulator.close.assert_called_once_with()

    def test_oversleep_does_not_accumulate(self):
        clock = FakeClock(.0005)
        pacer = RealtimePacer(.001, clock=clock, sleep=clock.sleep)
        for _ in range(10000):
            clock.now += .0007
            pacer.finish_step()
        state = pacer.snapshot()
        self.assertEqual(state['physics_steps'], 10000)
        self.assertAlmostEqual(state['physics_elapsed_s'], 10)
        self.assertLess(state['max_lag_ms'], .501)
        self.assertLess(state['wall_elapsed_s'], 10.001)

    def test_transient_stall_catches_up_without_skipping_steps(self):
        clock = FakeClock()
        pacer = RealtimePacer(.001, clock=clock, sleep=clock.sleep)
        clock.now += .009
        pacer.finish_step()
        self.assertAlmostEqual(pacer.lag, .008)
        for _ in range(30):
            clock.now += .0005
            pacer.finish_step()
        self.assertEqual(pacer.steps, 31)
        self.assertAlmostEqual(pacer.now - pacer.start, .031)

    def test_sustained_overload_stops_instead_of_resetting_epoch(self):
        clock = FakeClock()
        pacer = RealtimePacer(.001, clock=clock, sleep=clock.sleep)
        with self.assertRaises(RealtimeOverrun):
            for _ in range(100):
                clock.now += .0015
                pacer.finish_step()
        self.assertLess(pacer.steps, 42)
        self.assertIsNotNone(pacer.failure)
        self.assertGreater(pacer.max_lag, .02)

    def test_large_oversleep_is_also_rejected(self):
        clock = FakeClock(.05)
        pacer = RealtimePacer(.001, clock=clock, sleep=clock.sleep)
        with self.assertRaises(RealtimeOverrun):
            pacer.finish_step()

    def test_invalid_limits(self):
        for invalid in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                RealtimePacer(invalid)
            with self.assertRaises(ValueError):
                RealtimePacer(.001, invalid)

    def test_simulator_overrun_marks_diagnostics_incomplete(self):
        from sim.environment import BaseSimulator

        clock = FakeClock()
        rows, closes = [], []
        diagnostics = SimpleNamespace(
            log=SimpleNamespace(metadata={}), record_timing=rows.append,
            close=lambda **kwargs: closes.append(kwargs))
        sim = BaseSimulator.__new__(BaseSimulator)
        sim.config = {'REALTIME_MAX_LAG_MS': 20}
        sim.sim_dt = .001
        sim.viewer_dt = sim.reward_dt = sim.image_dt = .02
        sim._running = True
        sim.redis_client = None
        sim.unitree_bridge = SimpleNamespace(WarmupSerialization=lambda: None)

        def step():
            clock.now += .0015

        sim.sim_env = SimpleNamespace(viewer=None, diagnostics=diagnostics,
            image_publish_process=None, sim_step=step, update_viewer=lambda: None,
            update_reward=lambda: None, update_render_caches=lambda: None, prepare_obs=lambda: None)
        module = 'sim.environment'
        with patch(module+'.time.monotonic', clock), patch(module+'.RealtimePacer',
                side_effect=lambda dt, limit: RealtimePacer(dt, limit, clock=clock, sleep=clock.sleep)):
            with self.assertRaises(RealtimeOverrun):
                sim.start()
        self.assertEqual(closes, [{'incomplete': True}])
        self.assertFalse(diagnostics.log.metadata['realtime']['valid'])
        self.assertGreater(rows[-1]['max_lag_ms'], 20)

    def test_viewer_state_sync_flag_is_explicit(self):
        from sim.environment import DefaultEnv
        from unittest.mock import Mock

        env = DefaultEnv.__new__(DefaultEnv)
        env.viewer = Mock()
        env.config = {}
        env.update_viewer()
        env.viewer.sync.assert_called_once_with(state_only=False)
        env.config['VIEWER_STATE_ONLY'] = True
        env.update_viewer()
        env.viewer.sync.assert_called_with(state_only=True)


if __name__ == '__main__':
    unittest.main()
