import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/audit_h2_actuator_effort.py'
SPEC = importlib.util.spec_from_file_location('audit_h2_actuator_effort', SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class EffortAuditTests(unittest.TestCase):
    def test_causal_hold_never_selects_future_command(self):
        np.testing.assert_array_equal(audit.causal_indices(np.array([2., 4., 8.]),
                                     np.array([1., 2., 3., 4., 9.])), [-1, 0, 0, 1, 2])

    def test_causal_hold_rejects_nonmonotonic_time(self):
        with self.assertRaises(ValueError):
            audit.causal_indices(np.array([2., 1.]), np.array([3.]))

    def test_affine_relation(self):
        x = np.arange(20.)
        result = audit.linear_relation(x, .4*x-3)
        self.assertAlmostEqual(result['gain'], .4)
        self.assertAlmostEqual(result['intercept'], -3.)
        self.assertLess(result['residual_rms'], 1e-12)

    def test_constant_signal_is_not_a_gain_estimate(self):
        self.assertIsNone(audit.linear_relation(np.arange(20.), np.ones(20))['gain'])

    def test_positive_lag_means_past_effort(self):
        rng = np.random.default_rng(10)
        t = np.arange(1000.)
        pd = rng.normal(size=len(t))
        tau = np.interp(t-7, t, pd)
        rows = audit.lag_scan(t, pd, tau, t > 40, 15)
        best = max(rows, key=lambda r: r['correlation'])
        self.assertEqual(best['lag_ms'], 7)
        self.assertAlmostEqual(best['correlation'], 1.)

    def test_publisher_handles_are_not_rounded_through_float(self):
        handles = np.array(['18446744073709551614', '18446744073709551615'])
        rows = audit.publisher_summary(np.array([1., 2.]), handles, [(1., 3.)])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['handle'], handles[0])


if __name__ == '__main__':
    unittest.main()
