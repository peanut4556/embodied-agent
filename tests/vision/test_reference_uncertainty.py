"""First-order bounds cover correlated errors and never consume validation residuals."""

import unittest

import numpy as np

from embodied_agent.reference_fit import jacobian, predict
from embodied_agent.reference_uncertainty import check_uncertain_reference, propagated_threshold


class ReferenceUncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.rays = np.array(
            [[x, y, -1.0] for x in np.linspace(-0.25, 0.25, 9) for y in np.linspace(-0.2, 0.2, 9)]
        )
        self.planes = [
            (np.array([0.0, 0.0, 1.0]), 0.0),
            (np.array([1.0, 0.0, 1.0]) / np.sqrt(2), 0.4 / np.sqrt(2)),
            (np.array([0.0, 1.0, 1.0]) / np.sqrt(2), 0.0),
            (np.array([-1.0, 1.0, 1.0]) / np.sqrt(3), -0.4 / np.sqrt(3)),
        ]
        self.p = np.zeros(7)

    def test_zero_noise_keeps_original_floor(self):
        threshold, _ = propagated_threshold(self.rays, self.planes, self.p, 0)
        np.testing.assert_array_equal(threshold, np.full(len(self.rays), 0.003))

    def test_correlated_worst_case_linear_error_is_covered(self):
        b = 0.003
        threshold, gain = propagated_threshold(self.rays, self.planes, self.p, b)
        train = jacobian(self.rays, self.planes[:3], self.p)
        validation = jacobian(self.rays, self.planes[3:], self.p)
        w = validation @ np.linalg.pinv(train, rcond=1e-7)
        i = int(np.argmax(gain))
        error = b * np.sign(w[i])
        propagated = w @ error + b
        self.assertTrue(np.all(np.abs(propagated) <= threshold + 1e-12))
        self.assertAlmostEqual(propagated[i], threshold[i], places=10)

    def test_larger_budget_cannot_reduce_threshold(self):
        a, _ = propagated_threshold(self.rays, self.planes, self.p, 0.001)
        b, _ = propagated_threshold(self.rays, self.planes, self.p, 0.003)
        self.assertTrue(np.all(b >= a))

    def test_validation_values_do_not_choose_threshold(self):
        observed = predict(self.rays, self.planes, self.p)
        fit = {"status": "converged", "parameters": self.p.tolist()}
        a = check_uncertain_reference(self.rays, self.planes, observed, fit, 0.001)
        changed = observed.copy()
        changed[3] += 0.1
        b = check_uncertain_reference(self.rays, self.planes, changed, fit, 0.001)
        self.assertEqual(a["threshold_range_m"], b["threshold_range_m"])
        self.assertEqual(a["status"], "no_alarm")
        self.assertEqual(b["status"], "alarm")
        self.assertFalse(b["calibration_certified"])

    def test_unknown_fit_and_invalid_budget(self):
        observed = predict(self.rays, self.planes, self.p)
        result = check_uncertain_reference(
            self.rays, self.planes, observed, {"status": "stalled"}, 0.001
        )
        self.assertEqual(result["status"], "unknown")
        with self.assertRaises(ValueError):
            propagated_threshold(self.rays, self.planes, self.p, -1)
