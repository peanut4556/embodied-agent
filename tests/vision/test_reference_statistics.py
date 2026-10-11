"""Noise correlation, linear variance and small-sample interval regressions."""

import unittest

import numpy as np

from embodied_agent.reference_fit import jacobian
from embodied_agent.reference_statistics import mean_prediction_uncertainty, wilson_interval


class ReferenceStatisticsTests(unittest.TestCase):
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

    def test_global_depth_constant_is_absorbed(self):
        result = mean_prediction_uncertainty(self.rays, self.planes, self.p, 0, 0, 0.001)
        self.assertLess(result["variance_components_m2"]["shared_across_all_exposures"], 1e-20)
        self.assertFalse(result["calibration_certified"])

    def test_exposure_correlation_does_not_average_away(self):
        a = mean_prediction_uncertainty(self.rays, self.planes, self.p, 0.001)
        b = mean_prediction_uncertainty(self.rays, self.planes, self.p, 0.001, 0.001)
        self.assertGreater(b["standard_error_m"], a["standard_error_m"])
        self.assertGreaterEqual(b["variance_components_m2"]["shared_within_exposure"], 1e-6)

    def test_linear_monte_carlo_variance_matches_formula(self):
        result = mean_prediction_uncertainty(self.rays, self.planes, self.p, 0.001, 0.001)
        train = jacobian(self.rays, self.planes[:3], self.p)
        val = jacobian(self.rays, self.planes[3:], self.p)
        w = val.mean(0) @ np.linalg.pinv(train, rcond=1e-7)
        rng = np.random.default_rng(811)
        pixel = rng.normal(0, 0.001, (5000, 4, len(self.rays)))
        exposure = rng.normal(0, 0.001, (5000, 4, 1))
        noise = pixel + exposure
        errors = noise[:, 3].mean(1) - noise[:, :3].reshape(5000, -1) @ w
        ratio = np.var(errors) / result["standard_error_m"] ** 2
        self.assertGreater(ratio, 0.9)
        self.assertLess(ratio, 1.1)

    def test_zero_observed_false_alarms_does_not_imply_zero_rate(self):
        low, high = wilson_interval(0, 80)
        self.assertEqual(low, 0)
        self.assertGreater(high, 0.04)
        self.assertIsNone(wilson_interval(0, 0))

    def test_invalid_noise_or_counts(self):
        with self.assertRaises(ValueError):
            mean_prediction_uncertainty(self.rays, self.planes, self.p, -0.001)
        with self.assertRaises(ValueError):
            wilson_interval(2, 1)
