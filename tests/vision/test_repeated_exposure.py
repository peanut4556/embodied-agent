"""Noise estimation uses temporal repetition, not known true geometry or validation."""

import unittest

import numpy as np

from embodied_agent.repeated_exposure import estimate_repeated_noise


class RepeatedExposureTests(unittest.TestCase):
    @staticmethod
    def observations():
        rng = np.random.default_rng(701)
        static = rng.uniform(0.8, 1.3, (4, 128))
        return (
            static
            + rng.normal(0, 0.001, (500, 4, 128))
            + rng.normal(0, 0.0015, (500, 4, 1))
            + rng.normal(0, 0.0007, (500, 1, 1))
        )

    def test_recovers_mixed_variance_components(self):
        result = estimate_repeated_noise(self.observations())
        for key, expected in (
            ("sigma_pixel_m", 0.001),
            ("sigma_exposure_m", 0.0015),
            ("sigma_global_m", 0.0007),
        ):
            self.assertAlmostEqual(result[key], expected, delta=0.00015)
        self.assertFalse(result["calibration_certified"])

    def test_static_geometry_and_offsets_do_not_reveal_bias(self):
        x = self.observations()
        a = estimate_repeated_noise(x)
        shift = np.random.default_rng(5).uniform(-0.1, 0.1, (4, 128))
        b = estimate_repeated_noise(x + shift)
        for key in ("sigma_pixel_m", "sigma_exposure_m", "sigma_global_m"):
            self.assertAlmostEqual(a[key], b[key], places=12)

    def test_repeat_order_cannot_change_estimate(self):
        x = self.observations()
        a, b = estimate_repeated_noise(x), estimate_repeated_noise(x[::-1])
        for key in ("sigma_pixel_m", "sigma_exposure_m", "sigma_global_m"):
            self.assertAlmostEqual(a[key], b[key], places=12)

    def test_constant_repeats_have_no_measurable_noise(self):
        x = np.broadcast_to(np.arange(12).reshape(3, 4), (20, 3, 4))
        result = estimate_repeated_noise(x)
        self.assertEqual(result["sigma_pixel_m"], 0)
        self.assertEqual(result["sigma_exposure_m"], 0)
        self.assertEqual(result["sigma_global_m"], 0)

    def test_invalid_data_rejected(self):
        for x in (np.zeros((1, 4, 8)), np.zeros((10, 8)), np.full((10, 4, 8), np.nan)):
            with self.assertRaises(ValueError):
                estimate_repeated_noise(x)
