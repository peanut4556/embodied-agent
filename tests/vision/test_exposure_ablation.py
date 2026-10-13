"""Numerical bounds, five-rule ablation and independent batch regression checks."""

import json
import unittest
from pathlib import Path

import numpy as np

from embodied_agent.exposure_chernoff import bound_repeated_noise_chernoff, chi_square_ratios
from embodied_agent.exposure_uncertainty import variance_bounds
from scripts.audit_exposure_ablation import RULES, rule_inputs, summarize
from scripts.audit_exposure_uncertainty import seed_plan


class ExposureAblationTests(unittest.TestCase):
    def test_inversion_and_tighter_endpoints(self):
        for df in (127, 381, 223520):
            for p in (0.001, 0.01, 0.1):
                low, high = chi_square_ratios(df, p)
                for ratio in (low, high):
                    self.assertAlmostEqual(
                        df * (ratio - 1 - np.log(ratio)) / 2, -np.log(p), places=9
                    )
                lm = variance_bounds(1.0, df, p)
                self.assertLess(1 / low, lm["upper"])
                self.assertGreater(1 / high, lm["lower"])
                self.assertLess(low, 1)
                self.assertGreater(high, 1)

    def test_exact_df_two_tails_and_invalid_inputs(self):
        # chi-square(2) is exponential with scale 2: independent analytic reference.
        for p in (0.001, 0.01, 0.1):
            low, high = chi_square_ratios(2, p)
            self.assertLessEqual(-np.expm1(-low), p)
            self.assertLessEqual(np.exp(-high), p)
        for df, p in ((0, 0.1), (-1, 0.1), (127, 0), (127, 1), (np.nan, 0.1)):
            with self.assertRaises(ValueError):
                chi_square_ratios(df, p)

    def test_new_seed_plan(self):
        c = json.loads(Path("config/exposure-ablation.json").read_text())
        calibration, validation = seed_plan(c)
        all_seeds = [s for group in (calibration, validation) for batch in group for s in batch]
        self.assertEqual(len(set(all_seeds)), 32 * (128 + 80))
        self.assertGreater(min(all_seeds), 32559)
        self.assertEqual(c["rule_order"], list(RULES))

    def test_nested_component_bounds_and_budget_only_control(self):
        rng = np.random.default_rng(104)
        x = rng.normal(0, 0.001, (128, 4, 32)) + rng.normal(0, 0.001, (128, 4, 1))
        bounds = bound_repeated_noise_chernoff(x)
        config = {"nominal_alpha": 0.01, "bounded_rule_tail_probability": 0.005}
        sigmas, alphas = rule_inputs(bounds, [0.001, 0.001, 0], config)
        np.testing.assert_array_equal(sigmas[0], sigmas[1])
        self.assertEqual(alphas, [0.01, 0.005, 0.005, 0.005, 0.01])
        self.assertTrue(np.all(np.array(sigmas[3]) <= sigmas[2]))
        self.assertTrue(np.all(np.array(sigmas[3]) >= sigmas[1]))
        shifted = bound_repeated_noise_chernoff(x[::-1] + rng.uniform(-0.2, 0.2, (4, 32)))
        np.testing.assert_allclose(
            bounds["chernoff"]["upper_sigmas_m"], shifted["chernoff"]["upper_sigmas_m"], rtol=1e-10
        )

    def test_gaussian_coverage_regression(self):
        rng = np.random.default_rng(105)
        truth = np.array([0.001, 0.0015, 0.0007]) ** 2
        misses = 0
        for _ in range(1000):
            x = (
                rng.normal(0, 0.001, (128, 4, 16))
                + rng.normal(0, 0.0015, (128, 4, 1))
                + rng.normal(0, 0.0007, (128, 1, 1))
            )
            result = bound_repeated_noise_chernoff(x)["chernoff"]
            misses += np.any(np.array(result["upper_variance_components_m2"]) < truth)
        self.assertLessEqual(misses, 15)  # Regression guard, not a coverage proof.

    def test_paired_contrasts_and_unknowns(self):
        a = np.array([[[1, 1, 0, 1, 1], [-1] * 5], [[1, 0, 0, 0, 1], [0] * 5]])[..., None]
        indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        result = summarize(a, [0], indices)[0]
        self.assertEqual(result["rules"]["chernoff"]["unknown"], 1)
        self.assertEqual(result["rules"]["chernoff"]["alarm_fraction_converged"], 1 / 3)
        contrast = result["contrasts"]["chernoff_minus_lm"]
        self.assertEqual(contrast["alarms_added"], 1)
        self.assertEqual(contrast["alarms_removed"], 0)
        self.assertIsNone(
            summarize(np.full_like(a, -1), [0], indices)[0]["contrasts"]["chernoff_minus_lm"][
                "paired_cluster_95_percent"
            ]
        )
