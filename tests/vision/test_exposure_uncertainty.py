"""Check variance coverage, static invariance, data separation and paired summaries."""

import json
import unittest
from pathlib import Path

import numpy as np

from embodied_agent.exposure_uncertainty import bound_repeated_noise, variance_bounds
from scripts.audit_exposure_batches import SIGMA_KEYS
from scripts.audit_exposure_uncertainty import seed_plan, summarize


class ExposureUncertaintyTests(unittest.TestCase):
    def test_new_seed_batches_are_disjoint(self):
        config = json.loads(Path("config/exposure-uncertainty.json").read_text())
        calibration, validation = seed_plan(config)
        self.assertEqual((len(calibration), len(validation)), (32, 32))
        flat = [s for group in (calibration, validation) for batch in group for s in batch]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertGreater(min(flat), 10511)
        self.assertEqual((len(calibration[0]), len(validation[0])), (128, 80))
        config["validation_seed_start"] = config["calibration_seed_start"]
        with self.assertRaises(ValueError):
            seed_plan(config)

    def test_endpoints_and_budget_monotonicity(self):
        loose = variance_bounds(2.0, 127, 0.01)
        strict = variance_bounds(2.0, 127, 0.001)
        self.assertLess(strict["lower"], loose["lower"])
        self.assertLess(loose["lower"], 2)
        self.assertGreater(strict["upper"], loose["upper"])
        self.assertGreater(loose["upper"], 2)

    def test_static_shifts_and_repeat_permutations_preserve_bounds(self):
        rng = np.random.default_rng(81)
        x = rng.normal(0, 0.001, (128, 4, 16))
        x += rng.normal(0, 0.001, (128, 4, 1))
        x += rng.normal(0, 0.001, (128, 1, 1))
        a = bound_repeated_noise(x)
        b = bound_repeated_noise(x[::-1] + rng.uniform(-0.2, 0.2, (4, 16)))
        np.testing.assert_allclose(a["upper_sigmas_m"], b["upper_sigmas_m"], rtol=1e-11)
        self.assertTrue(
            np.all(np.array(a["upper_sigmas_m"]) >= [a["estimate"][key] for key in SIGMA_KEYS])
        )
        pixel, exposure, global_shared = [
            a["estimate"]["raw_variance_components_m2"][key]
            for key in ("pixel", "exposure", "global")
        ]
        aa, bb, cc = a["orthogonal_variance_estimates_m2"]
        np.testing.assert_allclose(
            [aa, bb - aa / 16, cc - bb / 4], [pixel, exposure, global_shared], atol=1e-18
        )

    def test_mixed_gaussian_variances_covered_in_independent_simulations(self):
        rng = np.random.default_rng(82)
        truth = np.array([0.001, 0.0015, 0.0007]) ** 2
        misses = 0
        for _ in range(1000):
            x = (
                rng.normal(0, np.sqrt(truth[0]), (128, 4, 16))
                + rng.normal(0, np.sqrt(truth[1]), (128, 4, 1))
                + rng.normal(0, np.sqrt(truth[2]), (128, 1, 1))
            )
            upper = bound_repeated_noise(x)["upper_variance_components_m2"]
            misses += bool(np.any(np.array(upper) < truth))
        # Loose finite-Monte-Carlo regression guard, not a proof of 99.5% coverage.
        self.assertLessEqual(misses, 15)

    def test_zero_data_and_invalid_bounds(self):
        self.assertEqual(
            bound_repeated_noise(np.zeros((128, 4, 16)))["upper_sigmas_m"], [0.0, 0.0, 0.0]
        )
        for variance, df, p in (
            (-1, 127, 0.01),
            (1, 0, 0.01),
            (1, 127, 0),
            (1, 127, 1),
            (1, 127, np.nan),
            (1, 2, 0.001),
        ):
            with self.assertRaises(ValueError):
                variance_bounds(variance, df, p)

    def test_cluster_summary_retains_pairing_and_unknowns(self):
        # Two batches, three cases, three rules, one offset.
        decisions = np.array(
            [[[1, 0, 1], [0, 0, 0], [-1, -1, -1]], [[1, 1, 1], [1, 0, 1], [0, 0, 0]]]
        )[..., None]
        indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        result = summarize(decisions, [0], indices)[0]
        self.assertEqual(result["rules"]["plugin"]["alarm_fraction_converged"], 3 / 5)
        self.assertEqual(result["rules"]["bounded"]["alarm_fraction_all_cases"], 1 / 6)
        self.assertEqual(result["rules"]["bounded"]["unknown"], 1)
        self.assertEqual(result["plugin_alarm_removed_by_bound"], 2)
        self.assertEqual(result["bound_only_alarm"], 0)
        self.assertLess(result["bounded_minus_plugin_paired_cluster_95_percent"][1], 0)
        self.assertNotIn("wilson_95_percent_converged", result["rules"]["plugin"])
        all_unknown = summarize(np.full_like(decisions, -1), [0], indices)[0]
        self.assertIsNone(all_unknown["bounded_minus_plugin_paired_cluster_95_percent"])
