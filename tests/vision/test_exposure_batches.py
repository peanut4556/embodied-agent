"""Check batch separation, unknown denominators and cached variance propagation."""

import json
import unittest
from pathlib import Path

import numpy as np

from embodied_agent.reference_statistics import mean_prediction_uncertainty
from scripts.audit_exposure_batches import VARIANCE_KEYS, counts, seed_plan, thresholds


class ExposureBatchTests(unittest.TestCase):
    def test_frozen_seeds_are_disjoint_from_each_other_and_prior_experiments(self):
        config = json.loads(Path("config/repeated-exposure-batches.json").read_text())
        batches, validation = seed_plan(config)
        flat = [seed for batch in batches for seed in batch]
        self.assertEqual(len(set(flat)), 32 * 128)
        self.assertEqual(len(validation), 512)
        self.assertFalse(set(flat) & set(validation))
        self.assertTrue(min(flat + validation) > 3079)

    def test_overlapping_validation_rejected(self):
        with self.assertRaises(ValueError):
            seed_plan(
                {
                    "batch_count": 2,
                    "repeats_per_batch": 128,
                    "calibration_seed_start": 4000,
                    "validation_seed_start": 4100,
                    "validation_count": 80,
                }
            )

    def test_unknowns_not_counted_as_negative_or_converged(self):
        result = counts([-1, -1, 0, 1])
        self.assertEqual(result["unknown"], 2)
        self.assertEqual(result["alarm_fraction_all_cases"], 0.25)
        self.assertEqual(result["alarm_fraction_converged"], 0.5)
        self.assertIsNone(counts([-1, -1])["wilson_95_percent_converged"])

    def test_cached_thresholds_match_direct_propagation(self):
        baseline = json.loads(Path("docs/evaluations/redundant-reference-v1.json").read_text())
        planes = [(np.array(p["normal"]), p["offset_m"]) for p in baseline["planes"]]
        for extent in (0.25, 0.025):
            rays = np.array(
                [
                    [x, y, -1]
                    for x in np.linspace(-extent, extent, 21)
                    for y in np.linspace(-extent * 0.8, extent * 0.8, 21)
                ]
            )
            unit = mean_prediction_uncertainty(rays, planes, baseline["true_parameters"], 1, 1, 1)
            coefficients = [unit["variance_components_m2"][key] for key in VARIANCE_KEYS]
            sigmas = np.array(
                [[0, 0, 0], [0.001, 0, 0.001], [0.001, 0.001, 0], [0.0008, 0.0012, 0.0006]]
            )
            cached = thresholds(coefficients, sigmas)
            direct = [
                mean_prediction_uncertainty(rays, planes, baseline["true_parameters"], *s)[
                    "two_sided_99_percent_threshold_m"
                ]
                for s in sigmas
            ]
            np.testing.assert_allclose(cached, direct, rtol=1e-12, atol=1e-15)
