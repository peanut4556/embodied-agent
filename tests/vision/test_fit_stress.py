"""Stress mechanisms and scoring remain separate from solver certification."""

import json
import unittest
from pathlib import Path

import numpy as np

from embodied_agent.fit_stress import stress_geometry, stress_observation
from embodied_agent.reference_fit import predict
from embodied_agent.reference_fit_candidate import USABLE, fit_reference_candidate
from scripts.audit_exposure_uncertainty import seed_plan
from scripts.audit_fit_stress import risk_summary, summarize


class StressTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path("config/fit-stress.json").read_text())
        b = json.loads(Path("docs/evaluations/redundant-reference-v1.json").read_text())
        self.planes = [(np.array(p["normal"]), p["offset_m"]) for p in b["planes"]]
        self.truth = np.array(b["true_parameters"])
        self.rays = np.array(
            [[x, y, -1.0] for x in np.linspace(-0.25, 0.25, 21) for y in np.linspace(-0.2, 0.2, 21)]
        )
        self.clean = predict(self.rays, self.planes, self.truth)

    def test_coherent_shift_is_unobservable_camera_error(self):
        rays, actual = stress_geometry(
            self.rays, self.planes, "coherent_reference_shift", self.config
        )
        observed = predict(rays, actual, self.truth)
        equivalent = self.truth.copy()
        equivalent[:3] -= self.config["coherent_reference_translation_m"]
        np.testing.assert_allclose(observed, predict(rays, self.planes, equivalent), atol=1e-15)
        fit = fit_reference_candidate(rays, self.planes[:3], observed[:3])
        self.assertIn(fit["status"], USABLE)
        np.testing.assert_allclose(fit["parameters"], equivalent, atol=1e-8, rtol=0)
        self.assertGreater(np.linalg.norm(np.array(fit["parameters"][:3]) - self.truth[:3]), 0.005)
        np.testing.assert_allclose(
            predict(rays, self.planes[3:], fit["parameters"])[0], observed[3], atol=1e-8
        )

    def test_geometry_changes_are_local_and_nonmutating(self):
        original = self.rays.copy()
        rays, planes = stress_geometry(self.rays, self.planes, "near_degenerate", self.config)
        np.testing.assert_array_equal(rays[:, :2], original[:, :2] * 0.01)
        np.testing.assert_array_equal(self.rays, original)
        _, planes = stress_geometry(self.rays, self.planes, "single_reference_shift", self.config)
        self.assertAlmostEqual(planes[0][1] - self.planes[0][1], 0.006)
        self.assertEqual([p[1] for p in planes[1:]], [p[1] for p in self.planes[1:]])

    def test_bursts_are_validation_only_exposure_offsets(self):
        cfg = {**self.config, "validation_burst_probability": 1.0}
        base = stress_observation(self.clean, self.rays, 123, "baseline", "calibration", cfg)
        cal = stress_observation(
            self.clean, self.rays, 123, "validation_bursts", "calibration", cfg
        )
        val = stress_observation(self.clean, self.rays, 123, "validation_bursts", "validation", cfg)
        np.testing.assert_array_equal(base, cal)
        np.testing.assert_allclose(val - base, 0.01, atol=1e-15)

    def test_outliers_are_paired_signed_spikes(self):
        cfg = {**self.config, "outlier_probability": 1.0}
        base = stress_observation(self.clean, self.rays, 123, "baseline", "validation", cfg)
        outlier = stress_observation(
            self.clean, self.rays, 123, "pixel_outliers", "validation", cfg
        )
        np.testing.assert_allclose(abs(outlier - base), 0.02, atol=1e-15)
        self.assertTrue(np.any(outlier > base) and np.any(outlier < base))
        np.testing.assert_array_equal(self.clean, predict(self.rays, self.planes, self.truth))
        with self.assertRaises(ValueError):
            stress_observation(self.clean, self.rays, 123, "baseline", "invalid", cfg)

    def test_fresh_independent_batches(self):
        groups = seed_plan(self.config)
        seeds = [s for group in groups for batch in group for s in batch]
        self.assertEqual(len(seeds), 32 * (128 + 32))
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertGreater(min(seeds), 72559)

    def test_unknown_denominators_and_pose_risk(self):
        a = np.array([[[[-1], [0]], [[1], [1]], [[0], [0]]]])
        result = summarize(a, [0], np.array([[0]]))[0]
        self.assertEqual(result["newly_usable"], 1)
        self.assertEqual(result["common_usable"], 2)
        self.assertEqual(result["common_decision_disagreements"], 0)
        self.assertEqual(result["rules"]["candidate"]["alarm_fraction_usable"], 1 / 3)
        self.assertIsNone(
            summarize(np.full_like(a, -1), [0], np.array([[0]]))[0]["rules"]["candidate"][
                "cluster_95_percent"
            ]
        )
        cases = [
            {"decisions": [[-1], [0]], "scoring_errors": [{"translation_norm_m": 0.008}] * 2},
            {"decisions": [[1], [1]], "scoring_errors": [{"translation_norm_m": 0.008}] * 2},
        ]
        risk = risk_summary(cases, 0.005)
        self.assertEqual(risk["candidate"]["no_alarm_translation_error_over_limit"], 1)
        self.assertEqual(risk["candidate"]["newly_usable_no_alarm_translation_error_over_limit"], 1)
        self.assertEqual(risk["legacy"]["no_alarm_translation_error_over_limit"], 0)
