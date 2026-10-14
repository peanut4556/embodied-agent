"""Physical covariance propagation and truth-blind rejection accounting."""

import json
import unittest
from pathlib import Path
from statistics import NormalDist

import numpy as np

from embodied_agent.fit_reliability import translation_covariance, translation_gate
from embodied_agent.reference_fit import SCALES, jacobian
from scripts.audit_exposure_uncertainty import seed_plan
from scripts.audit_fit_reliability import summarize


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        b = json.loads(Path("docs/evaluations/redundant-reference-v1.json").read_text())
        self.planes = [(np.array(p["normal"]), p["offset_m"]) for p in b["planes"][:3]]
        self.truth = np.array(b["true_parameters"])
        self.rays = np.array(
            [[x, y, -1.0] for x in np.linspace(-0.25, 0.25, 11) for y in np.linspace(-0.2, 0.2, 11)]
        )

    def test_covariance_matches_dense_correlated_observations(self):
        n = len(self.rays)
        sigmas = np.array([0.001, 0.002, 0.003])
        cov = translation_covariance(self.rays, self.planes, self.truth, sigmas)
        observation_cov = (
            sigmas[0] ** 2 * np.eye(3 * n)
            + sigmas[1] ** 2 * np.kron(np.eye(3), np.ones((n, n)))
            + sigmas[2] ** 2 * np.ones((3 * n, 3 * n))
        )
        inverse = SCALES[:, None] * np.linalg.pinv(
            jacobian(self.rays, self.planes, self.truth), rcond=1e-7
        )
        expected = (inverse @ observation_cov @ inverse.T)[:3, :3]
        np.testing.assert_allclose(cov, expected, rtol=1e-8, atol=1e-16)
        doubled = translation_covariance(self.rays, self.planes, self.truth, 2 * sigmas)
        np.testing.assert_allclose(doubled, 4 * cov, atol=1e-18)

    def test_noise_propagation_monte_carlo(self):
        rng = np.random.default_rng(151000)
        n = len(self.rays)
        sigma = [0.001, 0.002, 0.003]
        noise = (
            rng.normal(0, sigma[0], (8000, 3, n))
            + rng.normal(0, sigma[1], (8000, 3, 1))
            + rng.normal(0, sigma[2], (8000, 1, 1))
        )
        inverse = SCALES[:, None] * np.linalg.pinv(
            jacobian(self.rays, self.planes, self.truth), rcond=1e-7
        )
        errors = noise.reshape(8000, -1) @ inverse[:3].T
        expected = translation_covariance(self.rays, self.planes, self.truth, sigma)
        np.testing.assert_allclose(np.diag(np.cov(errors.T)), np.diag(expected), rtol=0.06)

    def test_radius_union_bound_and_threshold_boundary(self):
        cov = np.diag([1e-6, 4e-6, 9e-6])
        radius = NormalDist().inv_cdf(1 - 0.045 / 6) * np.sqrt(14e-6)
        gate = translation_gate(cov, radius)
        self.assertTrue(gate["passes"])
        self.assertFalse(translation_gate(cov, radius * 0.999)["passes"])
        self.assertFalse(gate["calibration_certified"])
        self.assertTrue(translation_gate(np.zeros((3, 3)))["passes"])

    def test_invalid_covariance_noise_and_rank(self):
        for cov in (np.eye(2), np.diag([-1.0, 1.0, 1.0]), np.full((3, 3), np.nan)):
            with self.assertRaises(ValueError):
                translation_gate(cov)
        with self.assertRaises(ValueError):
            translation_covariance(self.rays, self.planes, self.truth, [-1, 0, 0])
        with self.assertRaises(ValueError):
            translation_covariance(self.rays, [self.planes[0]] * 3, self.truth, [1, 1, 1])

    def test_fresh_seed_plan(self):
        c = json.loads(Path("config/fit-reliability.json").read_text())
        seeds = [s for g in seed_plan(c) for b in g for s in b]
        self.assertEqual(len(set(seeds)), 32 * (128 + 32))
        self.assertGreater(min(seeds), 111023)

    def test_rejections_and_empty_acceptance_denominator(self):
        cases = [
            {
                "translation_error_m": 0.008,
                "accepted": {"residual_only": True, "plugin": True, "bounded": False},
            },
            {
                "translation_error_m": 0.001,
                "accepted": {"residual_only": True, "plugin": False, "bounded": False},
            },
            {
                "translation_error_m": 0.001,
                "accepted": {"residual_only": False, "plugin": False, "bounded": False},
            },
        ]
        result = summarize([{"cases": cases}], np.array([[0]]), 0.005)
        self.assertEqual(result["plugin"]["accepted_over_limit"], 1)
        self.assertEqual(result["bounded"]["within_limit_rejected"], 2)
        self.assertEqual(result["bounded"]["within_limit_extra_rejections"], 1)
        self.assertIsNone(result["bounded"]["cluster_95_percent"]["risk_among_accepted"])


if __name__ == "__main__":
    unittest.main()
