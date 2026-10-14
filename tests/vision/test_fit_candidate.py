"""Fixed stationarity gates must not accept large gradients, gains or boundary steps."""

import json
import unittest
from pathlib import Path

import numpy as np

from embodied_agent.reference_fit import fit_reference, predict
from embodied_agent.reference_fit_candidate import (
    EPS,
    LIMITS,
    USABLE,
    fit_reference_candidate,
    stationary_stop,
)
from scripts.audit_exposure_batches import draw
from scripts.audit_exposure_uncertainty import seed_plan
from scripts.audit_fit_stop import aggregate, negative_controls


class FitCandidateTests(unittest.TestCase):
    def setUp(self):
        b = json.loads(Path("docs/evaluations/redundant-reference-v1.json").read_text())
        self.planes = [(np.array(p["normal"]), p["offset_m"]) for p in b["planes"]]
        self.truth = np.array(b["true_parameters"])
        self.rays = np.array(
            [
                [x, y, -1.0]
                for x in np.linspace(-0.025, 0.025, 21)
                for y in np.linspace(-0.02, 0.02, 21)
            ]
        )

    def test_each_stationarity_guard(self):
        j = np.vstack([np.eye(7), np.zeros(7)])
        small = np.full(7, 1e-10)
        r = np.r_[small, 1.0]
        self.assertTrue(stationary_stop(j, r, small, np.zeros(7), 7)["eligible"])
        self.assertFalse(stationary_stop(j, r, small, np.zeros(7), 6)["eligible"])
        self.assertFalse(stationary_stop(j, r, small, LIMITS.copy(), 7)["eligible"])
        self.assertFalse(
            stationary_stop(j, np.r_[np.ones(7), 1.0], np.ones(7), np.zeros(7), 7)["eligible"]
        )
        # Each column cosine passes, but their combined predicted gain exceeds epsilon*SSE.
        step = np.full(7, 0.7 * np.sqrt(EPS))
        guard = stationary_stop(j, np.r_[step, 1.0], step, np.zeros(7), 7)
        self.assertLess(guard["max_residual_column_cosine"], np.sqrt(EPS))
        self.assertFalse(guard["eligible"])

    def test_historical_development_stall_gets_distinct_label(self):
        observed = draw(predict(self.rays, self.planes, self.truth), 50003, 0, 0)[:3]
        original = fit_reference(self.rays, self.planes[:3], observed)
        candidate = fit_reference_candidate(self.rays, self.planes[:3], observed)
        self.assertEqual(original["status"], "stalled")
        self.assertEqual(candidate["status"], "numerically_stationary")
        self.assertTrue(candidate["terminal_guard"]["eligible"])
        self.assertFalse(candidate["calibration_certified"])

    def test_noiseless_recovery_and_invalid_input(self):
        result = fit_reference_candidate(
            self.rays, self.planes[:3], predict(self.rays, self.planes[:3], self.truth)
        )
        self.assertIn(result["status"], USABLE)
        np.testing.assert_allclose(result["parameters"], self.truth, atol=1e-8, rtol=0)
        with self.assertRaises(ValueError):
            fit_reference_candidate(
                self.rays, self.planes[:3], np.full((3, len(self.rays)), np.nan)
            )

    def test_rank_and_boundary_controls_remain_unknown(self):
        for c in negative_controls(self.rays, self.planes, self.truth):
            self.assertNotIn(c["candidate"]["status"], USABLE, c["name"])
            self.assertNotEqual(c["legacy"]["status"], "converged", c["name"])

    def test_new_data_separation(self):
        c = json.loads(Path("config/fit-stop-validation.json").read_text())
        groups = seed_plan(c)
        seeds = [s for g in groups for b in g for s in b]
        self.assertEqual(len(seeds), 32 * (128 + 80))
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertGreater(min(seeds), 52559)

    def test_common_and_new_usable_denominators(self):
        # batch, case, rule, offset; first case newly usable, second common.
        a = np.array([[[-1, 1, -1, 1], [0, 0, 0, 0]], [[1, 1, 1, 1], [-1, -1, -1, -1]]])[..., None]
        result = aggregate(a, [0], np.array([[0, 1], [1, 0]]))[0]
        self.assertEqual(result["newly_usable"]["cases"], 1)
        self.assertEqual(result["common_usable"]["cases"], 2)
        self.assertEqual(result["common_usable"]["plugin_disagreements"], 0)
        self.assertEqual(result["rules"]["candidate_plugin"]["alarm_fraction_all_cases"], 0.5)
        self.assertEqual(result["rules"]["candidate_plugin"]["alarm_fraction_converged"], 2 / 3)
        self.assertIsNone(
            aggregate(np.full_like(a, -1), [0], np.array([[0, 1]]))[0]["rules"]["candidate_plugin"][
                "cluster_95_percent"
            ]
        )
