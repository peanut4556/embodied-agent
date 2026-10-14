"""External position budgets, missing dimensions and systematic-bias controls."""

import json
import unittest
from pathlib import Path

import numpy as np

from embodied_agent.external_reference import external_translation_gate
from embodied_agent.reference_fit import predict
from scripts.audit_exposure_uncertainty import seed_plan
from scripts.audit_external_reference import RULES, summarize


class ExternalReferenceTests(unittest.TestCase):
    def test_full_translation_acceptance_and_axis_shifts(self):
        truth = np.zeros(3)
        self.assertTrue(external_translation_gate(truth, truth, [0, 1, 2], 0.0005)["passes"])
        for axis in range(3):
            p = np.zeros(3)
            p[axis] = -0.008
            self.assertFalse(external_translation_gate(p, truth, [0, 1, 2], 0.0005)["passes"])
            # The same error in both estimates is invisible to this check.
            self.assertTrue(external_translation_gate(p, p, [0, 1, 2], 0.0005)["passes"])

    def test_partial_measurement_never_admits_full_translation(self):
        p = [0.0, -0.008, 0.0]
        check = external_translation_gate(p, [0.0], [0], 0.0005)
        self.assertTrue(check["projected_passes"])
        self.assertFalse(check["passes"])
        self.assertFalse(check["full_translation_observed"])

    def test_triangle_budget_and_boundary(self):
        gate = external_translation_gate([0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0, 1, 2], 0.0005)
        self.assertAlmostEqual(gate["error_upper_m"], 0.001 + gate["sensor_radius_m"])
        self.assertTrue(external_translation_gate([0, 0, 0], [0.005, 0, 0], [0, 1, 2], 0)["passes"])
        self.assertFalse(
            external_translation_gate(
                [0, 0, 0], [0.005, 0, 0], [0, 1, 2], 0, systematic_error_bound_m=0.0001
            )["passes"]
        )
        self.assertFalse(gate["calibration_certified"])
        # On the declared sensor-error ball, the triangle upper bound holds for arbitrary fits.
        rng = np.random.default_rng(191000)
        for _ in range(100):
            truth = rng.normal(0, 0.01, 3)
            measurement = truth + rng.normal(0, 0.0005, 3)
            estimate = rng.normal(0, 0.01, 3)
            check = external_translation_gate(estimate, measurement, [0, 1, 2], 0.0005)
            if np.linalg.norm(measurement - truth) <= check["sensor_radius_m"]:
                self.assertLessEqual(
                    np.linalg.norm(estimate - truth), check["error_upper_m"] + 1e-15
                )

    def test_gaussian_sensor_radius_coverage(self):
        radius = external_translation_gate([0, 0, 0], [0, 0, 0], [0, 1, 2], 0.0005)[
            "sensor_radius_m"
        ]
        noise = np.random.default_rng(191001).normal(0, 0.0005, (50000, 3))
        self.assertLessEqual(np.mean(np.linalg.norm(noise, axis=1) > radius), 0.01)

    def test_reference_shift_equivalence_in_all_axes(self):
        b = json.loads(Path("docs/evaluations/redundant-reference-v1.json").read_text())
        planes = [(np.array(p["normal"]), p["offset_m"]) for p in b["planes"]]
        truth = np.array(b["true_parameters"])
        rays = np.array([[0.1, 0.1, -1.0], [-0.1, -0.1, -1.0]])
        for axis in range(3):
            shift = np.zeros(3)
            shift[axis] = 0.008
            moved = [(n, d + n @ shift) for n, d in planes]
            equivalent = truth.copy()
            equivalent[:3] -= shift
            np.testing.assert_allclose(
                predict(rays, moved, truth), predict(rays, planes, equivalent), rtol=0, atol=1e-15
            )

    def test_invalid_inputs_and_fresh_random_streams(self):
        for axes, measurement in [([], []), ([0, 0], [0, 0]), ([3], [0]), ([0], [float("nan")])]:
            with self.assertRaises(ValueError):
                external_translation_gate([0, 0, 0], measurement, axes, 0.0005)
        with self.assertRaises(ValueError):
            external_translation_gate([0, 0, 0], [0, 0, 0], [0, 1, 2], -0.001)
        config = json.loads(Path("config/external-reference.json").read_text())
        seeds = [s for group in seed_plan(config) for batch in group for s in batch]
        seeds += list(range(config["anchor_seed_start"], config["anchor_seed_start"] + 1024))
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertGreater(min(seeds), 141023)

    def test_zero_admission_is_not_zero_risk(self):
        c = {"translation_error_m": 0.008, "accepted": dict.fromkeys(RULES, False)}
        c["accepted"]["prior_gate"] = True
        summary = summarize([{"cases": [c]}], np.array([[0]]), 0.005)
        self.assertEqual(summary["prior_gate"]["accepted_over_limit"], 1)
        self.assertIsNone(summary["independent_xyz"]["cluster_95_percent"]["risk_among_accepted"])


if __name__ == "__main__":
    unittest.main()
