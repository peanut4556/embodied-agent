"""Paired external error mechanisms and honest budget accounting."""

import json
import unittest
from pathlib import Path

import numpy as np

from embodied_agent.external_reference import external_translation_gate
from embodied_agent.external_reference_stress import external_observation
from scripts.audit_exposure_uncertainty import seed_plan
from scripts.audit_external_reference_stress import summarize


class ExternalStressTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path("config/external-reference-stress.json").read_text())
        self.scenarios = {s["name"]: s for s in self.config["external_scenarios"]}
        self.truth = np.array([0.001, 0.002, 0.003])

    def test_pairing_and_understated_noise(self):
        normal, _ = external_observation(self.truth, 42, 0, self.scenarios["nominal"])
        high, _ = external_observation(self.truth, 42, 0, self.scenarios["noise_understated"])
        np.testing.assert_allclose(high - self.truth, 4 * (normal - self.truth), atol=1e-18)
        mount, bias = external_observation(self.truth, 42, 0, self.scenarios["mount_3mm"])
        np.testing.assert_allclose(mount - normal, [-0.003, 0, 0], atol=1e-18)
        np.testing.assert_array_equal(bias, [-0.003, 0, 0])

    def test_drift_endpoints_and_latency(self):
        start, b0 = external_observation(self.truth, 42, 0, self.scenarios["drift_6mm"])
        end, b1 = external_observation(self.truth, 42, 1, self.scenarios["drift_6mm"])
        np.testing.assert_array_equal(b0, np.zeros(3))
        np.testing.assert_array_equal(b1, [-0.006, 0, 0])
        np.testing.assert_allclose(end - start, b1, atol=1e-18)
        _, bias = external_observation(self.truth, 42, 0, self.scenarios["latency_1s"])
        np.testing.assert_array_equal(bias, [-0.004, 0, 0])

    def test_frame_rotation_and_range_bound(self):
        s = self.scenarios["frame_rotation"]
        _, bias = external_observation(self.truth, 42, 0, s)
        world = np.array(s["nominal_camera_origin_m"]) + self.truth
        self.assertLess(np.linalg.norm(world), s["world_position_bound_m"])
        self.assertLessEqual(np.linalg.norm(bias), s["budget_bias_m"])
        self.assertLess(bias[0], 0)
        zero = {**s, "rotation_y_rad": 0.0}
        _, bias = external_observation(self.truth, 42, 0, zero)
        np.testing.assert_array_equal(bias, np.zeros(3))

    def test_outlier_bound_and_budget_monotonicity(self):
        s = {**self.scenarios["outliers_1pct"], "outlier_probability": 1.0}
        measured, bias = external_observation(self.truth, 42, 0, s)
        self.assertAlmostEqual(np.linalg.norm(bias), 0.01)
        nominal = external_translation_gate(self.truth, measured, [0, 1, 2], 0.0005)
        bounded = external_translation_gate(
            self.truth, measured, [0, 1, 2], 0.0005, systematic_error_bound_m=0.01
        )
        self.assertAlmostEqual(bounded["error_upper_m"] - nominal["error_upper_m"], 0.01)
        self.assertFalse(bounded["passes"])

    def test_fresh_random_streams_and_invalid_time(self):
        c = self.config
        seeds = [s for group in seed_plan(c) for batch in group for s in batch]
        seeds += list(range(c["anchor_seed_start"], c["anchor_seed_start"] + 1024))
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertGreater(min(seeds), 181023)
        with self.assertRaises(ValueError):
            external_observation(self.truth, 1, 2, self.scenarios["nominal"])
        with self.assertRaises(ValueError):
            external_observation(self.truth, 1, 0, {"kind": "invalid"})

    def test_false_admission_and_empty_risk_denominator(self):
        nominal = {"accepted": True, "sensor_radius_m": 0.0025, "error_upper_m": 0.004}
        budgeted = {"accepted": False, "sensor_radius_m": 0.006, "error_upper_m": 0.007}
        c = {
            "translation_error_m": 0.006,
            "prior_accepted": True,
            "external": {
                "test": {"anchor_error_m": 0.004, "nominal": nominal, "budgeted": budgeted}
            },
        }
        result = summarize([{"cases": [c]}], "test", np.array([[0]]), 0.005)
        self.assertEqual(result["nominal"]["accepted_over_limit"], 1)
        self.assertEqual(result["nominal"]["anchor_outside_radius"], 1)
        self.assertEqual(result["nominal"]["translation_outside_upper"], 1)
        self.assertIsNone(result["budgeted"]["cluster_95_percent"]["risk_among_accepted"])


if __name__ == "__main__":
    unittest.main()
