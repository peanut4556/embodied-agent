"""Conditional containment under rotation, missing surfaces and bounded measurement errors."""

import unittest

import numpy as np

from embodied_agent.conservative_geometry import envelope


class ConservativeGeometryTests(unittest.TestCase):
    def test_rotated_cubes_with_sparse_noisy_observations_are_contained(self):
        rng = np.random.default_rng(91)
        corners = np.array(
            [[a, b, c] for a in (-0.025, 0.025) for b in (-0.025, 0.025) for c in (-0.025, 0.025)]
        )
        for _ in range(200):
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            center = rng.uniform(-1, 1, 3)
            truth = corners @ rotation + center
            observed = truth[rng.choice(8, 3, replace=False)] + rng.uniform(-0.003, 0.003, (3, 3))
            result = envelope(observed, 0.05, 0.003)
            self.assertTrue(np.all(np.array(result["aabb_min"]) <= truth.min(0)))
            self.assertTrue(np.all(np.array(result["aabb_max"]) >= truth.max(0)))

    def test_larger_error_cannot_increase_gap(self):
        points = np.array([[0.45, 0, 0.02], [0.53, 0.04, 0.06]])
        small = envelope(points, 0.05, 0.001, 0.54, 0.001)
        large = envelope(points, 0.05, 0.005, 0.54, 0.005)
        self.assertLess(large["conditional_x_gap_lower_m"], small["conditional_x_gap_lower_m"])
        self.assertTrue(np.all(np.array(large["aabb_max"]) >= small["aabb_max"]))

    def test_positive_or_missing_wall_never_certifies_clearance(self):
        points = np.zeros((1, 3))
        result = envelope(points, 0.05, 0.003, 1.0, 0.003)
        self.assertTrue(result["conditional_positive_gap"])
        self.assertFalse(result["clearance_certified"])
        self.assertIsNone(envelope(points, 0.05, 0.003)["conditional_x_gap_lower_m"])

    def test_invalid_inputs_and_contradictory_cloud_rejected(self):
        for points, side, error in [
            ([], 0.05, 0.003),
            ([[float("nan"), 0, 0]], 0.05, 0.003),
            ([[0, 0, 0]], 0, 0.003),
            ([[0, 0, 0]], 0.05, -1),
            ([[0, 0, 0], [1, 1, 1]], 0.05, 0.003),
        ]:
            with self.assertRaises(ValueError):
                envelope(points, side, error)
        with self.assertRaises(ValueError):
            envelope([[0, 0, 0]], 0.05, 0.003, 0.54, None)
