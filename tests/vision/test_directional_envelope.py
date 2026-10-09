"""Directional constraints must preserve diameter-based containment."""

import unittest

import numpy as np

from embodied_agent.directional_envelope import directional_envelope


class DirectionalEnvelopeTests(unittest.TestCase):
    def test_noisy_rotated_cubes_and_partial_surfaces_remain_contained(self):
        rng = np.random.default_rng(73)
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        for _ in range(100):
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            center = rng.uniform(-1, 1, 3)
            truth = 0.025 * signs @ rotation.T + center
            visible = rng.uniform(-0.025, 0.025, (40, 3))
            visible[:, 2] = 0.025
            observed = visible @ rotation.T + center + rng.uniform(-0.003, 0.003, visible.shape)
            result = directional_envelope(observed)
            self.assertTrue(np.all(truth >= result["aabb_min"]))
            self.assertTrue(np.all(truth <= result["aabb_max"]))
            projected = truth @ np.array(result["directions"]).T
            self.assertTrue(np.all(projected >= result["projection_min"]))
            self.assertTrue(np.all(projected <= result["projection_max"]))
            self.assertTrue(np.all(np.array(result["aabb_min"]) >= result["baseline"]["aabb_min"]))
            self.assertTrue(np.all(np.array(result["aabb_max"]) <= result["baseline"]["aabb_max"]))
            self.assertFalse(result["clearance_certified"])

    def test_directional_slabs_cut_corners_without_claiming_aabb_shrinkage(self):
        t = np.linspace(-0.025, 0.025, 10)
        points = np.array([[x, y, 0] for x in t for y in t])
        c, s = np.cos(0.6), np.sin(0.6)
        points = points @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        result = directional_envelope(points)
        low, high = result["aabb_min"], result["aabb_max"]
        corners = np.array(
            [
                [x, y, z]
                for x in (low[0], high[0])
                for y in (low[1], high[1])
                for z in (low[2], high[2])
            ]
        )
        projected = corners @ np.array(result["directions"]).T
        self.assertTrue(
            np.any((projected < result["projection_min"]) | (projected > result["projection_max"]))
        )
        self.assertEqual(result["aabb_volume_reduction_fraction"], 0)

    def test_larger_error_does_not_shrink_bounds_for_same_cloud(self):
        points = np.random.default_rng(4).uniform(-0.02, 0.02, (50, 3))
        small = directional_envelope(points, point_error_m=0.001)
        large = directional_envelope(points, point_error_m=0.006)
        self.assertTrue(np.all(np.array(large["aabb_min"]) <= small["aabb_min"]))
        self.assertTrue(np.all(np.array(large["aabb_max"]) >= small["aabb_max"]))

    def test_degenerate_cloud_and_missing_evidence(self):
        result = directional_envelope(np.zeros((30, 3)))
        self.assertTrue(np.isfinite(result["aabb_min"]).all())
        self.assertFalse(result["clearance_certified"])
        for points in (np.zeros((29, 3)), np.full((30, 3), np.nan), np.eye(3).repeat(10, axis=0)):
            with self.assertRaises(ValueError):
                directional_envelope(points)
