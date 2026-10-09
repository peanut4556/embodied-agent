"""Holdout fitting must not reuse candidate pixels or select by normal agreement."""

import unittest

import numpy as np

from embodied_agent.plane_holdout import heldout_groups, pixel_partitions


class PlaneHoldoutTests(unittest.TestCase):
    @staticmethod
    def scene():
        y, x = np.indices((48, 48))
        points = np.stack((x * 0.001, y * 0.001, np.full(x.shape, 0.05)), axis=-1)
        return points, np.ones((48, 48), dtype=bool)

    def test_split_and_neighbor_dependencies_are_disjoint(self):
        a, b = pixel_partitions((48, 48))
        self.assertFalse(np.any(a & b))

        def neighbors(mask):
            p = np.pad(mask, 1)
            return mask | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]

        self.assertFalse(np.any(neighbors(a) & neighbors(b)))

    def test_clean_plane_matches(self):
        points, mask = self.scene()
        result = heldout_groups(points, mask)
        self.assertEqual(len(result["groups"]), 1)
        self.assertEqual(result["groups"][0]["holdout"]["status"], "consistent_candidate")
        self.assertFalse(result["angle_bound_certified"])

    def test_validation_change_cannot_change_candidate(self):
        points, mask = self.scene()
        first = heldout_groups(points, mask)
        _, b = pixel_partitions(mask.shape)
        points[b, 2] += 0.1
        second = heldout_groups(points, mask)
        self.assertEqual(
            {k: v for k, v in first["groups"][0].items() if k != "holdout"},
            {k: v for k, v in second["groups"][0].items() if k != "holdout"},
        )
        self.assertEqual(second["groups"][0]["holdout"]["status"], "unknown")

    def test_disagreeing_normals_rejected(self):
        points, mask = self.scene()
        a, b = pixel_partitions(mask.shape)
        points[a, 2] += 0.2 * (points[a, 0] - 0.0235)
        points[b, 2] -= 0.2 * (points[b, 0] - 0.0235)
        result = heldout_groups(points, mask)
        self.assertEqual(result["groups"][0]["holdout"]["status"], "inconsistent")
        self.assertGreater(result["groups"][0]["holdout"]["angle_degrees"], 5)

    def test_missing_validation_stays_unknown(self):
        points, _ = self.scene()
        a, _ = pixel_partitions(points.shape[:2])
        result = heldout_groups(points, a)
        self.assertEqual(result["groups"][0]["holdout"]["status"], "unknown")
        self.assertEqual(heldout_groups(points, np.zeros(a.shape, dtype=bool))["status"], "unknown")
