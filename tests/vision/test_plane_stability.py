"""Sensitivity signals must preserve unknowns and cannot certify a stable bias."""

import unittest

import numpy as np

from embodied_agent.plane_stability import normal_stability


class PlaneStabilityTests(unittest.TestCase):
    @staticmethod
    def grid():
        return np.array(
            [
                [x, y, 0.0]
                for x in np.linspace(-0.025, 0.025, 16)
                for y in np.linspace(-0.025, 0.025, 16)
            ]
        )

    def test_clean_plane_is_stable_and_reproducible(self):
        points = self.grid()
        result = normal_stability(points, np.arange(len(points)))
        self.assertEqual(result["status"], "stable_candidate")
        self.assertLess(result["max_change_degrees"], 1e-6)
        self.assertEqual(len(result["views"]), 6)
        self.assertEqual(result, normal_stability(points, np.arange(len(points))))
        self.assertFalse(result["angle_bound_certified"])

    def test_small_spatial_halves_stay_unknown(self):
        points = self.grid()[::8]
        result = normal_stability(points, np.arange(len(points)))
        self.assertEqual(result["status"], "unknown")
        self.assertTrue(any(v["status"] == "unknown" for v in result["views"]))

    def test_bent_surface_is_unstable(self):
        points = self.grid()
        points[:, 2] = 0.4 * np.abs(points[:, 0])
        result = normal_stability(points, np.arange(len(points)))
        self.assertEqual(result["status"], "unstable")
        self.assertGreater(result["max_change_degrees"], 5)

    def test_consistent_tilt_can_pass_without_certification(self):
        points = self.grid()
        points[:, 2] = 0.3 * points[:, 0]
        result = normal_stability(points, np.arange(len(points)))
        self.assertEqual(result["status"], "stable_candidate")
        self.assertFalse(result["angle_bound_certified"])
        self.assertFalse(result["clearance_certified"])

    def test_invalid_indices_or_points_rejected(self):
        points = self.grid()
        for idx in ([0, 0, 1], [-1, 0, 1], [0, 1, 999], [0.0, 1.0, 2.0]):
            with self.assertRaises(ValueError):
                normal_stability(points, idx)
        with self.assertRaises(ValueError):
            normal_stability(np.full((30, 3), np.nan), np.arange(30))
