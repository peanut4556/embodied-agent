"""Diagnostic rotated extents and unchanged acceptance on saved RGB-D evidence."""

import unittest

import numpy as np

from embodied_agent.perception import locate_red_cube, oriented_extent


class PerceptionDiagnosticTests(unittest.TestCase):
    def test_rotated_square_has_same_oriented_size(self):
        points = np.array([[-0.025, -0.025], [-0.025, 0.025], [0.025, -0.025], [0.025, 0.025]])
        angle = np.deg2rad(25)
        points = points @ np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        self.assertTrue(np.all(np.ptp(points, axis=0) > 0.060))
        np.testing.assert_allclose(oriented_extent(points)["extent_m"], [0.05, 0.05], atol=1e-10)

    def test_bad_cloud_rejected(self):
        for points in ([[0, 0]], [[0, 0], [1, 1], [float("nan"), 2]]):
            with self.assertRaises(ValueError):
                oriented_extent(points)

    def test_optional_diagnostics_do_not_change_missing_target_error(self):
        rgb, depth = np.zeros((20, 20, 3)), np.ones((20, 20))
        diagnostic = {}
        for kwargs in ({}, {"diagnostics": diagnostic}):
            with self.assertRaisesRegex(ValueError, "found 0"):
                locate_red_cube(rgb, depth, [0, 0, 1], np.eye(3), 45, **kwargs)
        self.assertEqual(diagnostic["red_pixels"], 0)
