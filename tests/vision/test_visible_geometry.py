"""Visible geometry estimates must not turn occluded space into certified clearance."""

import unittest

import numpy as np

from embodied_agent.visible_geometry import dominant_plane, estimate, point_cloud


class VisibleGeometryTests(unittest.TestCase):
    def test_tilted_plane_with_outliers(self):
        rng = np.random.default_rng(11)
        xy = rng.uniform(-0.025, 0.025, (400, 2))
        points = np.column_stack((xy, 0.05 + xy[:, 0] * np.tan(np.deg2rad(30))))
        points += rng.normal(0, 0.00005, points.shape)
        points = np.vstack((points, rng.uniform(-0.1, 0.1, (80, 3))))
        result = dominant_plane(points)
        self.assertLess(abs(result["tilt_degrees"] - 30), 0.5)
        self.assertGreater(result["inlier_count"], 390)

    def test_degenerate_cloud_rejected(self):
        for points in (np.zeros((50, 3)), np.zeros((2, 3)), np.full((40, 3), np.nan)):
            with self.assertRaises(ValueError):
                dominant_plane(points)

    def test_depth_projection_and_invalid_depth_mask(self):
        depth = np.ones((2, 2))
        depth[0, 0] = 0
        points, valid = point_cloud(np.zeros((2, 2, 3)), depth, [0, 0, 1], np.eye(3), 90)
        self.assertFalse(valid[0, 0])
        np.testing.assert_allclose(points[1, 1], [0.5, -0.5, 0])

    def test_visible_gap_does_not_certify_clearance(self):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[24:40, 20:30] = [220, 30, 30]
        rgb[24:40, 40:45] = [40, 160, 140]
        result = estimate(rgb, np.full((64, 64), 0.9), [0, 0, 1], np.eye(3), 45)
        self.assertGreater(result["visible_x_gap_m"], 0)
        self.assertFalse(result["clearance_certified"])
