"""Analytical scoring geometry, including parallel rays and depth discontinuities."""

import unittest

import numpy as np

from embodied_agent.geometry_error import erode, ray_box_depth, stats, surface_distance


class GeometryErrorTests(unittest.TestCase):
    def test_axis_parallel_hits_and_misses(self):
        rays = np.array([[0, 0, -1], [2, 0, -1], [0, 1, 0]])
        depths = ray_box_depth([0, 0, 1], rays, np.zeros(3), np.eye(3), np.full(3, 0.025))
        self.assertAlmostEqual(depths[0], 0.975)
        self.assertTrue(np.isnan(depths[1:]).all())

    def test_rotated_ray_intersection_lands_on_box_surface(self):
        angle = 0.61
        rotation = np.array(
            [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        )
        ray = np.array([[0, 0, -1.0]])
        depth = ray_box_depth([0, 0, 1], ray, np.zeros(3), rotation, np.full(3, 0.025))
        point = np.array([0, 0, 1]) + depth[:, None] * ray
        self.assertLess(surface_distance(point, np.zeros(3), rotation, np.full(3, 0.025))[0], 1e-12)

    def test_surface_distance_counts_inside_and_outside(self):
        distances = surface_distance(
            [[0, 0, 0], [0.025, 0, 0], [0.035, 0, 0]], np.zeros(3), np.eye(3), np.full(3, 0.025)
        )
        np.testing.assert_allclose(distances, [0.025, 0, 0.01])

    def test_border_and_invalid_ray_statistics_remain_explicit(self):
        mask = np.ones((3, 3), dtype=bool)
        self.assertEqual(erode(mask).sum(), 1)
        self.assertEqual(stats([np.nan, 0.001])["count"], 1)
        self.assertIsNone(stats([np.nan])["max_m"])
