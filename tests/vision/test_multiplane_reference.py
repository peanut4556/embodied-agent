"""Ideal layout observability regressions, including exact ambiguity controls."""

import unittest

import numpy as np

from embodied_agent.multiplane_reference import observability, plane_depth, residuals


class MultiplaneReferenceTests(unittest.TestCase):
    def setUp(self):
        self.rays = np.array(
            [[x, y, -1.0] for x in np.linspace(-0.4, 0.4, 25) for y in np.linspace(-0.3, 0.3, 25)]
        )
        self.planes = [
            (np.array([0.0, 0.0, 1.0]), 0.0),
            (np.array([1.0, 0.0, 1.0]) / np.sqrt(2), 0.4 / np.sqrt(2)),
            (np.array([0.0, 1.0, 1.0]) / np.sqrt(2), 0.0),
        ]

    def test_horizontal_depth_and_cancellation(self):
        depth = plane_depth(self.rays, [0.4, 0, 1.2], np.eye(3), *self.planes[0])
        np.testing.assert_allclose(depth, 1.2)
        p = np.zeros(7)
        p[2] = p[6] = 0.006
        np.testing.assert_allclose(residuals(self.rays, self.planes[:1], p), 0, atol=1e-12)
        self.assertGreater(
            np.quantile(np.abs(residuals(self.rays, self.planes[:2], p)[1]), 0.95), 0.003
        )

    def test_two_planes_leave_tangent_translation_unobservable(self):
        p = np.zeros(7)
        p[1] = 0.006
        np.testing.assert_allclose(residuals(self.rays, self.planes[:2], p), 0, atol=1e-12)
        self.assertGreater(np.max(np.abs(residuals(self.rays, self.planes, p)[2])), 0.003)

    def test_rank_improves_three_six_seven(self):
        for count, rank in ((1, 3), (2, 6), (3, 7)):
            self.assertEqual(observability(self.rays, self.planes[:count])["rank"], rank)

    def test_duplicate_plane_adds_no_independent_constraints(self):
        self.assertEqual(observability(self.rays, [self.planes[0]] * 3)["rank"], 3)

    def test_zero_perturbation_is_zero_on_every_plane(self):
        np.testing.assert_array_equal(
            residuals(self.rays, self.planes, np.zeros(7)), np.zeros((3, 625))
        )

    def test_invalid_geometry_rejected(self):
        with self.assertRaises(ValueError):
            plane_depth([[1.0, 0, 0]], [0.4, 0, 1.2], np.eye(3), *self.planes[0])
        with self.assertRaises(ValueError):
            plane_depth(self.rays, [0.4, 0, -1.2], np.eye(3), *self.planes[0])
        with self.assertRaises(ValueError):
            residuals(self.rays, self.planes, [0] * 6)
