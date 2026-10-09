"""Reference-plane residuals are ambiguous under occlusion and calibration error."""

import unittest

import numpy as np

from embodied_agent.reference_plane import check_reference, reference_geometry


class ReferenceMismatchTests(unittest.TestCase):
    def setUp(self):
        self.depth = np.full((480, 640), 1.2)
        self.position = np.array([0.4, 0, 1.2])
        _, roi = reference_geometry(self.depth.shape, self.position, np.eye(3), 45.0)
        self.ids = np.flatnonzero(roi)

    def check(self, depth, position=None):
        return check_reference(
            depth, self.position if position is None else position, np.eye(3), 45.0
        )

    def test_missing_and_foreground_are_different_outcomes(self):
        ids = self.ids[: len(self.ids) // 4]
        missing = self.depth.copy()
        missing.flat[ids] = np.nan
        foreground = self.depth.copy()
        foreground.flat[ids] -= 0.02
        self.assertEqual(self.check(missing)["status"], "unknown")
        self.assertEqual(self.check(foreground)["status"], "alarm")

    def test_depth_height_cancellation_is_a_blind_spot(self):
        pos = self.position + [0, 0, 0.006]
        self.assertEqual(self.check(self.depth + 0.006)["status"], "alarm")
        self.assertEqual(self.check(self.depth, pos)["status"], "alarm")
        result = self.check(self.depth + 0.006, pos)
        self.assertEqual(result["status"], "no_alarm")
        self.assertLess(result["maximum_absolute_residual_m"], 1e-12)
        self.assertFalse(result["clearance_certified"])

    def test_planar_translation_is_not_observable(self):
        for change in ([0.02, 0, 0], [0, 0.02, 0]):
            result = self.check(self.depth, self.position + change)
            self.assertEqual(result["status"], "no_alarm")

    def test_geometry_expected_depth_matches_nominal_camera(self):
        expected, roi = reference_geometry(self.depth.shape, self.position, np.eye(3), 45.0)
        np.testing.assert_allclose(expected[roi], 1.2)
        with self.assertRaises(ValueError):
            reference_geometry((0, 640), self.position, np.eye(3), 45.0)
