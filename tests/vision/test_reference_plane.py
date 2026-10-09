"""Known-plane checking preserves missing evidence and exposes local-bias blindness."""

import unittest

import numpy as np

from embodied_agent.reference_plane import check_reference


class ReferencePlaneTests(unittest.TestCase):
    def check(self, depth):
        return check_reference(depth, [0.4, 0, 1.2], np.eye(3), 45.0)

    def test_control_and_fixed_threshold(self):
        depth = np.full((480, 640), 1.2)
        self.assertEqual(self.check(depth)["status"], "no_alarm")
        self.assertEqual(self.check(depth + 0.002)["status"], "no_alarm")
        for delta in (-0.006, 0.006):
            result = self.check(depth + delta)
            self.assertEqual(result["status"], "alarm")
            self.assertAlmostEqual(result["absolute_residual_p95_m"], 0.006)
            self.assertFalse(result["clearance_certified"])

    def test_missing_depth_and_out_of_view_are_unknown(self):
        self.assertEqual(self.check(np.full((480, 640), np.nan))["status"], "unknown")
        self.assertEqual(
            check_reference(np.ones((480, 640)), [10, 0, 1.2], np.eye(3), 45)["status"], "unknown"
        )

    def test_local_error_outside_reference_cannot_be_seen(self):
        depth = np.full((480, 640), 1.2)
        depth[220:260, 300:340] += 0.05
        self.assertEqual(self.check(depth)["status"], "no_alarm")

    def test_large_reference_error_is_not_filtered_out(self):
        result = self.check(np.full((480, 640), 1.4))
        self.assertEqual(result["status"], "alarm")
        self.assertEqual(result["reference_pixels"], result["valid_pixels"])
