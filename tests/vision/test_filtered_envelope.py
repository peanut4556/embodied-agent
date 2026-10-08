"""Combined estimator must retain unknown states and conditional-only semantics."""

import unittest

import numpy as np

from embodied_agent.filtered_envelope import estimate


class FilteredEnvelopeTests(unittest.TestCase):
    def fixture(self):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[20:44, 16:32] = [220, 20, 20]
        return rgb, np.full((64, 64), 0.9), [0, 0, 1], np.eye(3), 10

    def test_total_occlusion_returns_unknown_not_clear(self):
        rgb, *args = self.fixture()
        rgb[:] = 0
        result = estimate(rgb, *args)
        self.assertEqual(result["status"], "unknown")
        self.assertNotIn("bound", result)
        self.assertFalse(result["clearance_certified"])

    def test_missing_wall_keeps_object_bound_but_unknown_gap(self):
        result = estimate(*self.fixture())
        self.assertEqual(result["status"], "object_only")
        self.assertIsNone(result["bound"]["conditional_x_gap_lower_m"])
        self.assertFalse(result["bound"]["conditional_positive_gap"])

    def test_visible_gap_is_reduced_by_bound_not_certified(self):
        rgb, *args = self.fixture()
        rgb[20:44, 48:54] = [40, 160, 140]
        result = estimate(rgb, *args)
        self.assertEqual(result["status"], "conditional_bound")
        self.assertLess(
            result["bound"]["conditional_x_gap_lower_m"], result["visible_filtered_x_gap_m"]
        )
        self.assertFalse(result["clearance_certified"])
        self.assertFalse(result["bound"]["clearance_certified"])
