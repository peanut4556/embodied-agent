"""Swept bounds cover between-waypoint motion, not just endpoint overlap."""

import unittest

import numpy as np

from embodied_agent.gripper_sweep import box_aabb, overlaps, refine_overlap, swept_box


class GripperSweepTests(unittest.TestCase):
    def test_between_endpoints_collision_is_detected(self):
        a = (np.array([-1.0, 0, 0]), np.array([-1.0, 0, 0]))
        b = (np.array([1.0, 0, 0]), np.array([1.0, 0, 0]))
        obstacle = (np.full(3, -0.05), np.full(3, 0.05))
        self.assertFalse(overlaps(a, obstacle))
        self.assertFalse(overlaps(b, obstacle))
        result = swept_box(a, b, np.zeros(5), [2, 0, 0, 0, 0], [1, 0, 0])
        self.assertTrue(overlaps(result, obstacle))

    def test_rotating_box_arc_and_corners_stay_inside_bound(self):
        q0 = np.array([-np.pi / 2, 0, 0, 0, 0])
        q1 = -q0
        size = np.array([0.05, 0.03, 0.02])

        def pose(t):
            c, s = np.cos(t), np.sin(t)
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            return box_aabb([c, s, 0], rot, size)

        bound = swept_box(pose(q0[0]), pose(q1[0]), q0, q1, [1 + np.linalg.norm(size), 0, 0])
        for angle in np.linspace(q0[0], q1[0], 101):
            low, high = pose(angle)
            self.assertTrue(np.all(low >= bound[0]))
            self.assertTrue(np.all(high <= bound[1]))

    def test_stationary_bound_has_no_extra_padding(self):
        box = box_aabb([0, 0, 0.3], np.eye(3), [0.03, 0.065, 0.015])
        result = swept_box(box, box, np.zeros(5), np.zeros(5), [1, 0.6, 0.2])
        np.testing.assert_array_equal(result, box)
        self.assertFalse(overlaps(result, ([-0.1, -0.1, -0.1], [0.1, 0.1, 0])))

    def test_invalid_motion_rejected(self):
        box = (np.zeros(3), np.ones(3))
        with self.assertRaises(ValueError):
            swept_box(box, box, np.zeros(5), np.zeros(5), [-1, 0, 0])


class RefinementTests(unittest.TestCase):
    @staticmethod
    def point(q):
        p = np.array([q[0], 0, 0])
        return p, p

    def test_padding_only_overlap_clears_after_subdivision(self):
        obstacle = ([0.45, 0.1, -0.01], [0.55, 0.2, 0.01])
        result = refine_overlap(self.point, np.zeros(5), [1, 0, 0, 0, 0], [1, 0, 0], obstacle)
        self.assertEqual(result["status"], "clear")
        self.assertGreater(result["deepest_level"], 0)
        self.assertEqual(result["unresolved_intervals"], [])

    def test_endpoint_clear_middle_overlap_retained(self):
        obstacle = ([0.49, -0.01, -0.01], [0.51, 0.01, 0.01])
        result = refine_overlap(self.point, np.zeros(5), [1, 0, 0, 0, 0], [1, 0, 0], obstacle)
        self.assertEqual(result["status"], "static_envelope_overlap")
        self.assertIn(0.5, result["static_witness_fractions"])

    def test_depth_limit_never_passes_unresolved_interval(self):
        obstacle = ([0.45, 0.001, -0.01], [0.55, 0.002, 0.01])
        result = refine_overlap(
            self.point, np.zeros(5), [1, 0, 0, 0, 0], [1, 0, 0], obstacle, max_depth=2
        )
        self.assertEqual(result["status"], "unresolved")
        self.assertLessEqual(result["nodes_visited"], 7)
        self.assertEqual(result["deepest_level"], 2)
        self.assertTrue(result["unresolved_intervals"])

    def test_stationary_touch_is_static_not_padding(self):
        result = refine_overlap(
            self.point, np.zeros(5), np.zeros(5), [1, 0, 0], ([0, 0, 0], [1, 1, 1])
        )
        self.assertEqual(result["status"], "static_envelope_overlap")
        self.assertEqual(result["nodes_visited"], 1)

    def test_invalid_budget_rejected(self):
        for depth in (-1, 13, 1.5, True):
            with self.assertRaises(ValueError):
                refine_overlap(
                    self.point,
                    np.zeros(5),
                    np.zeros(5),
                    [1, 0, 0],
                    ([0, 0, 0], [1, 1, 1]),
                    max_depth=depth,
                )
