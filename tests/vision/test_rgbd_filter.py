"""Boundary/depth ablations and wall components must use observations only."""

import unittest

import numpy as np

from embodied_agent.rgbd_filter import filter_masks, wall_components


class RGBDFilterTests(unittest.TestCase):
    def test_boundary_removed_independently_of_depth(self):
        rgb = np.zeros((9, 9, 3), dtype=np.uint8)
        rgb[2:7, 2:7] = [220, 20, 20]
        masks = filter_masks(rgb, np.ones((9, 9)))
        self.assertEqual(masks["raw"].sum(), 25)
        self.assertEqual(masks["eroded"].sum(), 9)
        np.testing.assert_array_equal(masks["continuous"], masks["raw"])
        np.testing.assert_array_equal(masks["combined"], masks["eroded"])

    def test_depth_step_and_invalid_neighbors_rejected(self):
        rgb = np.full((9, 9, 3), [220, 20, 20], dtype=np.uint8)
        depth = np.ones((9, 9))
        depth[4, 4] = 1.02
        depth[2, 2] = np.nan
        masks = filter_masks(rgb, depth)
        self.assertFalse(masks["continuous"][4, 4])
        self.assertFalse(masks["continuous"][4, 3])
        self.assertFalse(masks["continuous"][2, 3])
        self.assertTrue(masks["continuous"][6, 6])

    def test_small_target_can_disappear_and_is_not_filled_back(self):
        rgb = np.zeros((5, 5, 3), dtype=np.uint8)
        rgb[2, 2] = [220, 20, 20]
        self.assertEqual(filter_masks(rgb, np.ones((5, 5)))["combined"].sum(), 0)

    def test_components_order_and_minimum_support(self):
        mask = np.zeros((6, 8), dtype=bool)
        mask[1:4, 1:3] = True
        mask[1:4, 5:7] = True
        mask[5, 0] = True
        points = np.zeros((6, 8, 3))
        points[:, 1:3, 0] = 0.7
        points[:, 5:7, 0] = 0.5
        groups = wall_components(mask, points, min_pixels=4)
        self.assertEqual([g["min_x"] for g in groups], [0.5, 0.7])
        self.assertEqual([g["count"] for g in groups], [6, 6])
        mask[2, 3:5] = True
        self.assertEqual(len(wall_components(mask, points, min_pixels=4)), 1)
