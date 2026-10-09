"""Shared and one-sided slopes have a fixed, auditable amplitude and support."""

import unittest

import numpy as np

from embodied_agent.depth_ramp import depth_ramp
from embodied_agent.plane_holdout import pixel_partitions


class DepthRampTests(unittest.TestCase):
    def test_shared_slope_endpoints_and_no_input_mutation(self):
        depth = np.ones((32, 32))
        red = np.zeros(depth.shape, dtype=bool)
        red[4:28, 4:28] = True
        changed, delta = depth_ramp(depth, red, 0.002, "shared")
        self.assertAlmostEqual(delta[10, 4], -0.002)
        self.assertAlmostEqual(delta[10, 27], 0.002)
        self.assertTrue(np.all(delta[~red] == 0))
        np.testing.assert_array_equal(depth, np.ones(depth.shape))
        np.testing.assert_allclose(changed, depth + delta)

    def test_sides_partition_shared_ramp_including_guards(self):
        depth = np.ones((32, 32))
        red = np.ones(depth.shape, dtype=bool)
        _, shared = depth_ramp(depth, red, 0.006, "shared")
        _, candidate = depth_ramp(depth, red, 0.006, "candidate")
        _, validation = depth_ramp(depth, red, 0.006, "validation")
        np.testing.assert_allclose(candidate + validation, shared)
        a, b = pixel_partitions(depth.shape)
        self.assertTrue(np.all(candidate[b] == 0))
        self.assertTrue(np.all(validation[a] == 0))
        self.assertFalse(np.any((candidate != 0) & (validation != 0)))

    def test_control_zero_and_sign_symmetry(self):
        depth = np.ones((32, 32))
        red = np.ones(depth.shape, dtype=bool)
        np.testing.assert_array_equal(depth_ramp(depth, red, 0.006, "none")[0], depth)
        positive = depth_ramp(depth, red, 0.006, "shared")[1]
        negative = depth_ramp(depth, red, -0.006, "shared")[1]
        np.testing.assert_array_equal(positive, -negative)
        np.testing.assert_array_equal(depth_ramp(depth, ~red, 0.006, "shared")[0], depth)

    def test_invalid_input_rejected(self):
        for amplitude, mode in ((float("nan"), "shared"), (0.002, "typo")):
            with self.assertRaises(ValueError):
                depth_ramp(np.ones((10, 10)), np.ones((10, 10), dtype=bool), amplitude, mode)
