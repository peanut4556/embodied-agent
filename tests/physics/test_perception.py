"""Image-only failure checks; no OpenGL context required."""

import unittest

import numpy as np

from embodied_agent.perception import locate_red_cube


class PerceptionTests(unittest.TestCase):
    def detect(self, rgb, depth=None):
        if depth is None:
            depth = np.full(rgb.shape[:2], 1.15)
        return locate_red_cube(rgb, depth, [0.4, 0, 1.2], np.eye(3), 45)

    def test_known_pixel_target_projects_into_world(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[228:253, 267:292] = [220, 35, 40]
        result = self.detect(rgb)
        np.testing.assert_allclose(result["xyz"], [0.3196, -0.001, 0.025], atol=0.002)

    def test_absent_ambiguous_and_invalid_depth_fail(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "found 0"):
            self.detect(rgb)
        rgb[200:225, 250:275] = [220, 35, 40]
        rgb[200:225, 350:375] = [220, 35, 40]
        with self.assertRaisesRegex(ValueError, "found 2"):
            self.detect(rgb)
        with self.assertRaisesRegex(ValueError, "found 0"):
            self.detect(rgb, np.full((480, 640), np.nan))

    def test_partial_target_is_rejected(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[200:225, 250:260] = [220, 35, 40]
        with self.assertRaisesRegex(ValueError, "occluded"):
            self.detect(rgb)
