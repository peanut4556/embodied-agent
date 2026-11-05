import unittest

import numpy as np

from embodied_agent.reference_perception import locate_references


class ReferenceDepthCoverageTests(unittest.TestCase):
    def images(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[10:22, 10:22] = (0, 0, 255)
        return rgb, np.ones((100, 100))

    def locate(self, rgb, depth, strict=True):
        return locate_references(
            rgb,
            depth,
            (0, 0, 2),
            np.eye(3),
            90,
            isolate_components=True,
            min_depth_fraction=0.8 if strict else None,
        )

    def test_depthless_second_marker_remains_ambiguous(self):
        rgb, depth = self.images()
        rgb[40:46, 40:46] = (0, 0, 255)
        depth[40:46, 40:46] = np.nan
        self.assertIn("blue", self.locate(rgb, depth, False))
        self.assertNotIn("blue", self.locate(rgb, depth))

    def test_depth_holes_do_not_fragment_color_identity(self):
        rgb, depth = self.images()
        depth[10:22, 15:17] = np.nan
        self.assertNotIn("blue", self.locate(rgb, depth, False))
        self.assertIn("blue", self.locate(rgb, depth))

    def test_low_depth_coverage_rejects_large_valid_fragment(self):
        rgb, depth = self.images()
        depth[10:22, 16:22] = 0
        self.assertIn("blue", self.locate(rgb, depth, False))
        self.assertNotIn("blue", self.locate(rgb, depth))

    def test_color_clipping_cannot_be_hidden_by_invalid_depth(self):
        rgb, depth = self.images()
        rgb[:] = 0
        rgb[:12, 10:22] = (0, 0, 255)
        depth[0] = np.nan
        self.assertIn("blue", self.locate(rgb, depth, False))
        self.assertNotIn("blue", self.locate(rgb, depth))

    def test_projection_uses_only_valid_depth(self):
        rgb, depth = self.images()
        depth[10, 10] = np.inf
        point = self.locate(rgb, depth)["blue"]
        self.assertTrue(np.isfinite(point).all())
        self.assertAlmostEqual(point[2], 1)

    def test_coverage_does_not_certify_correct_depth(self):
        rgb, depth = self.images()
        depth[10:22, 10:22] += 0.1
        self.assertAlmostEqual(self.locate(rgb, depth)["blue"][2], 0.9)

    def test_minimum_absolute_support_still_required(self):
        rgb, depth = self.images()
        rgb[:] = 0
        rgb[10:14, 10:15] = (0, 0, 255)
        depth[10, 10] = np.nan
        self.assertNotIn("blue", self.locate(rgb, depth))

    def test_invalid_coverage_configuration(self):
        rgb, depth = self.images()
        for fraction in (0, -1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                locate_references(
                    rgb,
                    depth,
                    (0, 0, 2),
                    np.eye(3),
                    90,
                    isolate_components=True,
                    min_depth_fraction=fraction,
                )
