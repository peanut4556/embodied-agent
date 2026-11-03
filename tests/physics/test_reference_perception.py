import unittest

import numpy as np

from embodied_agent.reference_perception import locate_references


class ReferencePerceptionTests(unittest.TestCase):
    def images(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[10:16, 10:16] = (10, 10, 240)
        rgb[40:46, 40:46] = (240, 240, 10)
        rgb[70:76, 70:76] = (240, 10, 240)
        return rgb, np.ones((100, 100))

    def locate(self, rgb, depth, position=(0, 0, 2)):
        return locate_references(rgb, depth, position, np.eye(3), 90)

    def test_color_identity_and_metric_projection(self):
        rgb, depth = self.images()
        points = self.locate(rgb, depth)
        self.assertEqual(set(points), {"blue", "yellow", "magenta"})
        np.testing.assert_allclose(points["blue"], (-0.74, 0.74, 1), atol=1e-12)
        moved = self.locate(rgb, depth, (0.02, 0, 2))
        for key in points:
            np.testing.assert_allclose(np.array(moved[key]) - points[key], (0.02, 0, 0), atol=1e-12)

    def test_absent_or_invalid_depth_is_not_replaced_with_known_coordinates(self):
        rgb, depth = self.images()
        depth[10:16, 10:16] = np.nan
        self.assertNotIn("blue", self.locate(rgb, depth))
        rgb[:] = 0
        self.assertEqual(self.locate(rgb, depth), {})

    def test_clipped_and_spread_out_detections_are_rejected(self):
        rgb, depth = self.images()
        rgb[85:91, 85:91] = (10, 10, 240)
        self.assertNotIn("blue", self.locate(rgb, depth))
        rgb, depth = self.images()
        rgb[0:6, 0:6] = (10, 10, 240)
        self.assertNotIn("blue", self.locate(rgb, depth))

    def test_mismatched_image_sizes_reject(self):
        rgb, depth = self.images()
        with self.assertRaises(ValueError):
            self.locate(rgb, depth[:50])
