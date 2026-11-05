import unittest

import numpy as np

from embodied_agent.reference_perception import locate_references


class ReferenceComponentsTests(unittest.TestCase):
    def images(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[10:22, 10:22] = (0, 0, 255)
        return rgb, np.ones((100, 100))

    def locate(self, rgb, depth, **kwargs):
        return locate_references(rgb, depth, (0, 0, 2), np.eye(3), 90, **kwargs)

    def test_small_remote_glints_no_longer_hide_marker(self):
        rgb, depth = self.images()
        expected = self.locate(rgb, depth)["blue"]
        rgb[70:83, 75] = (0, 0, 255)
        self.assertNotIn("blue", self.locate(rgb, depth))
        diagnostics = {}
        result = self.locate(rgb, depth, isolate_components=True, diagnostics=diagnostics)
        np.testing.assert_allclose(result["blue"], expected)
        self.assertEqual(diagnostics["blue"]["component_sizes"], [144, 13])

    def test_two_supported_candidates_are_rejected_even_nearby(self):
        rgb, depth = self.images()
        rgb[30:35, 30:34] = (0, 0, 255)
        self.assertNotIn("blue", self.locate(rgb, depth, isolate_components=True))

    def test_clipped_second_candidate_does_not_get_discarded(self):
        rgb, depth = self.images()
        rgb[:5, :4] = (0, 0, 255)
        self.assertNotIn("blue", self.locate(rgb, depth, isolate_components=True))

    def test_clipped_single_component_still_rejected(self):
        rgb, depth = self.images()
        rgb[:] = 0
        rgb[:12, 10:22] = (0, 0, 255)
        self.assertNotIn("blue", self.locate(rgb, depth, isolate_components=True))

    def test_small_fragments_do_not_merge_into_marker(self):
        rgb, depth = self.images()
        rgb[:] = 0
        rgb[10:14, 10:14] = (0, 0, 255)
        rgb[20:24, 20:24] = (0, 0, 255)
        self.assertNotIn("blue", self.locate(rgb, depth, isolate_components=True))

    def test_invalid_depth_does_not_supply_support(self):
        rgb, depth = self.images()
        depth[10:22, 10:22] = np.nan
        self.assertNotIn("blue", self.locate(rgb, depth, isolate_components=True))

    def test_connected_spread_is_still_rejected(self):
        rgb, depth = self.images()
        rgb[20, 10:80] = (0, 0, 255)
        self.assertNotIn("blue", self.locate(rgb, depth, isolate_components=True))

    def test_world_projection_does_not_use_catalog(self):
        rgb, depth = self.images()
        a = self.locate(rgb, depth, isolate_components=True)["blue"]
        b = locate_references(rgb, depth, (0.02, 0, 2), np.eye(3), 90, isolate_components=True)[
            "blue"
        ]
        np.testing.assert_allclose(np.array(b) - a, (0.02, 0, 0))
