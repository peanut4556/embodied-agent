"""Image-only inputs, held-out split isolation and learned model reconstruction."""

import unittest

import numpy as np

from embodied_agent.imitation import check_split, fit_context_model, predict_context, visual_context


class ImitationTests(unittest.TestCase):
    def test_visual_context_uses_pixels_and_rejects_missing_or_ambiguous_target(self):
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            visual_context(rgb)
        rgb[114:126, 124:136] = [220, 30, 40]
        self.assertAlmostEqual(visual_context(rgb), 130 / 320)
        rgb[114:126, 160:172] = [220, 30, 40]
        with self.assertRaises(ValueError):
            visual_context(rgb)

    def test_polynomial_fit_predicts_unseen_contexts(self):
        train_x = np.linspace(-1, 1, 9)
        train_y = np.stack([np.full((4, 5), 0.2 + 0.3 * x + 0.1 * x * x) for x in train_x])
        coefficients = fit_context_model(train_x, train_y, 2, 0, 1)
        unseen = np.array([-0.65, 0.35])
        expected = np.stack([np.full((4, 5), 0.2 + 0.3 * x + 0.1 * x * x) for x in unseen])
        np.testing.assert_allclose(predict_context(coefficients, unseen, 0, 1), expected, atol=1e-6)

    def test_split_rejects_frame_leakage_and_duplicate_scene(self):
        manifest = {"episodes": [{"initial_cube_x": x} for x in np.linspace(0.28, 0.4, 7)]}
        split = {"train": [0, 1, 2, 3, 4], "validation": [5], "test": [6]}
        check_split(manifest, split)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            check_split(manifest, {**split, "test": [5, 6]})
        manifest["episodes"][6]["initial_cube_x"] = manifest["episodes"][0]["initial_cube_x"]
        with self.assertRaisesRegex(ValueError, "same initial scene"):
            check_split(manifest, split)
