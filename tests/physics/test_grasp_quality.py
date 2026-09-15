"""An end position alone must not certify a pick-and-place demonstration."""

import unittest

import numpy as np

from embodied_agent.grasp_quality import GraspQuality


class QualityTests(unittest.TestCase):
    def test_pushed_or_accidentally_landed_cube_is_not_a_verified_grasp(self):
        quality = GraspQuality(25)
        for _ in range(26):
            quality.observe(False, [0.64, 0, 0.04], [0, 0, 0, 0.04, 0.04], True)
        self.assertTrue(quality.result()["stable_goal"])
        self.assertFalse(quality.result()["verified_pick_place"])
        self.assertEqual(quality.result()["classification"], "goal_without_verified_pick_place")

    def test_sustained_lift_and_intentional_release_qualify(self):
        quality = GraspQuality(25)
        for _ in range(3):
            quality.observe(True, [0.32, 0, 0.20], [0, 0, 0, 0, 0], False)
        quality.observe(True, [0.64, 0, 0.05], [0, 0, 0, 0.04, 0.04], False)
        for _ in range(25):
            quality.observe(False, [0.64, 0, 0.04], [0, 0, 0, 0.04, 0.04], True)
        self.assertTrue(quality.result()["verified_pick_place"])

    def test_brief_contact_or_unintended_drop_does_not_qualify(self):
        for frames in (1, 3):
            quality = GraspQuality(25)
            for _ in range(frames):
                quality.observe(True, [0.32, 0, 0.2], [0, 0, 0, 0, 0], False)
            for _ in range(26):
                quality.observe(False, [0.64, 0, 0.04], [0, 0, 0, 0, 0], True)
            self.assertFalse(quality.result()["verified_pick_place"])

    def test_nonfinite_input_rejected(self):
        with self.assertRaises(ValueError):
            GraspQuality(25).observe(False, [np.nan, 0, 0], [0] * 5, False)
