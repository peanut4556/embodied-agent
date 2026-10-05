"""Transient holding must not select the carry-retreat branch."""

import unittest
from types import SimpleNamespace

import numpy as np

from embodied_agent.correction_data import CorrectionExpert


class StableContactExpertTests(unittest.TestCase):
    def world(self, holding=True):
        return SimpleNamespace(
            holding=holding,
            data=SimpleNamespace(qpos=np.zeros(5), ctrl=np.zeros(5)),
            ik=lambda x, z: np.zeros(3),
        )

    def test_transient_contact_opens_for_reacquisition(self):
        for ticks in (0, 1, 2):
            expert = CorrectionExpert(self.world(), 25, ticks)
            self.assertFalse(expert.retreat_grasp)
            np.testing.assert_array_equal(expert.retreat_target[3:], [0.04, 0.04])
            self.assertFalse(expert.contact_decision["preserve_grasp"])

    def test_stable_history_and_current_contact_both_required(self):
        self.assertTrue(CorrectionExpert(self.world(), 25, 3).retreat_grasp)
        self.assertFalse(CorrectionExpert(self.world(False), 25, 20).retreat_grasp)
        self.assertFalse(CorrectionExpert(self.world(), 50, 5).retreat_grasp)
        self.assertTrue(CorrectionExpert(self.world(), 50, 6).retreat_grasp)

    def test_legacy_default_keeps_existing_behavior(self):
        expert = CorrectionExpert(self.world(), 25)
        self.assertTrue(expert.retreat_grasp)
        self.assertEqual(expert.contact_decision["mode"], "legacy")
        np.testing.assert_array_equal(expert.retreat_target[3:], [0.0, 0.0])
