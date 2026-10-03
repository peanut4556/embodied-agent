"""Projected one-step state-target derivatives and simulator isolation."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.audit_state_targets import improve, loss


class StateTargetTests(unittest.TestCase):
    def test_projected_step_reduces_known_linear_state_loss(self):
        bounds = np.array([[-1.0, 1.0]] * 5)
        data = SimpleNamespace(ctrl=np.zeros(5))
        action = np.zeros(5)
        target = np.full(5, 0.2)
        with patch(
            "scripts.audit_state_targets.shadow_step", side_effect=lambda m, d, a, s: a.copy()
        ):
            updated, gradient = improve(None, data, action, target, bounds, np.ones(5), 20)
        np.testing.assert_allclose(gradient, np.full(5, -0.04), atol=1e-12)
        np.testing.assert_allclose(updated, np.full(5, 0.02), atol=1e-12)
        self.assertLess(
            loss(updated, target, np.full(5, 2.0)), loss(action, target, np.full(5, 2.0))
        )
        np.testing.assert_array_equal(data.ctrl, np.zeros(5))
        np.testing.assert_array_equal(action, np.zeros(5))

    def test_trust_region_respects_actuator_bounds_and_previous_command_rate(self):
        bounds = np.array([[-1.0, 1.0]] * 5)
        data = SimpleNamespace(ctrl=np.full(5, 0.99))
        with patch(
            "scripts.audit_state_targets.shadow_step", side_effect=lambda m, d, a, s: a.copy()
        ):
            updated, _ = improve(
                None, data, np.full(5, 10.0), np.full(5, 5.0), bounds, np.full(5, 0.01), 20
            )
        self.assertTrue(np.all(updated <= 1.0))
        self.assertTrue(np.all(updated >= 0.98))

    def test_loss_normalizes_arm_and_finger_units(self):
        ranges = np.array([2.0, 2.0, 2.0, 0.04, 0.04])
        self.assertAlmostEqual(loss(ranges * 0.1, np.zeros(5), ranges), 0.01)
