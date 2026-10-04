"""Timeline onset, contact semantics and exact action residual decomposition."""

import unittest

import numpy as np

from embodied_agent.trajectory_divergence import (
    action_decomposition,
    compare,
    events,
    first_sustained,
)


class TrajectoryDivergenceTests(unittest.TestCase):
    def row(self, tick):
        return {
            "tick": tick,
            "command": [0.0, 0.0, 0.0, 0.04, 0.04],
            "joints": [0.0] * 5,
            "block_xyz": [0.302, 0.0, 0.025],
            "finger_contacts": 0,
            "holding": False,
        }

    def test_sustained_returns_onset_and_ignores_isolated_spikes(self):
        self.assertEqual(first_sustained([0, 2, 0, 2, 2, 2], 1), 3)
        self.assertIsNone(first_sustained([1, 1, 1], 1))
        self.assertIsNone(first_sustained([0, 2, 2], 1))
        with self.assertRaises(ValueError):
            first_sustained([float("nan")], 1)

    def test_mixed_units_and_pre_action_state_alignment(self):
        parent = [self.row(i) for i in range(5)]
        candidate = [self.row(i) for i in range(5)]
        for row in candidate:
            row["command"][3] += 0.0002
        for row in candidate[1:]:
            row["joints"][0] += 0.002
        result = compare(parent, candidate)
        self.assertEqual(result["first_sustained_tick"]["command_finger_max_m"], 0)
        self.assertEqual(result["first_sustained_tick"]["joints_arm_max_rad"], 1)
        self.assertIsNone(result["first_sustained_tick"]["object_distance_m"])
        candidate[1]["tick"] = 3
        with self.assertRaises(ValueError):
            compare(parent, candidate)

    def test_contact_not_equivalent_to_holding_or_close_command(self):
        rows = [self.row(i) for i in range(4)]
        rows[1]["finger_contacts"] = 1
        rows[2]["finger_contacts"] = 2
        rows[3]["command"][3:] = [0.005, 0.005]
        result = events(rows, 25)
        self.assertEqual(result["first_finger_contact_tick"], 1)
        self.assertEqual(result["first_bilateral_contact_tick"], 2)
        self.assertEqual(result["first_closing_command_tick"], 3)
        self.assertIsNone(result["first_holding_tick"])

    def test_signed_decomposition_is_exact_and_allows_cancellation(self):
        p, same, own = np.zeros((3, 5)), np.ones((3, 5)), np.full((3, 5), 0.25)
        terms = action_decomposition(p, same, own)
        np.testing.assert_array_equal(
            terms["weight_residual"] + terms["history_shift"], terms["total"]
        )
        self.assertTrue(np.all(terms["history_shift"] < 0))
        with self.assertRaises(ValueError):
            action_decomposition(p, same[:2], own)
