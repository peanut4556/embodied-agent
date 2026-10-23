import unittest

import numpy as np

from scripts.check_release_limit import contradiction_runs


class ReleaseLimitTests(unittest.TestCase):
    def test_counts_reset_on_contact_command_and_inactive_monitor(self):
        signal = np.array([True, True, False, True, True, True, True])
        command = np.tile([0, 0, 0, 0.04, 0.04], (7, 1))
        command[4, 3] = 0
        active = np.array([True] * 6 + [False])
        self.assertEqual(contradiction_runs(signal, command, active), [1, 2, 0, 1, 0, 1, 0])

    def test_empty_and_invalid_trace(self):
        self.assertEqual(
            contradiction_runs(
                np.array([], dtype=bool), np.empty((0, 5)), np.array([], dtype=bool)
            ),
            [],
        )
        for signal, command, active in [
            ([1], [[0] * 5], [True]),
            ([True], [[0] * 4], [True]),
            ([True], [[0, 0, 0, float("nan"), 0]], [True]),
        ]:
            with self.assertRaises(ValueError):
                contradiction_runs(signal, command, active)
