"""Single-frame measurement pulses preserve other inputs and recurrent history."""

import unittest
from types import SimpleNamespace

import numpy as np

from scripts.audit_joint_sensitivity import PulseSession, pulse, replay


class JointSensitivityTests(unittest.TestCase):
    def policy(self):
        class Session:
            def __init__(self):
                self.memory = 0.0

            def predict(self, x):
                self.memory = 0.5 * self.memory + x[0, 6]
                return np.full((1, 5), self.memory)

        return SimpleNamespace(
            bounds=np.array([[-1.0, 1.0]] * 5), metadata={"fps": 25}, session=Session
        )

    def test_pulse_is_single_channel_and_clips_requested_measurement(self):
        x = np.zeros((1, 12))
        x[0, 6] = 0.99
        used, applied = pulse(x, 0, 0.02, self.policy().bounds)
        self.assertAlmostEqual(applied, 0.01)
        self.assertEqual(used[0, 6], 1)
        self.assertEqual(x[0, 6], 0.99)
        np.testing.assert_array_equal(used[0, 7:], x[0, 7:])
        with self.assertRaises(ValueError):
            pulse(x, 5, 0.01, self.policy().bounds)

    def test_fixed_input_replay_retains_decaying_memory_after_one_pulse(self):
        x = np.zeros((6, 12))
        baseline, _ = replay(self.policy(), x)
        changed, applied = replay(self.policy(), x, 2, 0, 0.01)
        self.assertAlmostEqual(applied, 0.01)
        np.testing.assert_array_equal(changed[:2], baseline[:2])
        np.testing.assert_allclose(changed[2:, 0], [0.01, 0.005, 0.0025, 0.00125])

    def test_physical_wrapper_injects_only_once_and_zero_is_identity(self):
        policy = self.policy()
        for amount in (0.0, 0.01):
            wrapper = PulseSession(policy, 0, amount, tick=2)
            for _ in range(5):
                wrapper.predict(np.zeros((1, 12)))
            self.assertEqual([r["applied"] for r in wrapper.inputs], [0.0, 0.0, amount, 0.0, 0.0])
            self.assertTrue(all(r["actual"][11] == r["used"][11] for r in wrapper.inputs))
