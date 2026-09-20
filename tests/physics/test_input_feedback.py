"""Feature interventions must stay diagnostic, current-frame and modality-specific."""

import unittest
from types import SimpleNamespace

import numpy as np

from scripts.audit_input_feedback import InterventionSession, intervene


class InputFeedbackTests(unittest.TestCase):
    def test_slices_and_contact_preservation_without_mutating_inputs(self):
        actual = np.arange(12, dtype=np.float32)[None]
        reference = np.arange(12, dtype=np.float32) + 100
        for mode, indices in [
            ("actual", []),
            ("reference_vision", list(range(6))),
            ("reference_joints", list(range(6, 11))),
            ("reference_both", list(range(11))),
        ]:
            result = intervene(actual, reference, mode)
            expected = actual.copy()
            expected[0, indices] = reference[indices]
            np.testing.assert_array_equal(result, expected)
            self.assertEqual(result[0, 11], actual[0, 11])
        np.testing.assert_array_equal(actual, np.arange(12)[None])
        with self.assertRaises(ValueError):
            intervene(actual, reference, "bad")

    def test_current_reference_frame_and_independent_session_state(self):
        captured = []
        policy = SimpleNamespace(
            metadata={"fps": 25},
            bounds=np.array([[-1, 1]] * 5),
            session=lambda: SimpleNamespace(
                predict=lambda x: captured.append(x.copy()) or np.zeros((1, 5))
            ),
        )
        refs = np.stack([np.full(12, t) for t in range(3)])
        a = InterventionSession(policy, refs, "reference_vision")
        b = InterventionSession(policy, refs, "reference_vision")
        actual = np.full((1, 12), 99.0)
        a.predict(actual)
        a.predict(actual)
        b.predict(actual)
        self.assertEqual([x[0, 0] for x in captured], [0, 1, 0])
        self.assertTrue(all(x[0, 11] == 99 for x in captured))
