import copy
import json
import unittest
from pathlib import Path

from scripts.evaluate_contact_faults import ContactFaultSignal, observed_contact, validate


class ContactBoundaryTests(unittest.TestCase):
    def lag(self, frames):
        return ContactFaultSignal(
            {
                "name": "lag",
                "kind": "release_lag",
                "frames": frames,
                "start": 0,
                "end": 750,
                "value": True,
            }
        )

    def test_release_lag_exact_length_and_no_initial_false_holding(self):
        for frames in (1, 2, 3, 6):
            signal = self.lag(frames)
            self.assertFalse(signal.observe(False, 0, [0, 0, 0, 0.04, 0.04]))
            self.assertTrue(signal.observe(True, 1, [0, 0, 0, 0, 0]))
            actual = [signal.observe(False, t, [0, 0, 0, 0.04, 0.04]) for t in range(2, frames + 3)]
            self.assertEqual(actual, [True] * frames + [False])

    def test_contact_loss_without_open_command_is_not_release_lag(self):
        signal = self.lag(6)
        signal.observe(True, 0, [0, 0, 0, 0, 0])
        self.assertFalse(signal.observe(False, 1, [0, 0, 0, 0, 0]))
        self.assertFalse(signal.observe(False, 2, [0, 0, 0, 0.04, 0.04]))

    def test_window_boundary_and_independent_instances(self):
        signal = self.lag(6)
        signal.fault["end"] = 3
        signal.observe(True, 0, [0, 0, 0, 0, 0])
        self.assertTrue(signal.observe(False, 1, [0, 0, 0, 0.04, 0.04]))
        self.assertTrue(signal.observe(False, 2, [0, 0, 0, 0.04, 0.04]))
        self.assertFalse(signal.observe(False, 3, [0, 0, 0, 0.04, 0.04]))
        self.assertFalse(self.lag(6).observe(False, 1, [0, 0, 0, 0.04, 0.04]))

    def test_closed_only_fault_requires_both_fingers_commanded_closed(self):
        signal = ContactFaultSignal(
            {"name": "closed", "kind": "closed_only", "start": 2, "end": 4, "value": True}
        )
        self.assertFalse(signal.observe(False, 1, [0, 0, 0, 0, 0]))
        self.assertTrue(signal.observe(False, 2, [0, 0, 0, 0, 0]))
        self.assertFalse(signal.observe(False, 2, [0, 0, 0, 0.04, 0]))
        self.assertFalse(signal.observe(False, 2, [0, 0, 0, 0, 0.04]))
        self.assertFalse(signal.observe(False, 4, [0, 0, 0, 0, 0]))
        self.assertTrue(signal.observe(True, 4, [0, 0, 0, 0.04, 0.04]))

    def test_original_override_semantics_are_preserved(self):
        config = json.loads(Path("config/contact-faults.json").read_text())
        for fault in config["faults"]:
            signal = ContactFaultSignal(fault)
            for tick in range(750):
                truth = tick % 2 == 0
                self.assertEqual(
                    signal.observe(truth, tick, [0] * 5), observed_contact(truth, tick, fault)
                )

    def test_new_config_and_bad_lags(self):
        config = json.loads(Path("config/contact-consistency-boundaries.json").read_text())
        validate(config)
        for key, value in [
            ("kind", "unknown"),
            ("frames", True),
            ("frames", 0),
            ("frames", 26),
            ("value", False),
        ]:
            bad = copy.deepcopy(config)
            bad["faults"][1][key] = value
            with self.assertRaises(ValueError):
                validate(bad)
