import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from embodied_agent.contact_watchdog import GuardedExecutor
from scripts.evaluate_contact_faults import observed_contact, validate


class ContactFaultTests(unittest.TestCase):
    def guard(self, holding=False):
        data = SimpleNamespace(
            qpos=np.zeros(12),
            ctrl=np.zeros(5),
            site=lambda _: SimpleNamespace(xpos=np.array([0.3, 0, 0.2])),
        )
        world = SimpleNamespace(data=data, holding=holding)
        learner = Mock(fps=25, max_ticks=750, reason="")
        learner.step.return_value = np.zeros(5)
        learner.stop.return_value = np.zeros(5)
        return world, learner, GuardedExecutor(world, learner, 4, descent_grace=True)

    def test_fault_window_inclusive_start_exclusive_end_and_nominal(self):
        fault = {"name": "drop", "start": 110, "end": 113, "value": False}
        self.assertEqual(
            [observed_contact(True, t, fault) for t in range(109, 114)],
            [True, False, False, False, True],
        )
        fault["value"] = None
        self.assertTrue(observed_contact(True, 111, fault))
        self.assertFalse(observed_contact(False, 111, fault))

    def test_invalid_configuration(self):
        config = json.loads(Path("config/contact-faults.json").read_text())
        validate(config)
        for key, value in [("value", 1), ("start", -1), ("end", 751), ("end", 0), ("start", True)]:
            bad = copy.deepcopy(config)
            bad["faults"][0][key] = value
            with self.assertRaises(ValueError):
                validate(bad)
        config["faults"].append(config["faults"][0])
        with self.assertRaises(ValueError):
            validate(config)

    def test_injection_only_changes_monitor_not_learner_or_world(self):
        world, learner, guard = self.guard(False)
        for _ in range(3):
            guard.step(None, monitor_holding=True)
        self.assertTrue(guard.acquisition.acquired)
        self.assertFalse(world.holding)
        self.assertTrue(all(call.args[2] is False for call in learner.step.call_args_list))

    @patch("embodied_agent.contact_watchdog.CorrectionExpert")
    def test_expert_receives_original_world_on_false_loss(self, factory):
        world, learner, guard = self.guard(True)
        guard.step(None)
        factory.return_value.reason = ""
        factory.return_value.step.return_value = np.ones(5)
        guard.step(None, monitor_holding=False)
        self.assertEqual(guard.events[-1]["event"], "contact_lost")
        self.assertTrue(world.holding)
        factory.assert_called_once_with(world, 25, holding_ticks=0)
        self.assertEqual(learner.step.call_count, 1)

    def test_invalid_contact_is_rejected_and_stop_still_preempts(self):
        _, learner, guard = self.guard()
        for value in (1, "false", [False], float("nan")):
            with self.assertRaises(ValueError):
                guard.step(None, monitor_holding=value)
        learner.step.assert_not_called()
        guard.step(None, stop=True, monitor_holding="invalid")
        self.assertEqual(guard.reason, "user stop")

    def test_default_observation_preserves_existing_behavior(self):
        _, _, guard = self.guard(True)
        for _ in range(3):
            guard.step(None)
        self.assertTrue(guard.acquisition.acquired)
        self.assertIsNone(guard.takeover_tick)
