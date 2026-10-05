"""Frozen collection schedule and full autonomous takeover-state matching."""

import copy
import json
import unittest
from pathlib import Path

from scripts.collect_contact_recovery import check_plan, verify_branch


class ContactRecoveryTests(unittest.TestCase):
    def test_fixed_schedule_and_parent(self):
        plan = json.loads(Path("config/correction-contact-scenarios.json").read_text())
        check_plan(plan, plan["learner_weights_sha256"])
        for key in ("parent", "time", "position", "split"):
            changed = copy.deepcopy(plan)
            if key == "parent":
                changed["learner_weights_sha256"] = "different"
            elif key == "time":
                changed["cases"][0]["takeover_seconds"] += 0.04
            elif key == "position":
                changed["cases"][0]["x"] = 0.312
            else:
                changed["split"]["validation"] = [0]
            with self.assertRaises(ValueError):
                check_plan(changed, plan["learner_weights_sha256"])

    def test_position_velocity_and_previous_control_must_all_match(self):
        takeover = {"tick": 0, "qpos": [1.0, 2.0], "qvel": [0.1, 0.2], "ctrl": [0.3, 0.4]}
        trace = [
            {
                "tick": 0,
                "simulation_qpos": [1.0, 2.0],
                "simulation_qvel": [0.1, 0.2],
                "previous_command": [0.3, 0.4],
            }
        ]
        verify_branch(takeover, trace)
        for key in ("qpos", "qvel", "ctrl"):
            changed = copy.deepcopy(takeover)
            changed[key][0] += 0.0001
            with self.assertRaises(ValueError):
                verify_branch(changed, trace)

    def test_reference_tick_must_match(self):
        with self.assertRaises(ValueError):
            verify_branch({"tick": 0}, [{"tick": 1}])
