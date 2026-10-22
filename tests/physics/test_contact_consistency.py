import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from embodied_agent.contact_watchdog import ContactConsistency, GuardedExecutor
from embodied_agent.physics import PhysicsWorld
from scripts.audit_contact_watchdog import advance


class ContactConsistencyTests(unittest.TestCase):
    def test_requires_consecutive_contradictions_at_both_rates(self):
        for fps in (25, 50):
            check = ContactConsistency(fps)
            for _ in range(check.minimum - 1):
                self.assertFalse(check.observe(True, [0, 0, 0, 0.04, 0.04]))
            self.assertTrue(check.observe(True, [0, 0, 0, 0.04, 0.04]))
            self.assertFalse(check.observe(False, [0, 0, 0, 0.04, 0.04]))
            self.assertFalse(check.observe(True, [0, 0, 0, 0.04, 0.04]))

    def test_closed_or_asymmetric_commands_do_not_claim_sensor_failure(self):
        for fingers in ([0, 0], [0.04, 0], [0, 0.04]):
            check = ContactConsistency(25)
            for _ in range(30):
                self.assertFalse(check.observe(True, [0, 0, 0, *fingers]))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            ContactConsistency(10)
        for command in ([0], [0, 0, 0, float("nan"), 0]):
            with self.assertRaises(ValueError):
                ContactConsistency(25).observe(True, command)

    def executor(self, enabled=True):
        data = SimpleNamespace(
            qpos=np.zeros(12),
            ctrl=np.array([0, 0, 0, 0.04, 0.04]),
            site=lambda _: SimpleNamespace(xpos=np.array([0.3, 0, 0.2])),
        )
        world = SimpleNamespace(data=data, holding=False)
        learner = Mock(fps=25, max_ticks=750, reason="")
        learner.step.return_value = data.ctrl.copy()
        learner.stop.return_value = np.zeros(5)
        return learner, GuardedExecutor(world, learner, 4, True, contact_consistency=enabled)

    @patch("embodied_agent.contact_watchdog.CorrectionExpert")
    def test_fault_stops_without_expert_and_latches(self, factory):
        learner, guard = self.executor()
        for _ in range(3):
            guard.step(None, monitor_holding=True)
        self.assertEqual(guard.events, [{"tick": 2, "event": "contact_inconsistent"}])
        self.assertIsNone(guard.takeover_tick)
        for _ in range(25):
            np.testing.assert_array_equal(guard.step(None, monitor_holding=False), np.zeros(5))
        self.assertEqual(learner.step.call_count, 2)
        learner.stop.assert_called_once()
        factory.assert_not_called()

    def test_disabled_mode_and_user_stop_priority(self):
        _, guard = self.executor(False)
        for _ in range(10):
            guard.step(None, monitor_holding=True)
        self.assertEqual(guard.reason, "")
        _, guard = self.executor()
        for _ in range(2):
            guard.step(None, monitor_holding=True)
        guard.step(None, stop=True, monitor_holding=True)
        self.assertEqual(guard.reason, "user stop")
        self.assertEqual(guard.events, [])

    def test_physical_world_keeps_stop_command_while_simulation_advances(self):
        world = PhysicsWorld()
        try:
            world.data.ctrl[3:] = 0.04
            learner = Mock(fps=25, max_ticks=750, reason="")
            learner.step.return_value = world.data.ctrl.copy()
            learner.stop.side_effect = lambda joints, reason: np.r_[joints[:3], [0.04, 0.04]]
            guard = GuardedExecutor(world, learner, 4, True, contact_consistency=True)
            commands = []
            for _ in range(28):
                command = guard.step(None, monitor_holding=True)
                commands.append(command.copy())
                advance(world, command, 25)
            self.assertEqual(guard.events[0]["tick"], 2)
            self.assertTrue(np.all(np.array(commands[2:]) == commands[2]))
            self.assertEqual(learner.step.call_count, 2)
            self.assertGreater(world.data.time, 1)
        finally:
            world.close()
