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

    def test_release_window_is_bounded_and_not_refreshed(self):
        for fps in (25, 50):
            check = ContactConsistency(fps)
            for i in range(check.release_limit - 1):
                self.assertFalse(check.observe(True, [0, 0, 0, 0.04, 0.04], release_context=i == 0))
            self.assertTrue(check.observe(True, [0, 0, 0, 0.04, 0.04], release_context=True))
            check.observe(False, [0, 0, 0, 0.04, 0.04])
            for _ in range(check.minimum - 1):
                self.assertFalse(check.observe(True, [0, 0, 0, 0.04, 0.04]))
            self.assertTrue(check.observe(True, [0, 0, 0, 0.04, 0.04]))

    def test_late_release_context_cannot_extend_existing_contradiction(self):
        check = ContactConsistency(25)
        self.assertFalse(check.observe(True, [0, 0, 0, 0.04, 0.04]))
        self.assertFalse(check.observe(True, [0, 0, 0, 0.04, 0.04], release_context=True))
        self.assertTrue(check.observe(True, [0, 0, 0, 0.04, 0.04], release_context=True))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            ContactConsistency(10)
        for command in ([0], [0, 0, 0, float("nan"), 0]):
            with self.assertRaises(ValueError):
                ContactConsistency(25).observe(True, command)

    @patch("embodied_agent.contact_watchdog.CorrectionExpert")
    def test_delayed_release_requires_qualified_opening_and_no_reclosure(self, factory):
        for fps in (25, 50):
            for enabled, start_tip, reclose, expected in [
                (True, [0.64, 0, 0.06], False, "intentional_release"),
                (False, [0.64, 0, 0.06], False, "contact_lost"),
                (True, [0.3, 0, 0.06], False, "contact_lost"),
                (True, [0.64, 0, 0.2], False, "contact_lost"),
                (True, [0.64, 0, 0.06], True, "contact_lost"),
            ]:
                with self.subTest(fps=fps, enabled=enabled, tip=start_tip, reclose=reclose):
                    factory.return_value.reason = ""
                    factory.return_value.step.return_value = np.zeros(5)
                    learner, guard = self.executor(enabled)
                    learner.fps = fps
                    guard = GuardedExecutor(guard.world, learner, contact_consistency=enabled)
                    data = guard.world.data
                    data.ctrl[3:] = 0
                    for _ in range(int(np.ceil(0.12 * fps))):
                        guard.step(None, monitor_holding=True)
                    data.site = lambda _, tip=start_tip: SimpleNamespace(xpos=np.array(tip))
                    data.ctrl[3:] = 0.04
                    guard.step(None, monitor_holding=True)
                    data.site = lambda _: SimpleNamespace(xpos=np.array([0.64, 0, 0.2]))
                    if reclose:
                        data.ctrl[3:] = 0
                    guard.step(None, monitor_holding=False)
                    self.assertEqual(guard.events[-1]["event"], expected)
                    self.assertEqual(guard.takeover_tick is None, expected == "intentional_release")

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
