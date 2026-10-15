import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from embodied_agent.contact_watchdog import ContactWatchdog, GuardedExecutor


class ContactWatchdogTests(unittest.TestCase):
    def test_no_contact_does_not_trigger(self):
        monitor = ContactWatchdog(25)
        for _ in range(100):
            self.assertIsNone(monitor.observe(False, [0.3, 0.03], np.zeros(5)))

    def test_transient_contact_loss_triggers_once(self):
        monitor = ContactWatchdog(25)
        monitor.observe(True, [0.3, 0.03], np.zeros(5))
        self.assertEqual(monitor.observe(False, [0.3, 0.03], np.zeros(5)), "contact_lost")
        for holding in (True, False, True, False):
            self.assertIsNone(monitor.observe(holding, [0.3, 0.03], np.zeros(5)))

    def test_release_requires_lift_zone_height_and_opening(self):
        for fps in (25, 50):
            for count, tip, opening, expected in [
                (int(0.12 * fps), [0.64, 0.05], 0.04, "intentional_release"),
                (1, [0.64, 0.05], 0.04, "contact_lost"),
                (int(0.12 * fps), [0.3, 0.05], 0.04, "contact_lost"),
                (int(0.12 * fps), [0.64, 0.3], 0.04, "contact_lost"),
                (int(0.12 * fps), [0.64, 0.05], 0.0, "contact_lost"),
            ]:
                with self.subTest(fps=fps, count=count, tip=tip, opening=opening):
                    monitor = ContactWatchdog(fps)
                    for _ in range(count):
                        monitor.observe(True, [0.3, 0.2], np.zeros(5))
                    self.assertEqual(
                        monitor.observe(False, tip, [0, 0, 0, opening, opening]), expected
                    )

    def test_invalid_observations(self):
        with self.assertRaises(ValueError):
            ContactWatchdog(10)
        for tip, command in [
            ([0.3], np.zeros(5)),
            ([np.nan, 0.1], np.zeros(5)),
            ([0.3, 0.1], [0, 0, 0, 0, np.inf]),
        ]:
            with self.assertRaises(ValueError):
                ContactWatchdog(25).observe(False, tip, command)

    def executor(self):
        data = SimpleNamespace(
            qpos=np.zeros(12),
            ctrl=np.zeros(5),
            site=lambda _: SimpleNamespace(xpos=np.array([0.3, 0, 0.03])),
        )
        world = SimpleNamespace(data=data, holding=True)
        learner = Mock(fps=25, max_ticks=750, reason="")
        learner.step.return_value = np.zeros(5)
        learner.stop.return_value = np.zeros(5)
        return world, learner, GuardedExecutor(world, learner)

    @patch("embodied_agent.contact_watchdog.CorrectionExpert")
    def test_handover_stops_policy_and_never_retries(self, factory):
        world, learner, guard = self.executor()
        factory.return_value.step.return_value = np.ones(5)
        factory.return_value.reason = ""
        guard.step(None)
        world.holding = False
        guard.step(None)
        for _ in range(10):
            guard.step(None)
        self.assertEqual(guard.takeover_tick, 1)
        self.assertEqual(learner.step.call_count, 1)
        learner.stop.assert_called_once()
        factory.assert_called_once_with(world, 25, holding_ticks=0)

    @patch("embodied_agent.contact_watchdog.CorrectionExpert")
    def test_stop_preempts_contact_loss(self, factory):
        world, learner, guard = self.executor()
        guard.step(None)
        world.holding = False
        guard.step(None, stop=True)
        guard.step(None)
        factory.assert_not_called()
        self.assertEqual(guard.reason, "user stop")
        self.assertEqual(learner.step.call_count, 1)

    @patch("embodied_agent.contact_watchdog.CorrectionExpert")
    def test_stop_after_takeover_and_timeout_latch(self, factory):
        for timeout in (False, True):
            world, learner, guard = self.executor()
            expert = factory.return_value
            expert.reset_mock()
            expert.reason = ""
            expert.step.return_value = np.zeros(5)
            expert.stop.return_value = np.ones(5)
            guard.step(None)
            world.holding = False
            guard.step(None)
            if timeout:
                guard.tick = learner.max_ticks
            np.testing.assert_array_equal(guard.step(None, stop=not timeout), np.ones(5))
            np.testing.assert_array_equal(guard.step(None), np.ones(5))
            expert.stop.assert_called_once_with("execution timeout" if timeout else "user stop")
            self.assertEqual(expert.step.call_count, 1)


if __name__ == "__main__":
    unittest.main()
