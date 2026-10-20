import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from embodied_agent.contact_watchdog import AcquisitionWatchdog, GuardedExecutor


class AcquisitionTests(unittest.TestCase):
    def executor(self):
        data = SimpleNamespace(
            qpos=np.zeros(12),
            ctrl=np.zeros(5),
            site=lambda _: SimpleNamespace(xpos=np.array([0.3, 0, 0.03])),
        )
        world = SimpleNamespace(data=data, holding=False)
        learner = Mock(fps=25, max_ticks=750, reason="")
        learner.step.return_value = np.zeros(5)
        learner.stop.return_value = np.zeros(5)
        return world, learner, None

    def test_deadline_uses_elapsed_ticks_and_fires_once(self):
        for fps in (25, 50):
            monitor = AcquisitionWatchdog(fps, 4)
            for _ in range(4 * fps):
                self.assertIsNone(monitor.observe(False))
            self.assertEqual(monitor.observe(False), "acquisition_timeout")
            self.assertIsNone(monitor.observe(False))

    def test_stable_grasp_disarms_even_after_release(self):
        monitor = AcquisitionWatchdog(25, 4)
        for _ in range(3):
            monitor.observe(True)
        for _ in range(200):
            self.assertIsNone(monitor.observe(False))

    def test_new_contact_at_deadline_is_not_interrupted(self):
        monitor = AcquisitionWatchdog(25, 4)
        for _ in range(100):
            monitor.observe(False)
        self.assertIsNone(monitor.observe(True))
        self.assertEqual(monitor.observe(False), "acquisition_timeout")

    def test_invalid_deadline(self):
        for seconds in (0, -1, float("nan"), float("inf"), 30):
            with self.assertRaises(ValueError):
                AcquisitionWatchdog(25, seconds)
        with self.assertRaises(ValueError):
            AcquisitionWatchdog(10, 4)

    @patch("embodied_agent.contact_watchdog.CorrectionExpert")
    def test_deadline_handoff_and_stop_latch(self, factory):
        world, learner, _ = self.executor()
        world.holding = False
        guard = GuardedExecutor(world, learner, acquisition_seconds=4)
        expert = factory.return_value
        expert.reason = ""
        expert.step.return_value = np.ones(5)
        expert.stop.return_value = np.zeros(5)
        for _ in range(101):
            guard.step(None)
        self.assertEqual(guard.takeover_tick, 100)
        self.assertEqual(learner.step.call_count, 100)
        self.assertEqual(guard.events, [{"tick": 100, "event": "acquisition_timeout"}])
        factory.assert_called_once_with(world, 25, holding_ticks=0)
        guard.step(None, stop=True)
        guard.step(None)
        expert.stop.assert_called_once_with("user stop")
        self.assertEqual(expert.step.call_count, 1)

    @patch("embodied_agent.contact_watchdog.CorrectionExpert")
    def test_stop_and_total_budget_preempt_deadline(self, factory):
        for stop in (True, False):
            world, learner, _ = self.executor()
            world.holding = False
            guard = GuardedExecutor(world, learner, acquisition_seconds=4)
            guard.tick = learner.max_ticks if not stop else 100
            guard.acquisition.tick = 100
            guard.step(None, stop=stop)
            factory.assert_not_called()
            learner.step.assert_not_called()
