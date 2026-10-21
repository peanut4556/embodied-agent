import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from embodied_agent.contact_watchdog import DescentAcquisitionWatchdog, GuardedExecutor
from scripts.evaluate_recovery_positives import ReplayLearner


class DescentTests(unittest.TestCase):
    def test_stall_still_fires_at_original_deadline(self):
        for fps in (25, 50):
            monitor = DescentAcquisitionWatchdog(fps, 4)
            for _ in range(4 * fps):
                self.assertIsNone(monitor.observe(False, [0.3, 0.2]))
            self.assertEqual(monitor.observe(False, [0.3, 0.2]), "acquisition_timeout")
            self.assertIsNone(monitor.observe(False, [0.3, 0.2]))

    def test_descent_and_closure_wait_allow_stable_contact(self):
        for fps in (25, 50):
            monitor = DescentAcquisitionWatchdog(fps, 4)
            for tick in range(4 * fps + 1):
                self.assertIsNone(monitor.observe(False, [0.3, 0.3 - tick * 0.02 / fps]))
            for _ in range(round(0.6 * fps)):
                self.assertIsNone(monitor.observe(False, [0.3, 0.22]))
            for _ in range(monitor.minimum):
                self.assertIsNone(monitor.observe(True, [0.3, 0.22]))
            for _ in range(10 * fps):
                self.assertIsNone(monitor.observe(False, [0.3, 0.22]))
            self.assertTrue(monitor.acquired)
            self.assertGreater(monitor.deferred_ticks, 0)

    def test_settling_lease_expires(self):
        monitor = DescentAcquisitionWatchdog(25, 4)
        for tick in range(101):
            monitor.observe(False, [0.3, 0.3 - tick * 0.0008])
        events = [monitor.observe(False, [0.3, 0.22]) for _ in range(60)]
        self.assertEqual(events.count("acquisition_timeout"), 1)

    def test_hard_limit_even_with_continuing_progress(self):
        for fps in (25, 50):
            monitor = DescentAcquisitionWatchdog(fps, 4)
            for tick in range(8 * fps):
                self.assertIsNone(monitor.observe(False, [0.3, 0.3 - tick * 0.02 / fps]))
            self.assertEqual(monitor.observe(False, [0.3, 0.14]), "acquisition_timeout")

    def test_wrong_region_upward_or_low_motion_cannot_extend(self):
        for x, speed, height in [
            (0.6, 0.02, 0.3),
            (0.3, -0.02, 0.3),
            (0.3, 0.001, 0.3),
            (0.3, 0.02, 0.1),
        ]:
            monitor = DescentAcquisitionWatchdog(25, 4)
            for tick in range(100):
                monitor.observe(False, [x, height - tick * speed / 25])
            self.assertEqual(monitor.observe(False, [x, height - 4 * speed]), "acquisition_timeout")

    def test_invalid_tip_and_deadline(self):
        with self.assertRaises(ValueError):
            DescentAcquisitionWatchdog(25, 8)
        monitor = DescentAcquisitionWatchdog(25, 4)
        for tip in ([0.3], [0.3, float("nan")]):
            with self.assertRaises(ValueError):
                monitor.observe(False, tip)
        self.assertEqual(monitor.tick, 0)

    @patch("embodied_agent.contact_watchdog.CorrectionExpert")
    def test_stop_and_contact_loss_override_grace(self, factory):
        for stop in (False, True):
            data = SimpleNamespace(
                qpos=np.zeros(12),
                ctrl=np.zeros(5),
                site=lambda _: SimpleNamespace(xpos=np.array([0.3, 0, 0.2])),
            )
            world = SimpleNamespace(data=data, holding=False)
            learner = Mock(fps=25, max_ticks=750, reason="")
            learner.step.return_value = np.zeros(5)
            learner.stop.return_value = np.zeros(5)
            guard = GuardedExecutor(world, learner, 4, descent_grace=True)
            guard.watchdog.previous_holding = True
            guard.acquisition.tick = 110
            guard.acquisition.last_progress = 109
            factory.reset_mock()
            factory.return_value.reason = ""
            factory.return_value.step.return_value = np.zeros(5)
            guard.step(None, stop=stop)
            if stop:
                factory.assert_not_called()
                self.assertEqual(guard.reason, "user stop")
            else:
                factory.assert_called_once()
                self.assertEqual(guard.events[0]["event"], "contact_lost")

    def test_diagnostic_replay_interpolates_and_holds_final_command(self):
        learner = ReplayLearner(np.array([np.zeros(5), np.ones(5)]), 2)
        for expected in (0, 0.5, 1, 1):
            np.testing.assert_array_equal(learner.step(None, None, False), np.full(5, expected))
