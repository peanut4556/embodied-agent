"""Bounded feedback retries, stop latching and observation-driven replanning."""

import unittest

import numpy as np

from embodied_agent.feedback import FeedbackExecutor
from embodied_agent.imitation import visual_context

HOME = np.array([-1.0, 1.0, 0.0, 0.04, 0.04])


def image(center=130):
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb[114:126, center - 6 : center + 6] = [220, 30, 40]
    return rgb


class PolicyDouble:
    def __init__(self):
        self.metadata = {"fps": 25}
        self.bounds = np.array([[-2.8, 0.4], [0, 2.9], [-2.9, 2.9], [0, 0.04], [0, 0.04]])
        self.contexts = []

    def plan(self, rgb):
        self.contexts.append(visual_context(rgb))
        return np.tile(HOME, (261, 1))


class FeedbackTests(unittest.TestCase):
    def test_contact_failure_retries_once_then_latches_stop(self):
        policy = PolicyDouble()
        executor = FeedbackExecutor(policy, image(), HOME, max_retries=1)
        for _ in range(600):
            command = executor.step(image(), HOME, holding=False)
            if executor.state == "stopped":
                break
        self.assertEqual(executor.reason, "retry limit reached")
        self.assertEqual(executor.retries, 1)
        self.assertEqual(len(policy.contexts), 2)
        np.testing.assert_array_equal(command, executor.step(image(150), HOME, holding=True))
        self.assertEqual(len(policy.contexts), 2)

    def test_visible_motion_replans_from_fresh_image_after_return(self):
        policy = PolicyDouble()
        executor = FeedbackExecutor(policy, image(), HOME)
        for _ in range(12):
            executor.step(image(150), HOME, holding=False)
        self.assertEqual(executor.retries, 1)
        self.assertEqual(policy.contexts, [130 / 320, 150 / 320])
        self.assertEqual(executor.state, "running")

    def test_valid_contact_allows_completion_without_retry(self):
        executor = FeedbackExecutor(PolicyDouble(), image(), HOME)
        for _ in range(263):
            executor.step(image(), HOME, holding=True)
        self.assertEqual(executor.state, "completed")
        self.assertEqual(executor.retries, 0)

    def test_user_stop_and_invalid_observation_never_resume(self):
        for invalid in (False, True):
            with self.subTest(invalid=invalid):
                executor = FeedbackExecutor(PolicyDouble(), image(), HOME)
                joints = np.full(5, np.nan) if invalid else HOME
                held = executor.step(image(), joints, holding=False, stop=not invalid)
                self.assertTrue(np.isfinite(held).all())
                for _ in range(20):
                    np.testing.assert_array_equal(held, executor.step(image(150), HOME, True))
                self.assertEqual(executor.state, "stopped")

    def test_missing_initial_target_stops_before_movement(self):
        executor = FeedbackExecutor(PolicyDouble(), np.zeros((240, 320, 3), dtype=np.uint8), HOME)
        self.assertEqual(executor.state, "stopped")
        np.testing.assert_array_equal(executor.step(image(), HOME, False), HOME)
