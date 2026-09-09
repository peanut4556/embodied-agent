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
        fps = self.metadata["fps"]
        actions = np.tile(HOME, (round(10.44 * fps), 1))
        actions[3 * fps : 8 * fps, 3:5] = 0
        return actions


class FeedbackTests(unittest.TestCase):
    def transport(self, fps=25, max_retries=2):
        policy = PolicyDouble()
        policy.metadata["fps"] = fps
        executor = FeedbackExecutor(policy, image(), HOME, max_retries=max_retries)
        for _ in range(5 * fps):
            executor.step(image(), HOME, True)
        return executor

    def test_transient_contact_loss_pauses_then_resumes_without_retry(self):
        for fps in (25, 50):
            executor = self.transport(fps)
            index = executor.index
            held = executor.step(image(), HOME, False)
            self.assertEqual(executor.state, "checking_grasp")
            self.assertEqual(executor.index, index)
            np.testing.assert_array_equal(held[3:5], [0, 0])
            executor.step(image(), HOME, True)
            self.assertEqual(executor.state, "running")
            self.assertEqual(executor.index, index + 1)
            self.assertEqual(executor.retries, 0)
            self.assertEqual(executor.slips, 0)

    def test_persistent_transport_loss_replans_from_new_image_at_both_rates(self):
        for fps in (25, 50):
            executor = self.transport(fps)
            for _ in range(round(0.12 * fps)):
                executor.step(image(), HOME, False)
            self.assertEqual(executor.state, "returning")
            self.assertEqual(executor.slips, 1)
            for _ in range(3):
                executor.step(image(150), HOME, False)
            self.assertEqual(executor.state, "running")
            self.assertEqual(executor.policy.contexts, [130 / 320, 150 / 320])
            self.assertEqual(executor.retries, 1)

    def test_intentional_release_does_not_trigger_slip(self):
        executor = self.transport()
        for _ in range(76):
            executor.step(image(), HOME, True)
        for _ in range(100):
            executor.step(image(), HOME, False)
        self.assertEqual(executor.state, "completed")
        self.assertEqual(executor.slips, 0)
        self.assertEqual(executor.retries, 0)

    def test_stop_during_contact_confirmation_latches(self):
        executor = self.transport()
        executor.step(image(), HOME, False)
        command = executor.step(image(), HOME, False, stop=True)
        for _ in range(10):
            np.testing.assert_array_equal(command, executor.step(image(150), HOME, True))
        self.assertEqual(executor.state, "stopped")
        self.assertEqual(executor.retries, 0)

    def test_transport_loss_respects_shared_retry_limit(self):
        executor = self.transport(max_retries=0)
        for _ in range(3):
            executor.step(image(), HOME, False)
        self.assertEqual(executor.state, "stopped")
        self.assertEqual(executor.reason, "retry limit reached")
        self.assertEqual(executor.slips, 1)

    def test_second_slip_after_replan_exhausts_existing_retry_budget(self):
        executor = self.transport(max_retries=1)
        for _ in range(6):  # Confirm first loss and settle at the observation posture.
            executor.step(image(), HOME, False)
        for _ in range(125):
            executor.step(image(), HOME, True)
        for _ in range(3):
            executor.step(image(), HOME, False)
        self.assertEqual(executor.state, "stopped")
        self.assertEqual(executor.reason, "retry limit reached")
        self.assertEqual(executor.retries, 1)
        self.assertEqual(executor.slips, 2)
        self.assertEqual(len(executor.policy.contexts), 2)

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
