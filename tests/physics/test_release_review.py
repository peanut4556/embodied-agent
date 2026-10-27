import unittest
from unittest.mock import Mock

from embodied_agent.release_review import ReleaseReview


class ReleaseReviewTests(unittest.TestCase):
    def test_window_boundary_roundoff_does_not_reject_fresh_observation(self):
        review = ReleaseReview(0.3)
        review.step(3.3 - 1e-12, lambda: None)
        self.assertIsNotNone(review.recheck)
        self.assertEqual(review.status, "pending")

    def test_success_and_region_rejection_do_not_retry(self):
        for point, expected in [
            ([0.64, 0, 0.04], "confirmed_in_tray"),
            ([0.8, 0, 0.04], "outside_confirmation_region"),
        ]:
            review = ReleaseReview(0)
            locate = Mock(return_value=point)
            for i in range(41):
                review.step(i / 5, locate)
            self.assertEqual(review.status, expected)
            self.assertIsNone(review.recheck)
            self.assertEqual(locate.call_count, 6 if expected == "confirmed_in_tray" else 16)

    def test_one_retry_preserves_initial_failure(self):
        review = ReleaseReview(0)
        for i in range(41):
            review.step(i / 5, lambda i=i: None if i < 18 else [0.64, 0, 0.04])
        self.assertEqual(review.status, "confirmed_in_tray")
        self.assertEqual(review.initial.status, "unconfirmed")
        self.assertEqual(review.observation_count, 24)

    def test_missing_forever_is_bounded(self):
        review = ReleaseReview(0)
        locate = Mock(side_effect=ValueError("occluded"))
        for i in range(51):
            review.step(i / 5, locate)
        self.assertEqual(review.status, "unconfirmed")
        self.assertEqual(locate.call_count, 31)
        self.assertIsNotNone(review.recheck)

    def test_stop_preempts_camera_at_window_and_budget_boundaries(self):
        for stop_time in (0, 3, 6):
            review = ReleaseReview(0)
            locate = Mock(return_value=None)
            for i in range(int(stop_time * 5)):
                review.step(i / 5, locate)
            before = locate.call_count
            review.step(stop_time, locate, stop=True)
            review.step(10, locate)
            self.assertEqual(review.status, "cancelled")
            self.assertEqual(locate.call_count, before)

    def test_delayed_call_cannot_extend_deadline(self):
        review = ReleaseReview(0)
        review.step(3.8, lambda: None)
        self.assertEqual(review.recheck.start, 3)
        locate = Mock(return_value=[0.64, 0, 0.04])
        review.step(6.1, locate)
        locate.assert_not_called()
        self.assertEqual(review.reason, "overall_timeout")

    def test_no_retry_after_budget_and_invalid_time(self):
        review = ReleaseReview(0)
        review.step(6, lambda: None)
        self.assertIsNone(review.recheck)
        self.assertEqual(review.status, "unconfirmed")
        review = ReleaseReview(0)
        review.step(0, lambda: None)
        for time in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                review.step(time, lambda: None)
