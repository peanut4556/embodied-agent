import unittest
from unittest.mock import Mock

from embodied_agent.release_observer import ReleaseObserver


class ReleaseObserverTests(unittest.TestCase):
    def test_only_release_starts_and_sampling_is_bounded(self):
        observer = ReleaseObserver()
        locate = Mock(return_value=[0.64, 0, 0.04])
        observer.step(0, [{"event": "contact_lost"}], locate)
        locate.assert_not_called()
        for i in range(101):
            observer.step(i / 25, [{"event": "intentional_release"}], locate)
        self.assertEqual(observer.status, "confirmed_in_tray")
        self.assertEqual(locate.call_count, 6)

    def test_failed_detection_cannot_reuse_old_position(self):
        observer = ReleaseObserver()
        locate = Mock(side_effect=ValueError("occluded"))
        for i in range(76):
            observer.step(i / 25, [{"event": "intentional_release"}], locate)
        self.assertEqual(observer.status, "unconfirmed")
        self.assertEqual(observer.rows[-1]["reason"], "missing_observation")
        self.assertEqual(len(observer.rows), 16)
        self.assertTrue(all(row["xyz"] is None for row in observer.rows))
        observer.step(4, [{"event": "intentional_release"}], locate)
        self.assertEqual(len(observer.rows), 16)


class ReleaseReviewObserverTests(unittest.TestCase):
    def test_opt_in_recheck_preserves_initial_and_default_does_not_retry(self):
        for enabled in (False, True):
            observer = ReleaseObserver(allow_recheck=enabled)
            for i in range(161):
                observer.step(
                    i / 25,
                    [{"event": "intentional_release"}],
                    lambda i=i: None if i < 90 else [0.64, 0, 0.04],
                )
            self.assertEqual(observer.status, "confirmed_in_tray" if enabled else "unconfirmed")
            self.assertEqual(observer.confirmation.status, "unconfirmed")
            self.assertEqual(observer.review is not None, enabled)

    def test_cancel_prevents_camera_and_future_restart(self):
        for enabled in (False, True):
            for stop_tick in (0, 75, 100):
                observer = ReleaseObserver(allow_recheck=enabled)
                locate = Mock(return_value=None)
                for i in range(stop_tick):
                    observer.step(i / 25, [{"event": "intentional_release"}], locate)
                before = locate.call_count
                observer.step(stop_tick / 25, [], locate, stop=True)
                observer.step(7, [{"event": "intentional_release"}], locate)
                self.assertEqual(locate.call_count, before)
                if enabled or stop_tick <= 75:
                    self.assertEqual(observer.status, "cancelled")
