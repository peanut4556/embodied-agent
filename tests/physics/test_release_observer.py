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
        self.assertEqual(len(observer.rows), 16)
        self.assertTrue(all(row["xyz"] is None for row in observer.rows))
        observer.step(4, [{"event": "intentional_release"}], locate)
        self.assertEqual(len(observer.rows), 16)
