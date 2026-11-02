import unittest

from test_process_post_confirmation import Session

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.process_post_confirmation import ProcessPostConfirmation


class FollowupClockTests(unittest.TestCase):
    def setUp(self):
        self.host = 17
        self.session = Session()
        self.adapter = ProcessPostConfirmation(
            self.session,
            {"status": "confirmed_in_tray"},
            1,
            ExposureClockMap(16, 0),
            clock=lambda: self.host,
        )
        self.adapter.step(1)
        self.host = 17.125
        self.session.frame = StampedPosition((0.64, 0, 0.04), 17.109375)
        self.session.deliver = True
        self.adapter.step(1.125)
        self.assertEqual(self.adapter.watch.evidence_state, "inside_region")

    def assert_invalid_without_ipc(self, task):
        before = self.session.polls, self.session.requests
        self.assertEqual(self.adapter.step(task), "invalidated")
        self.assertEqual((self.session.polls, self.session.requests), before)
        self.assertEqual(self.adapter.watch.evidence_state, "unknown")
        self.assertTrue(self.adapter.result()["cleanup_required"])
        self.assertEqual(
            self.adapter.result()["original_confirmation"]["status"], "confirmed_in_tray"
        )
        self.adapter.close()
        self.assertFalse(self.adapter.result()["cleanup_required"])

    def test_waiting_host_pause_or_drift_cannot_keep_inside_state(self):
        self.assert_invalid_without_ipc(1.25)

    def test_host_regression_is_rejected_even_inside_drift_tolerance(self):
        self.host = 17.124
        self.assert_invalid_without_ipc(1.126)

    def test_nan_infinity_and_clock_read_exception_fail_closed(self):
        for value in (float("nan"), float("inf")):
            self.setUp()
            self.host = value
            self.assert_invalid_without_ipc(1.25)
        self.setUp()

        def broken():
            raise OSError("clock source unavailable")

        self.adapter.clock = broken
        self.assert_invalid_without_ipc(1.25)
        self.assertIn("clock source unavailable", self.adapter.watch.last_error)

    def test_task_pause_regression_and_nonfinite_invalidate_before_raising(self):
        for task in (1.125, 1, float("nan")):
            self.setUp()
            before = self.session.polls
            with self.assertRaises(ValueError):
                self.adapter.step(task)
            self.assertEqual(self.adapter.status, "invalidated")
            self.assertEqual(self.adapter.watch.evidence_state, "unknown")
            self.assertEqual(self.session.polls, before)

    def test_clock_change_during_poll_discards_frame_and_stops_prefetch(self):
        self.host = 17.25
        original_poll = self.session.poll

        def delayed_poll():
            value = original_poll()
            self.host = 17.35
            return value

        self.session.poll = delayed_poll
        self.session.deliver = True
        self.session.frame = StampedPosition((0.7, 0, 0.04), 17.24)
        requests = self.session.requests
        self.assertEqual(self.adapter.step(1.25), "invalidated")
        self.assertEqual(self.session.requests, requests)
        self.assertFalse(self.adapter.result()["outside_detected"])
        self.assertEqual(self.adapter.watch.last_exposure, 1.109375)

    def test_stop_and_expiry_do_not_read_broken_clock(self):
        for stop, task, status in ((True, float("nan"), "cancelled"), (False, 3, "completed")):
            self.setUp()

            def broken():
                self.fail("terminal path read clock")

            self.adapter.clock = broken
            self.assertEqual(self.adapter.step(task, stop=stop), status)

    def test_small_clock_error_and_pending_polls_remain_valid(self):
        self.host = 17.15
        self.adapter.step(1.15625)
        self.assertEqual(self.adapter.status, "monitoring")
        self.assertEqual(self.adapter.watch.evidence_state, "inside_region")
