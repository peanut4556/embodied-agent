import os
import time
import unittest

from test_process_post_confirmation import Session

from embodied_agent.camera_process import CameraProcessSession
from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.process_exposure_observer import ProcessExposureObserver
from embodied_agent.process_post_alert_verification import ProcessPostAlertVerification
from embodied_agent.process_post_confirmation import ProcessPostConfirmation


class ReturningCamera:
    def __init__(self):
        self.start = time.monotonic()

    def read(self):
        now = time.monotonic()
        outside = 1.4 <= now - self.start < 1.95
        return StampedPosition((0.7 if outside else 0.64, 0, 0.04), now)


class BlockAfterAlertCamera:
    def __init__(self):
        self.start = time.monotonic()

    def fail(self):
        time.sleep(10)

    def read(self):
        if time.monotonic() - self.start >= 0.55:
            self.fail()
        return StampedPosition((0.7, 0, 0.04), time.monotonic())


class CrashAfterAlertCamera(BlockAfterAlertCamera):
    def fail(self):
        os._exit(7)


class ProcessPostAlertTests(unittest.TestCase):
    def setUp(self):
        self.now = 17
        self.session = Session()
        self.adapter = ProcessPostAlertVerification(
            self.session,
            {"status": "confirmed_in_tray"},
            1,
            ExposureClockMap(16, 0),
            clock=lambda: self.now,
        )

    def step(self, task, *, frame=False, point=(0.7, 0, 0.04)):
        self.now = 16 + task
        if frame:
            self.session.deliver = True
            self.session.frame = (
                None if point is None else StampedPosition(point, self.now - 0.015625)
            )
        return self.adapter.step(task)

    def alert(self):
        self.step(1.01)
        self.step(1.125, frame=True)
        self.step(1.375, frame=True)
        self.assertEqual(self.adapter.status, "rechecking")

    def test_pending_polls_do_not_break_new_stable_window(self):
        self.alert()
        for tick in range(12, 21):
            task = tick / 8
            self.step(task, frame=True, point=(0.64, 0, 0.04))
            self.step(task + 0.03125)
        self.assertEqual(self.adapter.status, "reconfirmed")
        counters = self.session.polls, self.session.requests
        self.step(4)
        self.assertEqual((self.session.polls, self.session.requests), counters)
        self.adapter.close()
        self.assertFalse(self.adapter.result()["cleanup_required"])

    def test_expiry_is_recorded_once_and_breaks_stability(self):
        self.alert()
        self.step(1.5, frame=True, point=(0.64, 0, 0.04))
        self.step(1.8)
        self.step(1.85)
        self.assertEqual(sum(row.get("event") == "expired" for row in self.adapter.rows), 1)
        self.assertIsNone(self.adapter.watch.recheck.evidence.since)

    def test_deadline_ignores_buffered_frame_and_never_requests_again(self):
        self.alert()
        counts = self.session.polls, self.session.requests
        self.step(4.375, frame=True, point=(0.64, 0, 0.04))
        self.assertEqual(self.adapter.status, "needs_review")
        self.assertEqual((self.session.polls, self.session.requests), counts)

    def test_stop_in_recheck_precedes_bad_clock_and_does_no_ipc(self):
        self.alert()
        counts = self.session.polls, self.session.requests
        self.adapter.step(float("nan"), stop=True)
        self.assertEqual(self.adapter.status, "cancelled")
        self.assertEqual((self.session.polls, self.session.requests), counts)

    def test_shared_ownership_guard_blocks_other_adapter_and_refuses_wrong_cleanup(self):
        with self.assertRaises(RuntimeError):
            ProcessPostConfirmation(
                self.session, {"status": "confirmed_in_tray"}, 1, ExposureClockMap(16, 0)
            )
        self.alert()
        self.session._followup_claim = ("camera", object())
        counts = self.session.polls, self.session.requests
        self.step(1.5)
        self.assertEqual(self.adapter.status, "invalidated")
        self.assertEqual((self.session.polls, self.session.requests), counts)
        with self.assertRaises(RuntimeError):
            self.adapter.close()

    def test_cleanup_failure_preserves_body_error_and_can_retry(self):
        self.alert()
        self.session.fail_close = True
        with self.assertRaises(ExceptionGroup) as caught, self.adapter:
            raise ValueError("body failure")
        self.assertEqual(len(caught.exception.exceptions), 2)
        self.assertEqual(self.adapter.status, "cancelled")
        self.assertEqual(self.adapter.result()["cleanup"]["status"], "failed")
        self.session.fail_close = False
        self.adapter.close()
        self.assertFalse(self.adapter.result()["cleanup_required"])

    def test_clock_drift_during_poll_discards_frame_and_sends_no_request(self):
        self.alert()
        previous_poll = self.session.poll

        def drift():
            result = previous_poll()
            self.now += 0.125
            return result

        self.session.poll = drift
        requests = self.session.requests
        self.step(1.5, frame=True)
        self.assertEqual(self.adapter.status, "invalidated")
        self.assertEqual(self.session.requests, requests)

    def test_request_refusal_and_invalid_task_end_recheck_without_retry(self):
        self.alert()
        self.session.request = lambda: False
        self.step(1.5, frame=True)
        self.assertEqual(self.adapter.result()["reason"], "camera_request_failed")
        self.setUp()
        self.alert()
        counts = self.session.polls, self.session.requests
        self.adapter.step(float("nan"))
        self.assertEqual(self.adapter.result()["reason"], "invalid_task_clock")
        self.assertEqual((self.session.polls, self.session.requests), counts)

    def test_pending_session_cannot_be_claimed(self):
        session = Session()
        session.status = "pending"
        with self.assertRaises(ValueError):
            ProcessPostAlertVerification(
                session, {"status": "confirmed_in_tray"}, 1, ExposureClockMap(16, 0)
            )
        self.assertFalse(hasattr(session, "_followup_claim"))

    def wait_idle(self, session):
        deadline = time.monotonic() + 5
        while session.status == "starting" and time.monotonic() < deadline:
            session.poll()
            time.sleep(0.005)
        self.assertEqual(session.status, "idle")

    def test_spawned_camera_confirmation_alert_recheck_and_cleanup(self):
        session = CameraProcessSession(ReturningCamera, timeout=0.4)
        try:
            self.wait_idle(session)
            origin = time.monotonic()
            observer = ProcessExposureObserver(session, 0)
            observer.bind_mapping(
                session.session, ExposureClockMap(origin, 0), time.monotonic() - origin
            )
            limit = time.monotonic() + 1.35
            while observer.status == "pending" and time.monotonic() < limit:
                observer.step(time.monotonic() - origin)
                time.sleep(0.02)
            self.assertEqual(observer.status, "confirmed_in_tray")
            original = observer.result()
            adapter = ProcessPostAlertVerification.from_confirmed(observer)
            with adapter:
                observer.close()
                self.assertIsNotNone(session.process)
                with self.assertRaises(RuntimeError):
                    ProcessPostAlertVerification.from_confirmed(observer)
                limit = time.monotonic() + 4
                while adapter.status in {"monitoring", "rechecking"} and time.monotonic() < limit:
                    adapter.step(time.monotonic() - origin)
                    time.sleep(0.02)
                self.assertEqual(adapter.status, "reconfirmed", adapter.result())
                self.assertIsNotNone(adapter.result()["alert"])
                self.assertGreater(
                    adapter.result()["recheck"]["last_exposure_time"],
                    adapter.result()["recheck_start"],
                )
            self.assertEqual(observer.result(), original)
            self.assertIsNone(session.process)
            self.assertFalse(adapter.result()["cleanup_required"])
        finally:
            session.close()

    def test_spawned_block_and_crash_after_alert_are_reaped(self):
        for factory in (BlockAfterAlertCamera, CrashAfterAlertCamera):
            with self.subTest(factory=factory):
                session = CameraProcessSession(factory, timeout=0.15)
                try:
                    self.wait_idle(session)
                    origin = time.monotonic()
                    adapter = ProcessPostAlertVerification(
                        session, {"status": "confirmed_in_tray"}, 0, ExposureClockMap(origin, 0)
                    )
                    with adapter:
                        limit = time.monotonic() + 2
                        while (
                            adapter.status in {"monitoring", "rechecking"}
                            and time.monotonic() < limit
                        ):
                            adapter.step(time.monotonic() - origin)
                            time.sleep(0.02)
                        self.assertEqual(adapter.status, "invalidated")
                        self.assertIsNotNone(adapter.result()["alert"])
                        self.assertEqual(adapter.result()["reason"], "camera_unavailable")
                    self.assertIsNone(session.process)
                    if factory is CrashAfterAlertCamera:
                        self.assertEqual(session.last_exitcode, 7)
                finally:
                    session.close()
