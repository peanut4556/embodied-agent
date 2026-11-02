import time
import unittest

from test_camera_process import BlockedCamera, CrashedCamera, FixedCamera

from embodied_agent.camera_process import CameraProcessSession
from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.process_exposure_observer import ProcessExposureObserver
from embodied_agent.process_post_confirmation import ProcessPostConfirmation


class Session:
    def __init__(self):
        self.session, self.status, self.error = "camera", "ready", None
        self.frame, self.deliver = None, False
        self.requests = self.polls = 0
        self.fail_close = False

    def poll(self):
        self.polls += 1
        if self.deliver:
            self.deliver = False
            self.status = "ready"
            return self.frame
        return None

    def request(self):
        self.requests += 1
        self.status = "pending"
        return True

    def close(self):
        if self.fail_close:
            raise RuntimeError("cleanup failed")
        self.status = "closed"


class ProcessFollowupTests(unittest.TestCase):
    def setUp(self):
        self.now = 17
        self.session = Session()
        self.adapter = ProcessPostConfirmation(
            self.session,
            {"status": "confirmed_in_tray"},
            1,
            ExposureClockMap(16, 0),
            clock=lambda: self.now,
        )

    def step(self, task):
        self.now = 16 + task
        return self.adapter.step(task)

    def frame(self, task, point=(0.7, 0, 0.04)):
        self.session.frame = None if point is None else StampedPosition(point, 16 + task - 0.015625)
        self.session.deliver = True
        return self.step(task)

    def test_pending_polls_preserve_continuity_without_duplicate_frames(self):
        self.step(1.01)
        self.frame(1.125)
        self.step(1.15)
        self.step(1.2)
        self.assertEqual(len(self.adapter.rows), 1)
        self.assertEqual(self.session.requests, 2)
        self.frame(1.375)
        self.assertTrue(self.adapter.result()["outside_detected"])
        self.assertEqual(
            self.adapter.result()["original_confirmation"]["status"], "confirmed_in_tray"
        )

    def test_pending_frame_eventually_expires_to_unknown(self):
        self.step(1.01)
        self.frame(1.125, (0.64, 0, 0.04))
        self.step(1.2)
        self.assertEqual(self.adapter.watch.evidence_state, "inside_region")
        self.step(1.4)
        self.assertEqual(self.adapter.watch.evidence_state, "unknown")
        self.assertEqual(self.adapter.watch.reason, "missing_observation")

    def test_missing_result_breaks_outside_sequence(self):
        self.step(1.01)
        self.frame(1.125)
        self.frame(1.25, None)
        self.frame(1.375)
        self.assertFalse(self.adapter.result()["outside_detected"])

    def test_terminal_paths_do_not_poll_or_request_and_close_is_explicit(self):
        for cause in ("stop", "deadline", "session"):
            with self.subTest(cause=cause):
                self.setUp()
                if cause == "session":
                    self.session.session = "replacement"
                self.adapter.step(float("nan") if cause == "stop" else 3, stop=cause == "stop")
                self.assertEqual(self.session.polls, 0)
                self.assertEqual(self.session.requests, 0)
                self.assertTrue(self.adapter.result()["cleanup_required"])
                if cause == "session":
                    with self.assertRaisesRegex(RuntimeError, "ownership changed"):
                        self.adapter.close()
                    self.assertNotEqual(self.session.status, "closed")
                    continue
                self.adapter.close()
                self.assertEqual(self.session.status, "closed")
                self.assertFalse(self.adapter.result()["cleanup_required"])

    def test_cleanup_failure_is_visible_and_retryable(self):
        self.session.fail_close = True
        with self.assertRaises(RuntimeError):
            self.adapter.close()
        self.assertEqual(self.adapter.result()["cleanup"]["status"], "failed")
        self.session.fail_close = False
        self.adapter.close()
        self.assertEqual(self.adapter.result()["cleanup"]["status"], "succeeded")

    def test_request_failure_invalidates_immediately(self):
        self.session.request = lambda: False
        self.assertEqual(self.step(1.01), "invalidated")
        self.assertEqual(self.adapter.watch.reason, "camera_request_failed")

    def test_reject_pending_handoff_and_unconfirmed_source(self):
        self.session.status = "pending"
        with self.assertRaises(ValueError):
            ProcessPostConfirmation(
                self.session, {"status": "confirmed_in_tray"}, 1, ExposureClockMap(16, 0)
            )
        observer = ProcessExposureObserver(self.session, 0)
        with self.assertRaises(ValueError):
            ProcessPostConfirmation.from_confirmed(observer)

    def wait_idle(self, session):
        deadline = time.monotonic() + 5
        while session.status == "starting" and time.monotonic() < deadline:
            session.poll()
            time.sleep(0.005)
        self.assertEqual(session.status, "idle")

    def test_real_success_handoff_collects_new_frame_and_context_reaps(self):
        session = CameraProcessSession(FixedCamera, timeout=0.5)
        adapter = None
        try:
            self.wait_idle(session)
            origin = time.monotonic()
            observer = ProcessExposureObserver(session, 0)
            now = time.monotonic()
            observer.bind_mapping(session.session, ExposureClockMap(origin, 0), now - origin)
            deadline = time.monotonic() + 2.5
            while observer.status == "pending" and time.monotonic() < deadline:
                observer.step(time.monotonic() - origin)
                time.sleep(0.05)
            self.assertEqual(observer.status, "confirmed_in_tray")
            original = observer.result()
            adapter = ProcessPostConfirmation.from_confirmed(observer)
            with self.assertRaises(RuntimeError):
                ProcessPostConfirmation.from_confirmed(observer)
            observer.close()
            self.assertTrue(session.process.is_alive())
            with adapter:
                deadline = time.monotonic() + 0.5
                while not adapter.rows and time.monotonic() < deadline:
                    adapter.step(time.monotonic() - origin)
                    time.sleep(0.005)
                self.assertEqual(adapter.watch.evidence_state, "inside_region")
                self.assertGreater(adapter.watch.last_exposure, observer.last_task)
                adapter.step(float("nan"), stop=True)
            self.assertEqual(observer.result(), original)
            self.assertIsNone(session.process)
            self.assertFalse(adapter.result()["cleanup_required"])
        finally:
            session.close()

    def test_real_block_and_crash_invalidate_then_reap(self):
        for factory in (BlockedCamera, CrashedCamera):
            with self.subTest(factory=factory):
                session = CameraProcessSession(factory, timeout=0.1)
                try:
                    self.wait_idle(session)
                    origin = time.monotonic()
                    adapter = ProcessPostConfirmation(
                        session, {"status": "confirmed_in_tray"}, 0, ExposureClockMap(origin, 0)
                    )
                    with adapter:
                        deadline = time.monotonic() + 1
                        while adapter.status == "monitoring" and time.monotonic() < deadline:
                            adapter.step(time.monotonic() - origin)
                            time.sleep(0.005)
                        self.assertEqual(adapter.status, "invalidated")
                        self.assertEqual(adapter.watch.reason, "camera_unavailable")
                        self.assertIsNotNone(session.process)
                    self.assertIsNone(session.process)
                    if factory is CrashedCamera:
                        self.assertEqual(session.last_exitcode, 7)
                finally:
                    session.close()

    def test_real_expiry_context_reaps_without_reusing_final_frame(self):
        session = CameraProcessSession(FixedCamera, timeout=0.5)
        try:
            self.wait_idle(session)
            origin = time.monotonic()
            adapter = ProcessPostConfirmation(
                session,
                {"status": "confirmed_in_tray"},
                0,
                ExposureClockMap(origin, 0),
                duration=0.05,
            )
            with adapter:
                while adapter.status == "monitoring":
                    adapter.step(time.monotonic() - origin)
                    time.sleep(0.005)
                self.assertEqual(adapter.status, "completed")
                self.assertEqual(adapter.watch.evidence_state, "unknown")
                requests = session.request_id
                adapter.step(time.monotonic() - origin)
                self.assertEqual(session.request_id, requests)
            self.assertIsNone(session.process)
        finally:
            session.close()
