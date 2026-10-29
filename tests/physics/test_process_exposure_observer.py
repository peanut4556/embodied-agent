import time
import unittest

from embodied_agent.camera_process import CameraProcessSession
from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.process_exposure_observer import ProcessExposureObserver


class Camera:
    def read(self):
        return StampedPosition((0.64, 0, 0.04), time.monotonic())


class SessionStub:
    def __init__(self):
        self.session, self.started, self.status = "first", 10, "idle"
        self.error = None
        self.requests = 0
        self.value = None
        self.fail_restart = False

    def poll(self):
        value, self.value = self.value, None
        return value

    def request(self):
        if self.status not in {"idle", "ready", "error"}:
            return False
        self.requests += 1
        self.status = "pending"
        return True

    def restart(self, factory=None):
        if self.fail_restart:
            self.status = "cleanup_failed"
            raise RuntimeError("not reaped")
        self.session, self.started, self.status = "second", 11, "idle"
        self.value = None

    def close(self):
        self.status = "closed"


class ProcessExposureTests(unittest.TestCase):
    def setUp(self):
        self.now = 10
        self.session = SessionStub()
        self.observer = ProcessExposureObserver(self.session, 0, clock=lambda: self.now)

    def bind(self, task=0):
        self.now = 10 + task
        self.observer.bind_mapping(self.session.session, ExposureClockMap(self.now, task), task)

    def sample(self, task, position=(0.64, 0, 0.04)):
        self.now = 10 + task
        self.session.status = "ready"
        self.session.value = StampedPosition(position, self.now - 0.001)
        return self.observer.step(task)

    def test_restart_discards_continuity_and_requires_new_mapping(self):
        self.bind()
        for task in (0.1, 0.3, 0.5, 0.7, 0.9):
            self.assertEqual(self.sample(task), "pending")
        self.observer.restart()
        self.assertIsNone(self.observer.confirmation)
        requests = self.session.requests
        self.observer.step(1)
        self.assertEqual(self.session.requests, requests)
        with self.assertRaises(ValueError):
            self.observer.bind_mapping("first", ExposureClockMap(11, 1), 1.01)
        with self.assertRaises(ValueError):
            self.observer.bind_mapping("second", ExposureClockMap(10, 0), 1.01)
        self.bind(1.01)
        for task in (1.1, 1.3, 1.5, 1.7, 1.9):
            self.assertEqual(self.sample(task), "pending")
        self.assertEqual(self.sample(2.1), "confirmed_in_tray")

    def test_restart_does_not_extend_deadline_or_poll_after_expiry(self):
        self.bind()
        self.observer.restart()
        self.bind(2.5)
        self.sample(2.6)
        requests = self.session.requests
        self.session.value = StampedPosition((0.64, 0, 0.04), 13)
        self.assertEqual(self.observer.step(3), "unconfirmed")
        self.assertIsNotNone(self.session.value)
        self.assertEqual(self.session.requests, requests)
        self.assertEqual(self.observer.result()["deadline"], 3)
        with self.assertRaises(RuntimeError):
            self.observer.restart()

    def test_external_restart_invalidates_before_consuming_result(self):
        self.bind()
        self.sample(0.2)
        self.session.restart()
        self.session.status = "ready"
        self.session.value = StampedPosition((0.64, 0, 0.04), 11.1)
        requests = self.session.requests
        self.observer.step(1.1)
        self.assertIsNone(self.observer.confirmation)
        self.assertEqual(self.session.requests, requests)

    def test_restart_failure_and_timeout_clear_evidence(self):
        self.bind()
        self.sample(0.2)
        self.session.fail_restart = True
        with self.assertRaises(RuntimeError):
            self.observer.restart()
        self.assertIsNone(self.observer.confirmation)
        self.session.status = "timed_out"
        self.observer.step(0.4)
        self.assertEqual(self.observer.reason, "camera_unavailable")

    def test_stop_latches_without_poll_or_new_request(self):
        self.bind()
        self.sample(0.2)
        requests = self.session.requests
        self.assertEqual(self.observer.step(float("nan"), stop=True), "cancelled")
        self.assertIsNone(self.observer.confirmation)
        self.observer.step(0.4)
        self.assertEqual(self.session.requests, requests)
        with self.assertRaises(RuntimeError):
            self.bind(0.5)
        self.observer.close()
        self.assertEqual(self.session.status, "closed")

    def test_missing_and_rejected_frames_break_continuity(self):
        self.bind()
        for task in (0.1, 0.3, 0.5, 0.7, 0.9):
            self.sample(task)
        self.now = 11
        self.session.status = "ready"
        self.session.value = None
        self.observer.step(1)
        self.assertIsNone(self.observer.confirmation.evidence.since)
        self.sample(1.1)
        self.now = 11.3
        self.session.status = "ready"
        self.session.value = StampedPosition((0.64, 0, 0.04), 10)
        self.observer.step(1.3)
        self.assertIsNotNone(self.observer.confirmation.last_error)
        self.assertIsNone(self.observer.confirmation.evidence.since)

    def test_mapping_drift_future_origin_and_task_regression_rejected(self):
        for mapping in (ExposureClockMap(10, 1), ExposureClockMap(11, 0)):
            with self.assertRaises(ValueError):
                self.observer.bind_mapping("first", mapping, 0)
        self.bind()
        self.observer.step(0.2)
        with self.assertRaises(ValueError):
            self.observer.step(0.1)

    def test_real_spawn_rebind_and_collect_after_restart(self):
        session = CameraProcessSession(Camera, timeout=1)
        origin = time.monotonic()
        observer = ProcessExposureObserver(session, 0)
        try:
            for attempt in range(2):
                deadline = time.monotonic() + 5
                while session.status == "starting" and time.monotonic() < deadline:
                    session.poll()
                    time.sleep(0.005)
                self.assertEqual(session.status, "idle")
                now = time.monotonic()
                observer.bind_mapping(
                    session.session, ExposureClockMap(now, now - origin), now - origin
                )
                observer.step(time.monotonic() - origin)
                deadline = time.monotonic() + 1
                count = len(observer.rows)
                while len(observer.rows) == count and time.monotonic() < deadline:
                    observer.step(time.monotonic() - origin)
                    time.sleep(0.005)
                self.assertGreater(len(observer.rows), count)
                self.assertIsNotNone(observer.confirmation.last_exposure_time)
                if attempt == 0:
                    old = session.session
                    observer.restart()
                    self.assertNotEqual(session.session, old)
                    self.assertIsNone(observer.confirmation)
        finally:
            observer.close()
        self.assertIsNone(session.process)
