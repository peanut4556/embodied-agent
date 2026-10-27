import threading
import unittest

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.exposure_observer import ExposureObserver
from embodied_agent.observation_capture import ObservationCapture, StampedPosition
from scripts.check_exposure_pipeline import run_case


class ExposureObserverTests(unittest.TestCase):
    def test_camera_pipeline_cases(self):
        for mode, expected in [
            ("stationary", "confirmed_in_tray"),
            ("moving", "outside_confirmation_region"),
            ("cached", "unconfirmed"),
            ("drift", "unconfirmed"),
            ("stop", "cancelled"),
        ]:
            with self.subTest(mode=mode):
                row = run_case(mode)
                self.assertEqual(row["status"], expected)
                if mode == "stop":
                    self.assertEqual(len(row["calls"]), 3)
                if mode == "drift":
                    self.assertTrue(
                        any("drift" in (r["last_error"] or "") for r in row["observations"])
                    )

    def test_strict_mode_required(self):
        capture = ObservationCapture(lambda cancel: None)
        with self.assertRaises(ValueError):
            ExposureObserver(capture, ExposureClockMap(0, 0))
        capture.cancel()

    def test_delivery_after_deadline_never_requests_another_frame(self):
        returned = threading.Event()
        calls = []

        def read(cancel):
            calls.append(1)
            returned.set()
            return StampedPosition((0.64, 0, 0.04), 0)

        capture = ObservationCapture(read, clock=lambda: 0, require_timestamp=True)
        observer = ExposureObserver(capture, ExposureClockMap(0, 0))
        observer.step(0, [{"event": "intentional_release"}])
        self.assertTrue(returned.wait(1))
        observer.step(3.1, [])
        self.assertEqual(observer.status, "unconfirmed")
        self.assertEqual(observer.confirmation.reason, "delivery_deadline_exceeded")
        self.assertTrue(capture.closed)
        self.assertEqual(len(calls), 1)

    def test_stop_before_release_prevents_all_capture(self):
        calls = []
        capture = ObservationCapture(lambda cancel: calls.append(1), require_timestamp=True)
        observer = ExposureObserver(capture, ExposureClockMap(0, 0))
        observer.step(0, [], stop=True)
        observer.step(1, [{"event": "intentional_release"}])
        self.assertEqual(observer.status, "cancelled")
        self.assertEqual(calls, [])

    def test_blocked_capture_cannot_delay_stop_or_restart(self):
        entered, release, returned = threading.Event(), threading.Event(), threading.Event()

        def read(cancel):
            entered.set()
            release.wait(2)
            returned.set()
            return StampedPosition((0.64, 0, 0.04), 0)

        capture = ObservationCapture(read, clock=lambda: 0, require_timestamp=True)
        observer = ExposureObserver(capture, ExposureClockMap(0, 0))
        try:
            observer.step(0, [{"event": "intentional_release"}])
            self.assertTrue(entered.wait(1))
            observer.step(0.1, [], stop=True)
            self.assertFalse(returned.is_set())
            self.assertEqual(observer.status, "cancelled")
        finally:
            release.set()
        self.assertTrue(returned.wait(1))
        self.assertEqual(observer.step(1, [{"event": "intentional_release"}]), "cancelled")
