import threading
import time
import unittest

from embodied_agent.observation_capture import CapturedReleaseObserver, ObservationCapture


class CaptureTests(unittest.TestCase):
    def wait_done(self, capture):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with capture.lock:
                if capture.job["done"]:
                    return
            time.sleep(0.001)
        self.fail("test capture did not finish")

    def test_cancel_returns_while_driver_is_blocked_and_discards_late_result(self):
        entered, release = threading.Event(), threading.Event()
        calls = []

        def read(cancel):
            calls.append(1)
            entered.set()
            release.wait(2)
            return [0.64, 0, 0.04]

        capture = ObservationCapture(read)
        try:
            capture.sample()
            self.assertTrue(entered.wait(1))
            capture.cancel()
            self.assertTrue(capture.cancel_event.is_set())
            self.assertFalse(release.is_set())
            for _ in range(100):
                self.assertIsNone(capture.sample())
            self.assertEqual(len(calls), 1)
        finally:
            release.set()
        self.wait_done(capture)
        self.assertIsNone(capture.sample())
        self.assertEqual(capture.status, "cancelled")

    def test_timeout_quarantines_call_and_drops_late_success(self):
        entered, release = threading.Event(), threading.Event()
        now, calls = [0.0], []

        def read(cancel):
            calls.append(1)
            entered.set()
            release.wait(2)
            return [0.64, 0, 0.04]

        capture = ObservationCapture(read, timeout=0.15, clock=lambda: now[0])
        try:
            capture.sample()
            self.assertTrue(entered.wait(1))
            now[0] = 0.2
            for _ in range(100):
                self.assertIsNone(capture.sample())
            self.assertEqual(capture.status, "timed_out")
            self.assertEqual(len(calls), 1)
        finally:
            release.set()
        self.wait_done(capture)
        self.assertIsNone(capture.sample())  # Discard, then start a fresh request.
        self.wait_done(capture)
        self.assertEqual(capture.sample(), [0.64, 0, 0.04])
        capture.cancel()

    def test_result_is_copied_and_consumed_once(self):
        value = [0.64, 0, 0.04]
        capture = ObservationCapture(lambda cancel: value)
        capture.sample()
        self.wait_done(capture)
        value[0] = 0.8
        self.assertEqual(capture.sample(), [0.64, 0, 0.04])
        self.assertIsNone(capture.sample())
        capture.cancel()
        self.wait_done(capture)

    def test_invalid_or_failed_camera_is_missing(self):
        def broken(cancel):
            raise OSError("disconnected")

        for read in (broken, lambda cancel: [float("nan"), 0, 0], lambda cancel: [0, 0]):
            capture = ObservationCapture(read)
            capture.sample()
            self.wait_done(capture)
            self.assertIsNone(capture.sample())
            self.assertEqual(capture.status, "error")
            self.assertIsNotNone(capture.error)
            capture.cancel()

    def test_finished_but_stale_result_is_not_used(self):
        now = [0.0]
        capture = ObservationCapture(
            lambda cancel: [0.64, 0, 0.04], timeout=0.15, clock=lambda: now[0]
        )
        capture.sample()
        self.wait_done(capture)
        now[0] = 0.2
        self.assertIsNone(capture.sample())
        capture.cancel()
        self.wait_done(capture)

    def test_observer_stop_propagates_without_waiting_for_capture(self):
        entered, release = threading.Event(), threading.Event()

        def read(cancel):
            entered.set()
            release.wait(2)
            return [0.64, 0, 0.04]

        capture = ObservationCapture(read)
        observer = CapturedReleaseObserver(capture, allow_recheck=True)
        try:
            observer.step(0, [])
            self.assertFalse(entered.is_set())
            observer.step(0.1, [{"event": "intentional_release"}])
            self.assertTrue(entered.wait(1))
            self.assertEqual(observer.step(0.2, [], stop=True), "cancelled")
            self.assertEqual(observer.step(0.3, [{"event": "intentional_release"}]), "cancelled")
            self.assertTrue(capture.closed)
        finally:
            release.set()
        self.wait_done(capture)

    def test_invalid_timeouts_rejected(self):
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                ObservationCapture(lambda cancel: None, timeout=timeout)

    def test_prefetch_supports_continuous_fresh_confirmation(self):
        now = [0.0]
        capture = ObservationCapture(lambda cancel: [0.64, 0, 0.04], clock=lambda: now[0])
        observer = CapturedReleaseObserver(capture, allow_recheck=True)
        for i in range(8):
            now[0] = i / 5
            observer.step(now[0], [{"event": "intentional_release"}])
            if not capture.closed:
                self.wait_done(capture)
        self.assertEqual(observer.observer.status, "confirmed_in_tray")
        self.assertTrue(capture.closed)
