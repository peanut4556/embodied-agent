import time
import unittest

from test_camera_process import CrashedCamera
from test_process_exposure_observer import Camera, SessionStub

from embodied_agent.camera_process import CameraProcessSession
from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.recovery_exposure_observer import RecoveryExposureObserver


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.task = 0.5
        self.session = SessionStub()
        self.observer = RecoveryExposureObserver(
            self.session, 0, task_clock=lambda: self.task, clock=lambda: 10 + self.task
        )

    def fault(self, status="timed_out"):
        self.session.status, self.session.error = status, "driver unavailable"

    def test_fault_restart_mapping_and_new_evidence_restore(self):
        self.task = 1.01
        self.fault()
        self.assertTrue(self.observer.restart())
        self.assertEqual(self.observer.attempts[0]["outcome"], "awaiting_clock_mapping")
        self.task = 1.02
        self.observer.bind_mapping("second", ExposureClockMap(11.02, 1.02), self.task)
        for self.task in (1.1, 1.3, 1.5, 1.7, 1.9, 2.1):
            self.session.status = "ready"
            self.session.value = StampedPosition((0.64, 0, 0.04), 10 + self.task - 0.001)
            self.observer.step(self.task)
        self.assertEqual(self.observer.status, "confirmed_in_tray")
        record = self.observer.result()["recovery_attempts"][0]
        self.assertEqual(record["outcome"], "observation_restored")
        self.assertEqual(record["fault"], "timed_out")
        self.assertEqual(record["old_session"], "first")
        self.assertEqual(record["new_session"], "second")
        record["outcome"] = "mutated"
        self.assertEqual(self.observer.attempts[0]["outcome"], "observation_restored")

    def test_repeated_failure_reaches_limit(self):
        self.fault()
        self.assertTrue(self.observer.restart())
        self.fault("failed")
        self.observer.step(0.6)
        self.assertEqual(self.observer.attempts[0]["outcome"], "camera_failed_again")
        self.task = 0.7
        self.assertFalse(self.observer.restart())
        self.assertEqual(self.observer.reason, "restart_limit_reached")
        self.assertEqual(len(self.observer.attempts), 1)
        self.task = 0.8
        self.assertFalse(self.observer.restart())

    def test_healthy_and_starting_sessions_do_not_consume_attempts(self):
        for self.task, status in enumerate(("idle", "pending", "starting")):
            self.task *= 0.1
            self.session.status = status
            self.assertFalse(self.observer.restart())
        self.assertEqual(self.observer.attempts, [])
        self.assertEqual(self.observer.status, "pending")

    def test_insufficient_budget_and_exact_deadline_do_not_spawn(self):
        for task, reason in (
            (1.75, "insufficient_recovery_budget"),
            (3, "delivery_deadline_reached"),
        ):
            with self.subTest(task=task):
                self.setUp()
                self.task = task
                self.fault()
                self.assertFalse(self.observer.restart())
                self.assertEqual(self.session.session, "first")
                self.assertEqual(self.observer.reason, reason)
                self.assertEqual(self.observer.attempts, [])

    def test_restart_failure_is_counted_and_terminal(self):
        self.fault()
        self.session.fail_restart = True
        self.assertFalse(self.observer.restart())
        self.assertEqual(self.observer.status, "unconfirmed")
        self.assertEqual(self.observer.attempts[0]["outcome"], "restart_failed")
        self.assertIn("not reaped", self.observer.attempts[0]["restart_error"])
        self.assertIsNone(self.observer.confirmation)

    def test_cleanup_failure_does_not_attempt_new_spawn(self):
        self.fault("cleanup_failed")
        self.assertFalse(self.observer.restart())
        self.assertEqual(self.observer.reason, "camera_cleanup_failed")
        self.assertEqual(self.observer.attempts, [])

    def test_restart_elapsed_time_and_clock_regression_fail_closed(self):
        for end, reason in (
            (3.1, "delivery_deadline_reached"),
            (0.1, "invalid_recovery_clock"),
            (float("nan"), "invalid_recovery_clock"),
        ):
            with self.subTest(end=end):
                self.setUp()
                original = self.session.restart

                def delayed(factory=None, original=original, end=end):
                    original(factory)
                    self.task = end

                self.session.restart = delayed
                self.fault()
                self.assertFalse(self.observer.restart())
                self.assertEqual(self.observer.reason, reason)
                self.assertFalse(self.observer.result()["mapping_bound"])
                self.observer.close()
                self.assertEqual(self.session.status, "closed")

    def test_stop_and_timeout_waiting_for_mapping(self):
        self.fault()
        self.observer.restart()
        self.assertEqual(self.observer.step(float("nan"), stop=True), "cancelled")
        self.assertFalse(self.observer.restart())
        self.assertEqual(self.observer.attempts[0]["outcome"], "cancelled")
        self.setUp()
        self.fault()
        self.observer.restart()
        self.observer.step(3)
        self.assertEqual(self.observer.status, "unconfirmed")
        self.assertEqual(self.observer.attempts[0]["outcome"], "unconfirmed")

    def test_invalid_options_and_disabled_recovery(self):
        for options in (
            {"max_restarts": True},
            {"max_restarts": -1},
            {"max_restarts": 0.5},
            {"min_remaining": 0.9},
            {"min_remaining": float("nan")},
        ):
            with self.assertRaises(ValueError):
                RecoveryExposureObserver(self.session, 0, task_clock=lambda: 0, **options)
        observer = RecoveryExposureObserver(self.session, 0, task_clock=lambda: 0.1, max_restarts=0)
        self.fault()
        self.assertFalse(observer.restart())
        self.assertEqual(observer.reason, "restart_limit_reached")

    def test_real_process_crash_recovery_requires_fresh_observation(self):
        session = CameraProcessSession(CrashedCamera, timeout=1)
        origin = time.monotonic()
        task_clock = lambda: time.monotonic() - origin
        observer = RecoveryExposureObserver(session, 0, task_clock=task_clock)

        def wait_status(expected):
            deadline = time.monotonic() + 5
            while session.status not in expected and time.monotonic() < deadline:
                session.poll()
                time.sleep(0.005)
            self.assertIn(session.status, expected)

        try:
            wait_status({"idle"})
            session.request()
            wait_status({"failed"})
            self.assertTrue(observer.restart(Camera))
            self.assertEqual(session.last_exitcode, 7)
            wait_status({"idle"})
            now = time.monotonic()
            observer.bind_mapping(
                session.session, ExposureClockMap(now, now - origin), now - origin
            )
            deadline = time.monotonic() + 1
            while (
                observer.attempts[0]["outcome"] == "awaiting_clock_mapping"
                and time.monotonic() < deadline
            ):
                observer.step(task_clock())
                time.sleep(0.005)
            self.assertEqual(observer.attempts[0]["outcome"], "observation_restored")
            self.assertEqual(observer.status, "pending")
        finally:
            observer.close()
        self.assertIsNone(session.process)
