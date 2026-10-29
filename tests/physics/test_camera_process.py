import os
import signal
import tempfile
import time
import unittest
from functools import partial
from pathlib import Path

from embodied_agent.camera_process import CameraProcessSession
from embodied_agent.observation_capture import StampedPosition


class FixedCamera:
    def read(self):
        return StampedPosition((0.64, 0, 0.04), time.monotonic())


class BlockedCamera:
    def __init__(self, marker=None, ignore_terminate=False):
        self.marker, self.ignore_terminate = marker, ignore_terminate

    def read(self):
        if self.ignore_terminate:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if self.marker:
            Path(self.marker).write_text("entered")
        time.sleep(60)
        return StampedPosition((0.8, 0, 0.04), time.monotonic())


class CrashedCamera:
    def read(self):
        os._exit(7)


class StaleCamera:
    def read(self):
        return StampedPosition((0.64, 0, 0.04), 0)


def failed_factory():
    raise OSError("device unavailable")


class CameraProcessTests(unittest.TestCase):
    def test_kill_fallback_reaps_driver_ignoring_termination(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "entered"
            session = CameraProcessSession(
                partial(BlockedCamera, str(marker), True), timeout=1, shutdown_timeout=0.05
            )
            try:
                self.wait_status(session, {"idle"})
                session.request()
                deadline = time.monotonic() + 2
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertTrue(marker.exists())
            finally:
                session.close()
            self.assertEqual(session.last_exitcode, -signal.SIGKILL)

    def wait_status(self, session, target):
        deadline = time.monotonic() + 6
        value = None
        while time.monotonic() < deadline:
            sample = session.poll()
            if sample is not None:
                value = sample
            if session.status in target:
                return value
            time.sleep(0.005)
        self.fail(f"camera session did not reach {target}: {session.status}")

    def test_fresh_result_consumed_once_and_single_request(self):
        session = CameraProcessSession(FixedCamera, timeout=1)
        try:
            self.wait_status(session, {"idle"})
            self.assertTrue(session.request())
            self.assertFalse(session.request())
            result = self.wait_status(session, {"ready"})
            self.assertEqual(result.position, (0.64, 0, 0.04))
            self.assertIsNone(session.poll())
        finally:
            session.close()
        self.assertIsNotNone(session.last_exitcode)

    def test_blocked_process_reaped_before_new_session(self):
        session = CameraProcessSession(BlockedCamera, timeout=0.15)
        try:
            self.wait_status(session, {"idle"})
            old_session, old_pid = session.session, session.process.pid
            session.request()
            self.wait_status(session, {"timed_out"})
            self.assertFalse(session.request())
            session.restart(FixedCamera)
            self.assertIsNotNone(session.last_exitcode)
            self.assertNotEqual(session.session, old_session)
            self.assertNotEqual(session.process.pid, old_pid)
            self.wait_status(session, {"idle"})
            session.request()
            result = self.wait_status(session, {"ready"})
            self.assertEqual(result.position, (0.64, 0, 0.04))
        finally:
            session.close()

    def test_unconsumed_old_result_cannot_cross_restart(self):
        session = CameraProcessSession(FixedCamera, timeout=1)
        try:
            self.wait_status(session, {"idle"})
            session.request()
            # Leave the response in the old pipe; restart must close that pipe.
            deadline = time.monotonic() + 2
            while not session.connection.poll() and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertTrue(session.connection.poll())
            old = session.session
            session.restart(StaleCamera)
            self.assertNotEqual(old, session.session)
            self.wait_status(session, {"idle"})
            self.assertIsNone(session.poll())
            session.request()
            self.assertIsNone(self.wait_status(session, {"error"}))
        finally:
            session.close()

    def test_crash_detected_and_reaped(self):
        session = CameraProcessSession(CrashedCamera, timeout=1)
        try:
            self.wait_status(session, {"idle"})
            session.request()
            self.wait_status(session, {"failed"})
        finally:
            session.close()
        self.assertEqual(session.last_exitcode, 7)

    def test_initialization_failure_and_stale_position_are_not_success(self):
        for factory, expected in ((failed_factory, "failed"), (StaleCamera, "error")):
            session = CameraProcessSession(factory, timeout=1)
            try:
                if factory is StaleCamera:
                    self.wait_status(session, {"idle"})
                    session.request()
                self.assertIsNone(self.wait_status(session, {expected}))
            finally:
                session.close()

    def test_invalid_timeout_rejected(self):
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                CameraProcessSession(FixedCamera, timeout=timeout)
