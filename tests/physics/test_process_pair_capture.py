import os
import signal
import time
import unittest

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.process_pair_capture import ProcessPairCapture
from embodied_agent.threaded_pair_capture import CapturedPair
from embodied_agent.two_stage_capture import StageCaptureRequest, TwoStageCaptureController
from embodied_agent.two_stage_runtime import TwoStageCaptureRuntime


class Reader:
    def __call__(self, req):
        p = StampedPosition((0.64, 0, 0.045), time.monotonic())
        return CapturedPair(req.session_id, p, p)


def factory():
    return Reader()


class StuckReader:
    def __call__(self, req):
        while True:
            time.sleep(0.01)


def stuck_factory():
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    return StuckReader()


def bad_factory():
    raise RuntimeError("initialization failed")


class ExitReader:
    def __call__(self, req):
        os._exit(7)


def exit_factory():
    return ExitReader()


def startup_hang_factory():
    while True:
        time.sleep(0.01)


class OutOfOrderReader:
    def __call__(self, req):
        if req.request_id == 1:
            time.sleep(0.03)
        return Reader()(req)


def out_of_order_factory():
    return OutOfOrderReader()


def req(i):
    now = time.monotonic()
    return StageCaptureRequest(i, "candidate", "camera", 0, now)


class ProcessPairTests(unittest.TestCase):
    def test_prewarm_pair_and_duplicate(self):
        b = ProcessPairCapture(factory)
        try:
            self.assertTrue(b.wait_ready())
            r = req(1)
            self.assertTrue(b.submit(r))
            self.assertFalse(b.submit(r))
            deadline = time.monotonic() + 1
            packets = []
            while not packets and time.monotonic() < deadline:
                packets = b.drain()
                time.sleep(0.001)
            self.assertEqual(len(packets), 1)
            self.assertEqual(packets[0][0], 1)
            self.assertGreaterEqual(packets[0][1].primary.captured_at, r.host_time)
            self.assertEqual(b.drain(), [])
        finally:
            b.cancel()
            self.assertTrue(b.reap(0.25)["workers_stopped"])

    def test_native_style_hang_requires_kill_and_is_reaped(self):
        b = ProcessPairCapture(stuck_factory)
        try:
            self.assertTrue(b.wait_ready())
            self.assertTrue(b.submit(req(1)))
            self.assertTrue(b.submit(req(2)))
            self.assertFalse(b.submit(req(3)))
            b.cancel()
            r = b.reap(0.25)
            self.assertTrue(r["workers_stopped"])
            self.assertTrue(all(w["killed"] for w in r["workers"]))
            self.assertTrue(all(w["exitcode"] is not None for w in r["workers"]))
            self.assertFalse(r["hardware_cleanup_confirmed"])
            self.assertFalse(b.submit(req(3)))
            self.assertEqual(b.drain(), [])
        finally:
            b.cancel()
            b.reap(0.25)

    def test_factory_failure_can_be_closed(self):
        b = ProcessPairCapture(bad_factory)
        try:
            self.assertFalse(b.wait_ready())
        finally:
            b.cancel()
            self.assertTrue(b.reap(0.25)["workers_stopped"])

    def test_runtime_stop_owns_backend_teardown(self):
        b = ProcessPairCapture(stuck_factory)
        try:
            self.assertTrue(b.wait_ready())
            origin = time.monotonic()
            c = TwoStageCaptureController(
                0, ExposureClockMap(origin, 0), "camera", position_error_m=(0.001,) * 3
            )
            r = TwoStageCaptureRuntime(c, capture_backend=b)
            r.step()
            self.assertEqual(r.step(stop=True), "cancelled")
            self.assertTrue(r.close(wait_s=0.25)["workers_stopped"])
        finally:
            b.cancel()
            b.reap(0.25)

    def test_capacity_validation(self):
        for n in (0, 3, True):
            with self.assertRaises(ValueError):
                ProcessPairCapture(factory, capacity=n)

    def test_child_crash_is_reported_once(self):
        b = ProcessPairCapture(exit_factory, capacity=1)
        try:
            self.assertTrue(b.wait_ready())
            b.submit(req(1))
            packets = []
            deadline = time.monotonic() + 1
            while not packets and time.monotonic() < deadline:
                packets = b.drain()
                time.sleep(0.001)
            self.assertEqual(len(packets), 1)
            self.assertIsNone(packets[0][1])
            self.assertIsNotNone(packets[0][2])
            self.assertEqual(b.drain(), [])
        finally:
            b.cancel()
            self.assertTrue(b.reap(0.25)["workers_stopped"])

    def test_startup_timeout_reclaims_children(self):
        b = ProcessPairCapture(startup_hang_factory)
        try:
            self.assertFalse(b.wait_ready(0.05))
            self.assertFalse(b.submit(req(1)))
        finally:
            b.cancel()
            self.assertTrue(b.reap(0.25)["workers_stopped"])

    def test_results_do_not_hide_out_of_order_completion(self):
        b = ProcessPairCapture(out_of_order_factory)
        try:
            self.assertTrue(b.wait_ready())
            b.submit(req(1))
            b.submit(req(2))
            deadline = time.monotonic() + 1
            while not all(s["connection"].poll() for s in b.slots) and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertEqual([p[0] for p in b.drain()], [2, 1])
        finally:
            b.cancel()
            self.assertTrue(b.reap(0.25)["workers_stopped"])
