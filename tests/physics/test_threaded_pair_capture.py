import threading
import unittest
from unittest.mock import patch

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.threaded_pair_capture import CapturedPair, ThreadedPairCapture
from embodied_agent.two_stage_capture import StageCaptureRequest, TwoStageCaptureController
from embodied_agent.two_stage_runtime import TwoStageCaptureRuntime


def request(i):
    return StageCaptureRequest(i, "candidate", "camera", 0, 0)


def pair(req):
    p = StampedPosition((0.64, 0, 0.045), req.host_time)
    return CapturedPair(req.session_id, p, p)


def runtime(read, clock):
    c = TwoStageCaptureController(
        0, ExposureClockMap(0, 0), "camera", position_error_m=(0.001,) * 3
    )
    return TwoStageCaptureRuntime(c, read, clock=clock)


class ThreadedPairCaptureTests(unittest.TestCase):
    def test_capacity_no_hidden_queue_and_cancel_uncooperative(self):
        release = threading.Event()

        def read(req, cancel):
            release.wait(1)
            return pair(req)

        worker = ThreadedPairCapture(read)
        try:
            self.assertTrue(worker.submit(request(1)))
            self.assertTrue(worker.submit(request(2)))
            self.assertFalse(worker.submit(request(3)))
            worker.cancel()
            self.assertFalse(worker.reap()["workers_stopped"])
            self.assertFalse(worker.submit(request(3)))
        finally:
            release.set()
            result = worker.reap(0.25)
        self.assertTrue(result["workers_stopped"])
        self.assertFalse(result["hardware_cleanup_confirmed"])
        self.assertEqual(result["outstanding_slots"], 0)
        self.assertEqual(result["discarded_results"], 2)
        self.assertEqual(worker.drain(), [])

    def test_results_keep_completion_order(self):
        release = threading.Event()

        def read(req, cancel):
            if req.request_id == 1:
                release.wait(1)
            return pair(req)

        worker = ThreadedPairCapture(read)
        try:
            worker.submit(request(1))
            worker.submit(request(2))
            worker.threads[2].join(0.25)
            self.assertFalse(worker.threads[2].is_alive())
            release.set()
            self.assertTrue(worker.reap(0.25)["workers_stopped"])
            self.assertEqual([x[0] for x in worker.drain()], [2, 1])
            self.assertFalse(worker.submit(request(1)))
        finally:
            release.set()
            worker.cancel()
            worker.reap(0.25)

    def test_driver_error_is_bounded_and_consumed_once(self):
        def read(req, cancel):
            raise RuntimeError("x" * 1000)

        worker = ThreadedPairCapture(read)
        worker.submit(request(1))
        self.assertTrue(worker.reap(0.25)["workers_stopped"])
        results = worker.drain()
        self.assertEqual(len(results[0][2]), 512)
        self.assertEqual(worker.drain(), [])
        worker.cancel()

    def test_actual_workers_complete_both_stages_under_virtual_clock(self):
        now = [0.0]
        identities = set()

        def read(req, cancel):
            identities.add(threading.get_ident())
            return pair(req)

        rt = runtime(read, lambda: now[0])
        try:
            for ms in range(3401):
                now[0] = ms / 1000
                rt.step()
                self.assertTrue(rt.capture.reap(0.25)["workers_stopped"])
                if rt.controller.status != "pending":
                    break
            self.assertEqual(rt.controller.status, "confirmed_recent_evidence")
            self.assertEqual(len(rt.controller.requests), 36)
            self.assertNotIn(threading.get_ident(), identities)
            self.assertTrue(rt.capture.result()["closed"])
        finally:
            rt.close(wait_s=0.25)

    def test_stop_works_with_broken_clock_and_cooperative_reader(self):
        entered = threading.Event()
        broken = [False]

        def clock():
            if broken[0]:
                raise RuntimeError("clock failed")
            return 0

        def read(req, cancel):
            entered.set()
            cancel.wait(1)
            return pair(req)

        rt = runtime(read, clock)
        try:
            rt.step()
            self.assertTrue(entered.wait(0.25))
            broken[0] = True
            rt.step(stop=True)
            self.assertEqual(rt.controller.status, "cancelled")
            self.assertTrue(rt.capture.reap(0.25)["workers_stopped"])
            self.assertEqual(rt.capture.drain(), [])
        finally:
            rt.close(wait_s=0.25)

    def test_driver_exception_never_promotes_success(self):
        now = [0]

        def read(req, cancel):
            raise RuntimeError("camera disconnected")

        rt = runtime(read, lambda: now[0])
        try:
            rt.step()
            rt.capture.reap(0.25)
            now[0] = 0.001
            rt.step()
            self.assertEqual(rt.controller.status, "unconfirmed")
            self.assertEqual(len(rt.errors), 1)
            self.assertTrue(rt.capture.result()["closed"])
        finally:
            rt.close(wait_s=0.25)

    def test_clock_error_cancels_worker_before_propagation(self):
        now = [0]

        def read(req, cancel):
            cancel.wait(1)
            return pair(req)

        rt = runtime(read, lambda: now[0])
        try:
            rt.step()
            now[0] = -1
            with self.assertRaises(ValueError):
                rt.step()
            self.assertTrue(rt.capture.result()["closed"])
            self.assertEqual(rt.controller.reason, "capture_runtime_exception")
        finally:
            self.assertTrue(rt.close(wait_s=0.25)["workers_stopped"])

    def test_bad_return_type_fails_closed(self):
        worker = ThreadedPairCapture(lambda req, cancel: object())
        worker.submit(request(1))
        worker.reap(0.25)
        self.assertIn("CapturedPair required", worker.drain()[0][2])
        worker.cancel()

    def test_capacity_validation(self):
        for value in (0, 3, True, 1.5):
            with self.assertRaises(ValueError):
                ThreadedPairCapture(lambda req, cancel: pair(req), capacity=value)

    def test_hot_loop_does_not_build_audit_snapshots(self):
        rt = runtime(lambda req, cancel: pair(req), lambda: 0)
        try:
            with patch.object(
                rt.controller, "result", side_effect=AssertionError("snapshot in hot loop")
            ):
                self.assertEqual(rt.step(), "pending")
                self.assertEqual(rt.step(stop=True), "cancelled")
        finally:
            rt.close(wait_s=0.25)
