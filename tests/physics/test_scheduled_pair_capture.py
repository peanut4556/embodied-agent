import math
import time
import unittest

from embodied_agent.observation_capture import StampedPosition
from embodied_agent.process_pair_capture import ProcessPairCapture
from embodied_agent.scheduled_pair_capture import (
    ScheduledPairReader,
    ScheduledPairRequest,
    ScheduledReaderFactory,
)
from embodied_agent.threaded_pair_capture import CapturedPair


class Clock:
    def __init__(self):
        self.t = 10.0

    def __call__(self):
        return self.t

    def sleep(self, duration):
        self.t += duration


def pair(req, stamp):
    p = StampedPosition((0.64, 0, 0.045), stamp)
    return CapturedPair(req.session_id, p, p)


class RealReader:
    def __call__(self, req):
        return pair(req, time.monotonic())


def factory():
    return RealReader()


class ScheduledPairTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.req = ScheduledPairRequest(1, "session", 10, 10.02, 10.022)

    def adapter(self, reader):
        return ScheduledPairReader(reader, clock=self.clock, sleep=self.clock.sleep)

    def test_no_early_read_and_preserve_exposure(self):
        calls = []

        def read(req):
            calls.append(self.clock())
            return pair(req, self.clock())

        result = self.adapter(read)(self.req)
        self.assertEqual(calls, [10.02])
        self.assertEqual(result.primary.captured_at, 10.02)
        self.assertEqual(self.req.host_time, 10)

    def test_expired_reservation_does_not_call_reader(self):
        self.clock.t = 10.023
        with self.assertRaisesRegex(ValueError, "window_missed"):
            self.adapter(lambda _: self.fail("expired request read"))(self.req)

    def test_oversleep_does_not_catch_up(self):
        def oversleep(_):
            self.clock.t = 10.03

        adapter = ScheduledPairReader(
            lambda _: self.fail("oversleep must reject"), clock=self.clock, sleep=oversleep
        )
        with self.assertRaisesRegex(ValueError, "window_missed"):
            adapter(self.req)

    def test_slow_read_before_exposure_rejected(self):
        def read(req):
            self.clock.sleep(0.005)
            return pair(req, self.clock())

        with self.assertRaisesRegex(ValueError, "exposure_outside_window"):
            self.adapter(read)(self.req)

    def test_post_exposure_processing_does_not_rewrite_timestamp(self):
        def read(req):
            result = pair(req, self.clock())
            self.clock.sleep(0.005)
            return result

        result = self.adapter(read)(self.req)
        self.assertEqual(result.primary.captured_at, 10.02)
        self.assertGreater(self.clock(), self.req.latest_host_time)

    def test_cached_future_bool_and_nan_exposures_rejected(self):
        for stamp in (10, 10.021, True, math.nan):
            self.clock.t = 10
            with self.assertRaisesRegex(ValueError, "exposure_outside_window"):
                self.adapter(lambda req, stamp=stamp: pair(req, stamp))(self.req)

    def test_wrong_session_or_pair_rejected(self):
        for value in (None, CapturedPair("old", None, None)):
            self.clock.t = 10
            with self.assertRaisesRegex(ValueError, "session_or_pair_invalid"):
                self.adapter(lambda _, value=value: value)(self.req)

    def test_backward_clock_rejected(self):
        self.clock.t = 9.9
        with self.assertRaisesRegex(ValueError, "clock_invalid"):
            self.adapter(lambda _: self.fail("clock failure must reject"))(self.req)

    def test_reservation_bounds(self):
        for kwargs in (
            {"request_id": True},
            {"request_id": 37},
            {"session_id": ""},
            {"host_time": math.nan},
            {"target_host_time": 10.251, "latest_host_time": 10.252},
            {"target_host_time": 9.9},
            {"latest_host_time": 10.04},
            {"latest_host_time": 10.019},
        ):
            args = {
                "request_id": 1,
                "session_id": "s",
                "host_time": 10,
                "target_host_time": 10.02,
                "latest_host_time": 10.022,
            } | kwargs
            with self.assertRaises(ValueError):
                ScheduledPairRequest(**args)

    def test_real_process_factory_and_packet_contract(self):
        backend = ProcessPairCapture(ScheduledReaderFactory(factory), trace_timing=True)
        try:
            self.assertTrue(backend.wait_ready())
            now = time.monotonic()
            req = ScheduledPairRequest(1, "real", now, now + 0.03, now + 0.04)
            self.assertTrue(backend.submit(req))
            packets = []
            deadline = now + 1
            while not packets and time.monotonic() < deadline:
                packets = backend.drain()
                time.sleep(0.001)
            self.assertEqual(len(packets), 1)
            # Host scheduler can miss even a 10 ms window: rejection is valid,
            # accepting an out-of-window exposure is not.
            _, result, error = packets[0]
            if result is not None:
                self.assertIsNone(error)
                self.assertGreaterEqual(result.primary.captured_at, req.target_host_time)
                self.assertLessEqual(result.primary.captured_at, req.latest_host_time)
            else:
                self.assertIsNotNone(error)
        finally:
            backend.cancel()
            self.assertTrue(backend.reap(0.25)["workers_stopped"])

    def test_process_cancel_during_reservation_discards_results(self):
        backend = ProcessPairCapture(ScheduledReaderFactory(factory), trace_timing=True)
        try:
            self.assertTrue(backend.wait_ready())
            now = time.monotonic()
            req = ScheduledPairRequest(1, "real", now, now + 0.2, now + 0.202)
            self.assertTrue(backend.submit(req))
            backend.cancel()
            self.assertTrue(backend.reap(0.25)["workers_stopped"])
            self.assertEqual(backend.drain(), [])
            self.assertFalse(backend.submit(req))
        finally:
            backend.cancel()
            backend.reap(0.25)


if __name__ == "__main__":
    unittest.main()
