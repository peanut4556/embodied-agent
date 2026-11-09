import unittest

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.serial_challenge import SerialChallengeController

P = (0.64, 0, 0.045)


def create(reserve=0.1):
    return SerialChallengeController(
        0,
        ExposureClockMap(0, 0),
        "camera",
        position_error_m=(0.001,) * 3,
        delivery_budget_s=reserve,
    )


def run(delay, reserve=0.1):
    c = create(reserve)
    now = 0
    while c.status == "pending":
        now = max(now, c.check.next_capture_time)
        req = c.poll(now, now)
        if req is None:
            break
        p = StampedPosition(P, now)
        now += delay
        c.complete(req.request_id, now, now, p, p, session_id="camera")
    return c


class SerialChallengeTests(unittest.TestCase):
    def test_twenty_ms_serial_path_is_causal(self):
        c = run(0.02)
        self.assertEqual(c.status, "confirmed_in_tray")
        r = c.result()
        self.assertEqual(r["request_count"], 31)
        self.assertAlmostEqual(r["events"][-1]["delivery_time"], 3.02)
        for req, previous in zip(r["requests"][1:], r["events"][:-1], strict=True):
            self.assertGreaterEqual(req["task_time"] + 1e-9, previous["delivery_time"])

    def test_zero_latency_default_contract(self):
        self.assertEqual(run(0, 0).status, "confirmed_in_tray")

    def test_eighty_ms_cannot_fake_twenty_hz(self):
        c = run(0.08)
        self.assertEqual(c.status, "unconfirmed")
        self.assertEqual(c.reason, "challenge_capture_gap")
        self.assertAlmostEqual(c.result()["requests"][-1]["task_time"], 2.08)

    def test_single_flight_timeout_and_late_result(self):
        c = create()
        req = c.poll(0, 0)
        self.assertIsNone(c.poll(0.1, 0.1))
        self.assertEqual(c.result()["request_count"], 1)
        c.poll(0.25, 0.25)
        self.assertEqual(c.reason, "capture_request_timeout")
        p = StampedPosition(P, 0)
        self.assertFalse(c.complete(req.request_id, 0.3, 0.3, p, p, session_id="camera"))
        self.assertTrue(c.result()["driver_cancel_required"])
        self.assertIsNone(c.poll(0.4, 0.4))

    def test_cached_frame_before_request_rejected(self):
        c = create()
        req = c.poll(0.01, 0.01)
        p = StampedPosition(P, 0)
        c.complete(req.request_id, 0.02, 0.02, p, p, session_id="camera")
        self.assertEqual(c.reason, "capture_outside_request_interval")

    def test_unknown_result_does_not_consume_request(self):
        c = create()
        req = c.poll(0, 0)
        p = StampedPosition(P, 0)
        self.assertFalse(c.complete(999, 0.01, 0.01, p, p, session_id="camera"))
        self.assertTrue(c.complete(req.request_id, 0.02, 0.02, p, p, session_id="camera"))
        self.assertFalse(c.complete(req.request_id, 0.03, 0.03, p, p, session_id="camera"))
        self.assertEqual(c.result()["ignored_results"], 2)

    def test_waiting_clock_drift_and_stop(self):
        c = create()
        c.poll(0, 0)
        c.poll(0.1, 0.16)
        self.assertEqual(c.reason, "controller_clock_mismatch")
        c = create()
        c.poll(0, 0)
        c.poll(0.1, 0.1, stop=True)
        self.assertEqual(c.status, "cancelled")
        self.assertTrue(c.result()["driver_cancel_required"])

    def test_capture_window_not_extended_by_reserve(self):
        c = create()
        self.assertIsNone(c.poll(3.02, 3.02))
        self.assertEqual(c.reason, "capture_deadline_missed")
        self.assertEqual(c.result()["request_count"], 0)

    def test_session_and_missing_pair_rejected(self):
        for session, p in (("other", StampedPosition(P, 0)), ("camera", None)):
            c = create()
            req = c.poll(0, 0)
            c.complete(req.request_id, 0.01, 0.01, p, p, session_id=session)
            self.assertEqual(c.status, "unconfirmed")

    def test_snapshot_and_backward_clock(self):
        c = create()
        c.poll(0.01, 0.01)
        r = c.result()
        r["requests"].clear()
        before = c.result()
        with self.assertRaises(ValueError):
            c.poll(0, 0)
        self.assertEqual(c.result(), before)
        self.assertEqual(c.result()["request_count"], 1)
