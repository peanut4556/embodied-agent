import unittest

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.two_stage_capture import TwoStageCaptureController
from scripts.evaluate_two_stage_capture import run

P = (0.64, 0, 0.045)


def candidate_ready():
    c = TwoStageCaptureController(
        0, ExposureClockMap(0, 0), "camera", position_error_m=(0.001,) * 3
    )
    jobs = {}
    for ms in range(3261):
        now = ms / 1000
        if ms in jobs:
            req = jobs.pop(ms)
            p = StampedPosition(P, req.host_time)
            c.complete(req.request_id, now, now, p, p, session_id="camera")
        if c.stage == "witness":
            return c
        req = c.poll(now, now)
        if req:
            jobs[ms + 80] = req
    raise AssertionError("candidate not reached")


class TwoStageCaptureTests(unittest.TestCase):
    def test_candidate_is_withheld_and_ids_do_not_restart(self):
        c = candidate_ready()
        self.assertEqual(c.status, "pending")
        self.assertEqual(c.reason, "candidate_withheld")
        self.assertEqual(c.candidate.status, "confirmed_in_tray")
        req = c.poll(c.gate.start, c.gate.start)
        self.assertEqual(req.request_id, 32)
        p = StampedPosition(P, req.host_time)
        self.assertFalse(
            c.complete(31, c.gate.start + 0.001, c.gate.start + 0.001, p, p, session_id="camera")
        )
        self.assertIn(32, c.pending)

    def test_return_time_matrix(self):
        for service in (1, 10, 20, 21, 80, "stall", "cached"):
            r = run("static", service, 202611123)["result"]
            self.assertEqual(r["status"] == "confirmed_recent_evidence", service in (1, 10, 20))
            self.assertLessEqual(len(r["requests"]), 36)

    def test_frame_after_candidate_but_before_request_rejected(self):
        c = candidate_ready()
        now = c.gate.start + 0.005
        req = c.poll(now, now)
        p = StampedPosition(P, c.gate.start)
        c.complete(req.request_id, now + 0.01, now + 0.01, p, p, session_id="camera")
        self.assertEqual(c.reason, "capture_outside_request_interval")

    def test_stalled_request_never_retried(self):
        c = candidate_ready()
        now = c.gate.start
        req = c.poll(now, now)
        self.assertIsNone(c.poll(now + 0.01, now + 0.01))
        c.poll(now + 0.03, now + 0.03)
        self.assertEqual(c.reason, "witness_request_timeout")
        self.assertEqual(c.result()["cancel_request_ids"], [32])
        p = StampedPosition(P, now)
        self.assertFalse(
            c.complete(req.request_id, now + 0.04, now + 0.04, p, p, session_id="camera")
        )
        self.assertEqual(len(c.requests), 32)

    def test_stop_and_no_request_after_terminal(self):
        for stage in ("candidate", "witness"):
            c = (
                candidate_ready()
                if stage == "witness"
                else TwoStageCaptureController(
                    0, ExposureClockMap(0, 0), "camera", position_error_m=(0.001,) * 3
                )
            )
            now = c.gate.start if c.gate else 0
            req = c.poll(now, now)
            c.poll(now, now, stop=True)
            self.assertEqual(c.status, "cancelled")
            self.assertEqual(c.result()["cancel_request_ids"], [req.request_id])
            self.assertIsNone(c.poll(now + 0.1, now + 0.1))

    def test_clock_and_snapshot(self):
        c = candidate_ready()
        before = c.result()
        with self.assertRaises(ValueError):
            c.poll(0, 0)
        self.assertEqual(c.result(), before)
        c.poll(c.gate.start, c.gate.start + 0.1)
        self.assertEqual(c.reason, "controller_clock_mismatch")
        r = c.result()
        r["requests"].clear()
        self.assertEqual(len(c.requests), 31)

    def test_session_and_missing_pair(self):
        for session, valid in (("wrong", True), ("camera", False)):
            c = candidate_ready()
            now = c.gate.start
            req = c.poll(now, now)
            p = StampedPosition(P, now) if valid else None
            c.complete(req.request_id, now + 0.01, now + 0.01, p, p, session_id=session)
            self.assertEqual(c.status, "unconfirmed")

    def test_new_movement_after_candidate_rejected(self):
        for scene in ("step", "sine"):
            r = run(scene, 10, 202611123)["result"]
            self.assertEqual(r["reason"], "witness_motion")
            self.assertEqual(r["candidate"]["status"], "confirmed_in_tray")
