import unittest

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.reserved_two_stage import ReservedCandidate, ReservedTwoStageController


def controller():
    return ReservedTwoStageController(
        0.02, ExposureClockMap(100, 0), "session", position_error_m=(0.001,) * 3
    )


def point(stamp, x=0.64):
    return StampedPosition((x, 0, 0.045), stamp)


def drive(c, *, candidate_delay=0.001, mode="static"):
    jobs = []
    handoff = None
    for tick in range(3501):
        now = tick / 1000
        for req, delivery in list(jobs):
            if delivery > now + 1e-9:
                continue
            jobs.remove((req, delivery))
            p = point(req.target_host_time + 0.0005)
            session = "session"
            if req.stage == "witness":
                if mode == "motion":
                    p = point(req.target_host_time + 0.0005, x=0.642)
                elif mode == "old_exposure":
                    p = point(100 + c.gate.start - 0.001)
                elif mode == "old_session":
                    session = "old"
            c.complete(req.request_id, now, 100 + now, p, p, session_id=session)
            if c.stage == "witness" and handoff is None:
                handoff = c.result()
        if c.status != "pending":
            break
        req = c.poll(now, 100 + now)
        if req is not None:
            if req.stage == "witness" and mode == "stop_witness":
                c.poll(now, 100 + now, stop=True)
                break
            if req.stage == "witness" and mode == "stall_witness":
                continue
            delay = candidate_delay if req.stage == "candidate" else 0.001
            jobs.append((req, req.target_host_time - 100 + delay))
    return handoff


class ReservedTwoStageTests(unittest.TestCase):
    def test_witness_stop_or_timeout_never_promotes_candidate(self):
        for mode, status, reason in (
            ("stop_witness", "cancelled", "user_stop"),
            ("stall_witness", "unconfirmed", "witness_request_timeout"),
        ):
            c = controller()
            drive(c, mode=mode)
            self.assertEqual(c.status, status)
            self.assertEqual(c.reason, reason)
            self.assertEqual(c.candidate.status, "confirmed_in_tray")
            self.assertEqual(c.result()["cancel_request_ids"], [32])
            self.assertEqual(c.gate.rows, [])

    def test_complete_flow_withholds_candidate_and_uses_new_witness(self):
        c = controller()
        handoff = drive(c)
        self.assertEqual(handoff["status"], "pending")
        self.assertEqual(handoff["reason"], "candidate_withheld")
        self.assertEqual(len(handoff["requests"]), 31)
        self.assertEqual(handoff["witness"]["observations"], [])
        self.assertEqual(c.status, "confirmed_recent_evidence")
        self.assertEqual(len(c.requests), 36)
        self.assertEqual([r["request_id"] for r in c.requests], list(range(1, 37)))
        for row in c.requests[31:]:
            self.assertGreaterEqual(row["task_time"], c.gate.start)
            self.assertGreater(row["target_host_time"], 100 + c.gate.start)
        self.assertFalse(c.result()["witness"]["guarantees_current_or_future_stability"])

    def test_two_inflight_slow_candidate_returns_still_require_fresh_witness(self):
        c = controller()
        drive(c, candidate_delay=0.08)
        self.assertEqual(c.status, "confirmed_recent_evidence")
        self.assertEqual(c.candidate.peak_inflight, 2)

    def test_motion_and_old_evidence_never_promote_candidate(self):
        for mode, reason in (
            ("motion", "witness_motion"),
            ("old_exposure", "reservation_exposure_rejected"),
            ("old_session", "camera_session_changed"),
        ):
            c = controller()
            drive(c, mode=mode)
            self.assertEqual(c.status, "unconfirmed")
            self.assertEqual(c.reason, reason)
            self.assertEqual(c.candidate.status, "confirmed_in_tray")

    def test_stop_cancels_future_reservation_and_quarantines_return(self):
        c = controller()
        req = c.poll(0, 100)
        self.assertEqual(req.host_time, 100)
        self.assertAlmostEqual(req.target_host_time, 100.02)
        c.poll(0.001, 100.001, stop=True)
        p = point(req.target_host_time)
        self.assertFalse(c.complete(1, 0.021, 100.021, p, p, session_id="session"))
        self.assertEqual(c.status, "cancelled")
        self.assertEqual(c.result()["cancel_request_ids"], [1])
        self.assertEqual(c.ignored_results, 1)
        self.assertIsNone(c.poll(0.03, 100.03))

    def test_controller_rechecks_both_exposure_bounds(self):
        for stamp in (100.019, 100.023):
            c = controller()
            req = c.poll(0, 100)
            p, q = point(req.target_host_time), point(stamp)
            c.complete(1, 0.024, 100.024, p, q, session_id="session")
            self.assertEqual(c.reason, "reservation_exposure_rejected")
            self.assertEqual(c.candidate.check.rows, [])

    def test_candidate_boundary_itself_rejects_pre_target_exposure(self):
        c = ReservedCandidate(0.02, ExposureClockMap(100, 0), "s", position_error_m=(0.001,) * 3)
        c.poll(0, 100)
        p = point(100.01)
        c.complete(1, 0.011, 100.011, p, p, session_id="s")
        self.assertEqual(c.reason, "capture_not_at_scheduled_request")

    def test_late_submission_does_not_shift_schedule(self):
        c = controller()
        self.assertIsNone(c.poll(0.021, 100.021))
        self.assertEqual(c.reason, "reservation_submission_missed")
        self.assertEqual(c.requests, [])

    def test_timeout_includes_reservation_wait(self):
        c = controller()
        c.poll(0, 100)
        c.poll(0.25, 100.25)
        self.assertEqual(c.reason, "capture_request_timeout")
        self.assertEqual(c.result()["cancel_request_ids"], [1])

    def test_out_of_order_completion_fails(self):
        c = controller()
        c.poll(0, 100)
        req = c.poll(0.205, 100.205)
        p = point(req.target_host_time)
        c.complete(2, 0.226, 100.226, p, p, session_id="session")
        self.assertEqual(c.reason, "out_of_order_completion")

    def test_mapping_and_snapshot(self):
        for start, mapping in (
            (0, ExposureClockMap(100, 0)),
            (0.02, ExposureClockMap(100, 0, rate=2)),
        ):
            with self.assertRaises(ValueError):
                ReservedTwoStageController(start, mapping, "s", position_error_m=(0.001,) * 3)
        c = controller()
        c.poll(0, 100)
        result = c.result()
        result["requests"][0]["host_time"] = -1
        self.assertEqual(c.requests[0]["host_time"], 100)
        with self.assertRaises(ValueError):
            c.poll(-0.001, 99.999)


if __name__ == "__main__":
    unittest.main()
