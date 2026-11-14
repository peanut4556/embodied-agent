import math
import unittest
from unittest.mock import patch

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.recent_evidence_handoff import RecentEvidenceHandoff
from embodied_agent.reserved_two_stage import ReservedTwoStageController


def confirmed_controller():
    c = ReservedTwoStageController(
        0.02, ExposureClockMap(100, 0), "session", position_error_m=(0.001,) * 3
    )
    now = 0
    while c.status == "pending" and now < 4:
        req = c.poll(now, 100 + now)
        if req is None:
            now += 0.001
            continue
        p = StampedPosition((0.64, 0, 0.045), req.target_host_time + 0.0005)
        now = p.captured_at - 100 + 0.001
        c.complete(req.request_id, now, 100 + now, p, p, session_id="session")
    assert c.status == "confirmed_recent_evidence"
    return c


class HandoffTests(unittest.TestCase):
    def test_consume_once_without_mutating_evidence(self):
        c = confirmed_controller()
        h = RecentEvidenceHandoff(c, clock=lambda: c.last_host)
        receipt = h.consume("session")
        self.assertEqual(receipt["status"], "consumed")
        self.assertFalse(receipt["robot_action_authorized"])
        self.assertLessEqual(receipt["remaining_s"], 0.02)
        receipt["evidence"]["requests"].clear()
        receipt["remaining_s"] = 999
        self.assertEqual(len(c.requests), 36)
        self.assertLess(h.result()["receipt"]["remaining_s"], 0.02)
        self.assertEqual(h.consume("session")["reason"], "already_consumed")
        with self.assertRaises(RuntimeError):
            RecentEvidenceHandoff(c)

    def test_wrong_session_does_not_consume_current_evidence(self):
        c = confirmed_controller()
        h = RecentEvidenceHandoff(c, clock=lambda: c.last_host)
        self.assertEqual(h.consume("old")["reason"], "session_mismatch")
        self.assertEqual(h.consume("session")["status"], "consumed")

    def test_expiry_uses_exposure_not_decision_and_cannot_renew(self):
        c = confirmed_controller()
        exposure = 100 + c.gate.last_exposure
        host = [exposure + 0.02]
        h = RecentEvidenceHandoff(c, clock=lambda: host[0])
        self.assertEqual(h.consume("session")["reason"], "expired")
        self.assertEqual(c.status, "confirmed_recent_evidence")
        host[0] = c.last_host
        self.assertEqual(h.consume("session")["reason"], "expired")

    def test_snapshot_construction_time_counts_against_age(self):
        c = confirmed_controller()
        host = [c.last_host]
        h = RecentEvidenceHandoff(c, clock=lambda: host[0])
        original = c.result

        def slow_snapshot():
            result = original()
            host[0] += 0.03
            return result

        with patch.object(c, "result", side_effect=slow_snapshot):
            self.assertEqual(h.consume("session")["reason"], "expired")

    def test_invalid_clock_latches(self):
        for bad in (True, math.nan, math.inf, 0):
            c = confirmed_controller()
            host = [bad]
            h = RecentEvidenceHandoff(c, clock=lambda host=host: host[0])
            self.assertEqual(h.consume("session")["reason"], "invalid_clock")
            host[0] = c.last_host
            self.assertEqual(h.consume("session")["reason"], "invalid_clock")

    def test_clock_exception_fails_closed(self):
        c = confirmed_controller()

        def broken():
            raise RuntimeError("clock unavailable")

        h = RecentEvidenceHandoff(c, clock=broken)
        self.assertEqual(h.consume("session")["reason"], "invalid_clock")

    def test_revocation_does_not_erase_historical_confirmation(self):
        c = confirmed_controller()
        h = RecentEvidenceHandoff(c, clock=lambda: c.last_host)
        h.revoke()
        self.assertEqual(h.consume("session")["reason"], "revoked")
        self.assertEqual(c.status, "confirmed_recent_evidence")

    def test_pending_confirmation_cannot_be_delivered(self):
        c = ReservedTwoStageController(
            0.02, ExposureClockMap(100, 0), "session", position_error_m=(0.001,) * 3
        )
        h = RecentEvidenceHandoff(c, clock=lambda: 100)
        self.assertEqual(h.consume("session")["reason"], "confirmation_not_complete")
        self.assertEqual(h.state, "available")


if __name__ == "__main__":
    unittest.main()
