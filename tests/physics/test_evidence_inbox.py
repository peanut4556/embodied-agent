import unittest

from test_recent_evidence_handoff import confirmed_controller

from embodied_agent.evidence_inbox import EvidenceInbox
from embodied_agent.recent_evidence_handoff import RecentEvidenceHandoff


class InboxTests(unittest.TestCase):
    def setUp(self):
        self.c = confirmed_controller()
        self.now = self.c.last_host
        self.h = RecentEvidenceHandoff(self.c, clock=lambda: self.now)
        self.q = EvidenceInbox(clock=lambda: self.now)

    def test_queue_dispatch_once_and_no_payload_in_snapshots(self):
        offered = self.q.offer(self.h, "session")
        self.assertEqual(offered["status"], "queued")
        self.assertNotIn("evidence", offered)
        self.assertNotIn("evidence", self.q.result()["pending"])
        seen = []
        result = self.q.dispatch(seen.append)
        self.assertEqual(result["status"], "delivered")
        self.assertFalse(seen[0]["robot_action_authorized"])
        self.assertEqual(self.q.dispatch(seen.append)["reason"], "no_pending_evidence")
        self.assertEqual(len(seen), 1)

    def test_queue_delay_expires_successful_receipt_without_callback(self):
        self.q.offer(self.h, "session")
        snapshot = self.q.result()
        self.now = snapshot["pending"]["expires_host_time"]
        snapshot["pending"]["expires_host_time"] += 999
        result = self.q.dispatch(lambda _: self.fail("expired callback"))
        self.assertEqual(result["reason"], "expired_in_queue")
        self.assertEqual(self.h.state, "consumed")
        self.assertEqual(self.c.status, "confirmed_recent_evidence")

    def test_retirement_revokes_already_consumed_receipt(self):
        self.q.offer(self.h, "session")
        self.h.revoke()
        self.assertTrue(self.h.result()["revoked"])
        self.assertEqual(self.h.state, "consumed")
        result = self.q.dispatch(lambda _: self.fail("revoked callback"))
        self.assertEqual(result["reason"], "session_revoked")

    def test_queue_backpressure_does_not_consume_next_source(self):
        other = RecentEvidenceHandoff(confirmed_controller(), clock=lambda: self.now)
        self.q.offer(self.h, "session")
        self.assertEqual(self.q.offer(other, "session")["reason"], "inbox_busy")
        self.assertEqual(other.state, "available")
        self.q.dispatch(lambda _: None)
        self.assertEqual(self.q.offer(other, "session")["status"], "queued")

    def test_consumer_exception_has_unknown_outcome_and_no_retry(self):
        calls = []

        def consumer(_):
            calls.append("effect")
            raise RuntimeError("after effect")

        self.q.offer(self.h, "session")
        self.assertEqual(self.q.dispatch(consumer)["status"], "outcome_unknown")
        self.assertEqual(self.q.dispatch(consumer)["reason"], "no_pending_evidence")
        self.assertEqual(calls, ["effect"])

    def test_reentrant_offer_and_dispatch_are_blocked(self):
        other = RecentEvidenceHandoff(confirmed_controller(), clock=lambda: self.now)

        def consumer(_):
            self.assertEqual(self.q.offer(other, "session")["reason"], "inbox_busy")
            self.assertEqual(self.q.dispatch(consumer)["reason"], "inbox_busy")

        self.q.offer(self.h, "session")
        self.assertEqual(self.q.dispatch(consumer)["status"], "delivered")
        self.assertEqual(other.state, "available")

    def test_clock_regression_latches_queue_and_burns_slot(self):
        self.q.offer(self.h, "session")
        self.now -= 1
        self.assertEqual(self.q.dispatch(lambda _: self.fail())["reason"], "invalid_clock")
        self.now += 1
        other = RecentEvidenceHandoff(confirmed_controller(), clock=lambda: self.now)
        self.assertEqual(self.q.offer(other, "session")["reason"], "invalid_clock")
        self.assertEqual(other.state, "available")

    def test_close_and_wrong_token(self):
        self.assertEqual(self.q.offer(self.h, "old")["reason"], "session_mismatch")
        self.assertEqual(self.h.state, "available")
        self.q.offer(self.h, "session")
        self.q.close()
        self.assertEqual(self.q.dispatch(lambda _: self.fail())["reason"], "no_pending_evidence")
        self.assertEqual(self.q.result()["history"][-1]["reason"], "inbox_closed")

    def test_receipt_dict_cannot_be_imported_or_queued(self):
        receipt = self.h.consume("session")
        receipt["expires_host_time"] += 999
        with self.assertRaises(TypeError):
            self.q.offer(receipt, "session")
        self.assertEqual(self.q.offer(self.h, "session")["reason"], "already_consumed")

    def test_history_is_bounded_and_snapshot_isolated(self):
        for _ in range(18):
            h = RecentEvidenceHandoff(confirmed_controller(), clock=lambda: self.now)
            self.q.offer(h, "session")
            self.q.dispatch(lambda _: None)
        result = self.q.result()
        self.assertEqual(result["completed"], 18)
        self.assertEqual(len(result["history"]), 16)
        result["history"].clear()
        self.assertEqual(len(self.q.result()["history"]), 16)


if __name__ == "__main__":
    unittest.main()
