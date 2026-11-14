import copy
import sqlite3
import tempfile
import unittest
from pathlib import Path

from test_recent_evidence_handoff import confirmed_controller

from embodied_agent.evidence_inbox import EvidenceInbox
from embodied_agent.indexed_evidence_ledger import IndexedEvidenceLedger, discover_evidence
from embodied_agent.recent_evidence_handoff import RecentEvidenceHandoff


class IndexedLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "audit.sqlite"
        controller = confirmed_controller()
        self.now = controller.last_host
        self.receipt = RecentEvidenceHandoff(controller, clock=lambda: self.now).consume("session")
        self.ledger = IndexedEvidenceLedger(self.path, clock=lambda: self.now)
        self.addCleanup(self.ledger.close)

    def test_discovery_without_identity_and_historical_duplicate(self):
        self.ledger.record(self.receipt)
        self.now += 100
        self.assertTrue(self.ledger.record(self.receipt)["duplicate"])
        page = discover_evidence(self.path)
        self.assertEqual(len(page["entries"]), 1)
        self.assertEqual(page["entries"][0]["reconciliation"]["status"], "recorded")
        self.assertNotIn("evidence", str(page))
        self.assertFalse(page["robot_action_authorized"])

    def test_intent_commit_cost_expires_receipt_without_insertion(self):
        ticks = iter([self.now, self.receipt["expires_host_time"]])
        self.ledger.clock = lambda: next(ticks)
        with self.assertRaises(ValueError):
            self.ledger.record(self.receipt)
        page = discover_evidence(self.path)
        self.assertEqual(page["entries"][0]["reconciliation"]["status"], "not_found")
        self.assertEqual(discover_evidence(self.path), page)  # read-only, no replay

    def test_conflict_cannot_overwrite_intent(self):
        self.ledger.record(self.receipt)
        changed = copy.deepcopy(self.receipt)
        changed["session_id"] = "other"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.ledger.record(changed)
        self.assertEqual(discover_evidence(self.path)["entries"][0]["session_id"], "session")

    def test_pagination(self):
        for number in range(3):
            self.ledger.record(self.receipt | {"receipt_id": f"{number:032x}"})
        first = discover_evidence(self.path, limit=2)
        second = discover_evidence(self.path, after=first["next_after"], limit=2)
        self.assertEqual([e["sequence"] for e in first["entries"] + second["entries"]], [1, 2, 3])
        self.assertEqual(discover_evidence(self.path, after=second["next_after"])["entries"], [])

    def test_missing_database_does_not_create_or_claim_empty(self):
        missing = self.path.with_name("missing.sqlite")
        self.assertEqual(discover_evidence(missing)["status"], "unavailable")
        self.assertFalse(missing.exists())

    def test_intent_binding_conflict_and_payload_corruption(self):
        self.ledger.record(self.receipt)
        self.ledger._connection.execute("UPDATE evidence_intents_v1 SET session_id='other'")
        self.ledger._connection.commit()
        self.assertEqual(
            discover_evidence(self.path)["entries"][0]["reconciliation"]["status"],
            "intent_conflict",
        )
        self.ledger._connection.execute("UPDATE recent_evidence_v1 SET payload_json='{}'")
        self.ledger._connection.commit()
        self.assertEqual(
            discover_evidence(self.path)["entries"][0]["reconciliation"]["status"],
            "integrity_error",
        )

    def test_invalid_receipt_has_no_intent(self):
        with self.assertRaises(ValueError):
            self.ledger.record(self.receipt | {"robot_action_authorized": True})
        self.assertEqual(discover_evidence(self.path)["entries"], [])

    def test_invalid_page_bounds(self):
        for kwargs in (
            {"after": True},
            {"after": -1},
            {"after": 2**63},
            {"limit": 0},
            {"limit": 1001},
        ):
            with self.assertRaises(ValueError):
                discover_evidence(self.path, **kwargs)

    def test_inbox_dispatch_to_indexed_sink(self):
        controller = confirmed_controller()
        handoff = RecentEvidenceHandoff(controller, clock=lambda: self.now)
        inbox = EvidenceInbox(clock=lambda: self.now)
        self.assertEqual(inbox.offer(handoff, "session")["status"], "queued")
        self.assertEqual(inbox.dispatch(self.ledger.record)["status"], "delivered")
        self.assertEqual(inbox.dispatch(self.ledger.record)["reason"], "no_pending_evidence")
        entries = discover_evidence(self.path)["entries"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["reconciliation"]["status"], "recorded")

    def test_locked_writer_fails_without_intent(self):
        other = sqlite3.connect(self.path, timeout=0)
        try:
            other.execute("BEGIN IMMEDIATE")
            with self.assertRaises(sqlite3.OperationalError):
                self.ledger.record(self.receipt)
        finally:
            other.close()
        self.assertEqual(discover_evidence(self.path)["entries"], [])


if __name__ == "__main__":
    unittest.main()
