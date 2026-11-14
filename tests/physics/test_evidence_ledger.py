import copy
import tempfile
import unittest
from pathlib import Path

from test_recent_evidence_handoff import confirmed_controller

from embodied_agent.evidence_ledger import EvidenceLedger, reconcile_evidence
from embodied_agent.recent_evidence_handoff import RecentEvidenceHandoff


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "evidence.sqlite"
        c = confirmed_controller()
        self.now = c.last_host
        self.receipt = RecentEvidenceHandoff(c, clock=lambda: self.now).consume("session")
        self.ledger = EvidenceLedger(self.path, clock=lambda: self.now)
        self.addCleanup(self.ledger.close)

    def test_record_survives_reopen_and_duplicate_does_not_write(self):
        first = self.ledger.record(self.receipt)
        self.assertEqual(first["status"], "recorded")
        self.assertFalse(first["duplicate"])
        self.now += 100  # Old records remain historical, never become fresh again.
        second = self.ledger.record(self.receipt)
        self.assertTrue(second["duplicate"])
        result = reconcile_evidence(self.path, self.receipt["receipt_id"])
        self.assertEqual(result["payload_sha256"], first["payload_sha256"])
        self.assertFalse(result["robot_action_authorized"])
        self.assertNotIn("evidence", result)

    def test_conflicting_payload_cannot_replace_same_identity(self):
        original = self.ledger.record(self.receipt)
        changed = copy.deepcopy(self.receipt)
        changed["session_id"] = "different"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.ledger.record(changed)
        self.assertEqual(
            reconcile_evidence(self.path, self.receipt["receipt_id"])["payload_sha256"],
            original["payload_sha256"],
        )

    def test_expiry_before_insert_leaves_no_row(self):
        self.now = self.receipt["expires_host_time"]
        with self.assertRaises(ValueError):
            self.ledger.record(self.receipt)
        self.assertEqual(
            reconcile_evidence(self.path, self.receipt["receipt_id"])["status"], "not_found"
        )

    def test_expiry_after_insert_rolls_back_transaction(self):
        ticks = iter((self.now, self.receipt["expires_host_time"]))
        self.ledger.clock = lambda: next(ticks)
        with self.assertRaises(ValueError):
            self.ledger.record(self.receipt)
        self.assertEqual(
            reconcile_evidence(self.path, self.receipt["receipt_id"])["status"], "not_found"
        )

    def test_unavailable_database_is_not_missing_record(self):
        missing = Path(self.temp.name) / "missing.sqlite"
        self.assertEqual(
            reconcile_evidence(missing, self.receipt["receipt_id"])["status"], "unavailable"
        )
        self.assertFalse(missing.exists())

    def test_corrupt_payload_reports_integrity_error(self):
        self.ledger.record(self.receipt)
        self.ledger._connection.execute("UPDATE recent_evidence_v1 SET payload_json='{}'")
        self.ledger._connection.commit()
        self.assertEqual(
            reconcile_evidence(self.path, self.receipt["receipt_id"])["status"], "integrity_error"
        )

    def test_invalid_receipt_and_oversize_are_rejected(self):
        for changed in (
            self.receipt | {"receipt_id": "bad"},
            self.receipt | {"robot_action_authorized": True},
            self.receipt | {"expires_host_time": True},
            self.receipt | {"padding": "x" * (1024 * 1024)},
        ):
            with self.assertRaises(ValueError):
                self.ledger.record(changed)

    def test_clock_regression_rolls_back(self):
        self.now -= 1
        with self.assertRaises(ValueError):
            self.ledger.record(self.receipt)
        self.assertEqual(
            reconcile_evidence(self.path, self.receipt["receipt_id"])["status"], "not_found"
        )


if __name__ == "__main__":
    unittest.main()
