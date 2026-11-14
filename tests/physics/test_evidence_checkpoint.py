import copy
import hashlib
import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_recent_evidence_handoff import confirmed_controller

from embodied_agent.evidence_checkpoint import (
    CheckpointedEvidenceLedger,
    checkpoint_evidence,
    discover_checkpointed,
)
from embodied_agent.indexed_evidence_ledger import IndexedEvidenceLedger
from embodied_agent.indexed_evidence_ledger import _discover_connection as original_discover
from embodied_agent.recent_evidence_handoff import RecentEvidenceHandoff


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "audit.sqlite"
        controller = confirmed_controller()
        self.now = controller.last_host
        self.receipt = RecentEvidenceHandoff(controller, clock=lambda: self.now).consume("session")
        with CheckpointedEvidenceLedger(self.path, clock=lambda: self.now) as ledger:
            ledger.record(self.receipt)
        self.anchor = checkpoint_evidence(self.path)["checkpoint"]

    def test_reopen_and_append_preserves_anchor_and_pagination(self):
        with CheckpointedEvidenceLedger(self.path, clock=lambda: self.now) as ledger:
            ledger.record(self.receipt | {"receipt_id": "a" * 32})
        first = discover_checkpointed(self.path, self.anchor, limit=1)
        second = discover_checkpointed(self.path, self.anchor, after=first["next_after"])
        self.assertTrue(first["checkpoint_verified"])
        self.assertEqual(len(first["entries"]) + len(second["entries"]), 2)
        self.assertFalse(second["robot_action_authorized"])

    def test_different_database_rejected_even_if_empty(self):
        other = self.path.with_name("other.sqlite")
        with CheckpointedEvidenceLedger(other):
            pass
        result = discover_checkpointed(other, self.anchor)
        self.assertEqual(result["status"], "database_mismatch")
        self.assertEqual(result["entries"], [])

    def test_missing_file_is_not_created(self):
        missing = self.path.with_name("missing.sqlite")
        self.assertEqual(discover_checkpointed(missing, self.anchor)["status"], "unavailable")
        self.assertFalse(missing.exists())

    def test_unenrolled_database_is_not_silently_enrolled(self):
        other = self.path.with_name("legacy.sqlite")
        with IndexedEvidenceLedger(other):
            pass
        before = other.read_bytes()
        self.assertEqual(checkpoint_evidence(other)["status"], "unavailable")
        self.assertEqual(discover_checkpointed(other, self.anchor)["status"], "unavailable")
        self.assertEqual(other.read_bytes(), before)

    def test_record_loss_does_not_become_not_found(self):
        with sqlite3.connect(self.path) as connection:
            connection.execute("DELETE FROM recent_evidence_v1")
        result = discover_checkpointed(self.path, self.anchor)
        self.assertEqual((result["status"], result["component"]), ("history_mismatch", "records"))
        self.assertEqual(result["entries"], [])

    def test_intent_loss_is_detected(self):
        with sqlite3.connect(self.path) as connection:
            connection.execute("DELETE FROM evidence_intents_v1")
        self.assertEqual(discover_checkpointed(self.path, self.anchor)["component"], "intents")

    def test_consistent_rewrite_of_payload_and_hash_is_detected(self):
        changed = self.receipt | {"extra": "changed"}
        payload = json.dumps(changed, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode()).hexdigest()
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "UPDATE recent_evidence_v1 SET payload_json=?, payload_sha256=?", (payload, digest)
            )
            connection.execute("UPDATE evidence_intents_v1 SET payload_sha256=?", (digest,))
        self.assertEqual(
            discover_checkpointed(self.path, self.anchor)["status"], "history_mismatch"
        )

    def test_pending_intent_may_gain_record(self):
        pending = self.path.with_name("pending.sqlite")
        ticks = iter([self.now, self.receipt["expires_host_time"]])
        with (
            CheckpointedEvidenceLedger(pending, clock=lambda: next(ticks)) as ledger,
            self.assertRaises(ValueError),
        ):
            ledger.record(self.receipt)
        anchor = checkpoint_evidence(pending)["checkpoint"]
        with CheckpointedEvidenceLedger(pending, clock=lambda: self.now) as ledger:
            ledger.record(self.receipt)
        result = discover_checkpointed(pending, anchor)
        self.assertEqual(result["entries"][0]["reconciliation"]["status"], "recorded")

    def test_same_identity_old_backup_is_rejected(self):
        backup = self.path.with_name("backup.sqlite")
        shutil.copyfile(self.path, backup)
        with CheckpointedEvidenceLedger(self.path, clock=lambda: self.now) as ledger:
            ledger.record(self.receipt | {"receipt_id": "b" * 32})
        newer = checkpoint_evidence(self.path)["checkpoint"]
        self.assertEqual(discover_checkpointed(backup, newer)["status"], "history_mismatch")

    def test_unanchored_tail_loss_is_explicit_limit(self):
        backup = self.path.with_name("backup.sqlite")
        shutil.copyfile(self.path, backup)
        with CheckpointedEvidenceLedger(self.path, clock=lambda: self.now) as ledger:
            ledger.record(self.receipt | {"receipt_id": "b" * 32})
        self.assertEqual(discover_checkpointed(backup, self.anchor)["status"], "scanned")

    def test_reads_do_not_modify_database(self):
        before = self.path.read_bytes()
        checkpoint_evidence(self.path)
        discover_checkpointed(self.path, self.anchor)
        self.assertEqual(self.path.read_bytes(), before)

    def test_verification_and_page_share_snapshot_during_concurrent_change(self):
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA journal_mode=WAL")

        def mutate_then_read(connection, after, limit):
            with sqlite3.connect(self.path) as writer:
                writer.execute("DELETE FROM recent_evidence_v1")
            return original_discover(connection, after, limit)

        with patch("embodied_agent.evidence_checkpoint._discover_connection", mutate_then_read):
            page = discover_checkpointed(self.path, self.anchor)
        self.assertEqual(page["entries"][0]["reconciliation"]["status"], "recorded")
        self.assertEqual(
            discover_checkpointed(self.path, self.anchor)["status"], "history_mismatch"
        )

    def test_invalid_anchors_rejected(self):
        for changes in (
            {"version": True},
            {"database_id": "bad"},
            {"intents": {}},
            {"records": self.anchor["records"] | {"high_water": -1}},
        ):
            with self.assertRaises(ValueError):
                discover_checkpointed(self.path, copy.deepcopy(self.anchor) | changes)
        for kwargs in ({"after": True}, {"limit": 1001}):
            with self.assertRaises(ValueError):
                discover_checkpointed(self.path, self.anchor, **kwargs)

    def test_advancing_anchor_requires_previous_history(self):
        with CheckpointedEvidenceLedger(self.path, clock=lambda: self.now) as ledger:
            ledger.record(self.receipt | {"receipt_id": "c" * 32})
        advanced = checkpoint_evidence(self.path, previous=self.anchor)
        self.assertEqual(advanced["checkpoint"]["records"]["count"], 2)
        with sqlite3.connect(self.path) as connection:
            connection.execute("DELETE FROM recent_evidence_v1")
        failed = checkpoint_evidence(self.path, previous=advanced["checkpoint"])
        self.assertEqual(failed["status"], "history_mismatch")
        self.assertNotIn("checkpoint", failed)


if __name__ == "__main__":
    unittest.main()
