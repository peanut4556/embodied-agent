import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embodied_agent.checkpoint_store import CheckpointStore, read_checkpoint_store
from embodied_agent.evidence_audit import audit_evidence
from embodied_agent.evidence_checkpoint import CheckpointedEvidenceLedger, _publication_gate
from scripts.check_evidence_checkpoint import receipt


class VerifiedPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database = Path(self.temp.name) / "evidence.sqlite"
        self.store = Path(self.temp.name) / "store.sqlite"
        self.append()
        with CheckpointStore(self.store, create=True) as store:
            store.enroll(self.database)
        self.append()

    def append(self):
        with CheckpointedEvidenceLedger(self.database, clock=lambda: 1.002) as ledger:
            ledger.record(receipt())

    def publish(self, **kwargs):
        with CheckpointStore(self.store) as store:
            return store.advance_verified(self.database, expected_generation=1, **kwargs)

    def test_valid_tail_becomes_covered(self):
        self.assertEqual(self.publish()["status"], "published")
        self.assertEqual(audit_evidence(self.database, self.store)["status"], "verified")

    def test_faults_preserve_previous_head(self):
        for sql in (
            "DELETE FROM evidence_intents_v1 WHERE sequence=2",
            "DELETE FROM recent_evidence_v1 WHERE rowid=2",
            "UPDATE recent_evidence_v1 SET accepted_host_time=2 WHERE rowid=2",
            "UPDATE recent_evidence_v1 SET payload_json='{}' WHERE rowid=2",
        ):
            with self.subTest(sql=sql):
                # Save a clean database for each fault injection.
                original = self.database.read_bytes()
                before = read_checkpoint_store(self.store)
                with sqlite3.connect(self.database) as connection:
                    connection.execute(sql)
                result = self.publish()
                self.assertEqual(result["status"], "source_rejected")
                self.assertEqual(result["source"]["status"], "evidence_rejected")
                self.assertEqual(read_checkpoint_store(self.store), before)
                self.database.write_bytes(original)

    def test_bound_is_not_partial_approval(self):
        self.assertEqual(self.publish(max_records=1)["source"]["status"], "validation_incomplete")
        self.assertEqual(read_checkpoint_store(self.store)["generation"], 1)

    def test_stale_request_refused(self):
        self.publish()
        self.assertEqual(self.publish()["status"], "generation_conflict")

    def test_invalid_bound_refused(self):
        for bound in (True, 0, 100001):
            with self.assertRaises(ValueError):
                self.publish(max_records=bound)

    def test_validation_and_capture_share_snapshot(self):
        with sqlite3.connect(self.database) as connection:
            connection.execute("PRAGMA journal_mode=WAL")

        def mutate_after_gate(connection, bound):
            result = _publication_gate(connection, bound)
            with sqlite3.connect(self.database) as writer:
                writer.execute("DELETE FROM recent_evidence_v1 WHERE rowid=2")
            return result

        with patch("embodied_agent.evidence_checkpoint._publication_gate", mutate_after_gate):
            self.assertEqual(self.publish()["status"], "published")
        head = read_checkpoint_store(self.store)
        self.assertEqual(head["checkpoint"]["records"]["count"], 2)
        self.assertEqual(audit_evidence(self.database, self.store)["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
