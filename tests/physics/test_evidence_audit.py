import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from embodied_agent.checkpoint_store import CheckpointStore
from embodied_agent.evidence_checkpoint import CheckpointedEvidenceLedger
from scripts.check_evidence_checkpoint import receipt


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database = Path(self.temp.name) / "evidence.sqlite"
        self.store = Path(self.temp.name) / "store.sqlite"
        self.append()
        with CheckpointStore(self.store, create=True) as store:
            store.enroll(self.database)

    def append(self):
        with CheckpointedEvidenceLedger(self.database, clock=lambda: 1.002) as ledger:
            ledger.record(receipt())

    def run_cli(self, *extra):
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "embodied_agent.evidence_audit",
                "--database",
                str(self.database),
                "--store",
                str(self.store),
                *extra,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            env=os.environ | {"PYTHONPATH": "src:."},
            check=False,
        )
        return process.returncode, json.loads(process.stdout)

    def test_verified_and_read_only(self):
        before = self.database.read_bytes(), self.store.read_bytes()
        code, result = self.run_cli()
        self.assertEqual(code, 0)
        self.assertEqual(result["counts"], {"recorded": 1})
        self.assertFalse(result["robot_action_authorized"])
        self.assertEqual(before, (self.database.read_bytes(), self.store.read_bytes()))

    def test_tail_is_not_claimed_checkpoint_protected(self):
        self.append()
        code, result = self.run_cli()
        self.assertEqual(code, 6)
        self.assertEqual(result["examined"], 2)
        self.assertEqual(result["uncheckpointed_counts"], {"intents": 1, "records": 1})
        self.assertEqual(result["checkpoint_record_count"], 1)

    def test_limit_does_not_report_success(self):
        self.append()
        code, result = self.run_cli("--max-records", "1")
        self.assertEqual(code, 4)
        self.assertFalse(result["scan_complete"])

    def test_pending_intent(self):
        ticks = iter([1.002, 1.02])
        with (
            CheckpointedEvidenceLedger(self.database, clock=lambda: next(ticks)) as ledger,
            self.assertRaises(ValueError),
        ):
            ledger.record(receipt())
        code, result = self.run_cli()
        self.assertEqual(code, 3)
        self.assertEqual(result["counts"]["not_found"], 1)

    def test_history_loss_blocks(self):
        with sqlite3.connect(self.database) as connection:
            connection.execute("DELETE FROM recent_evidence_v1")
        code, result = self.run_cli()
        self.assertEqual(code, 5)
        self.assertEqual(result["cause"]["status"], "history_mismatch")

    def test_missing_store_never_created(self):
        self.store = self.store.with_name("missing.sqlite")
        self.assertEqual(self.run_cli()[0], 5)
        self.assertFalse(self.store.exists())

    def test_invalid_limit_and_alias(self):
        self.assertEqual(self.run_cli("--max-records", "0")[0], 5)
        self.store = self.database
        self.assertEqual(self.run_cli()[0], 5)

    def test_unanchored_corruption_blocks(self):
        self.append()
        with sqlite3.connect(self.database) as connection:
            connection.execute("UPDATE recent_evidence_v1 SET payload_json='{}' WHERE rowid=2")
        code, result = self.run_cli()
        self.assertEqual(code, 5)
        self.assertEqual(result["counts"]["integrity_error"], 1)

    def test_unanchored_index_deletion_is_detected(self):
        self.append()
        with sqlite3.connect(self.database) as connection:
            connection.execute("DELETE FROM evidence_intents_v1 WHERE sequence=2")
        before = self.database.read_bytes()
        code, result = self.run_cli()
        self.assertEqual(code, 5)
        self.assertTrue(result["checkpoint_history_verified"])
        self.assertEqual(result["record_counts"]["unindexed"], 1)
        self.assertEqual(self.database.read_bytes(), before)

    def test_legacy_writer_without_index_is_detected(self):
        from embodied_agent.evidence_ledger import EvidenceLedger

        with EvidenceLedger(self.database, clock=lambda: 1.002) as ledger:
            ledger.record(receipt())
        code, result = self.run_cli()
        self.assertEqual(code, 5)
        self.assertEqual(result["record_counts"]["unindexed"], 1)

    def test_reverse_limit_prevents_false_success(self):
        self.append()
        with sqlite3.connect(self.database) as connection:
            connection.execute("DELETE FROM evidence_intents_v1 WHERE sequence=2")
        code, result = self.run_cli("--max-records", "1")
        self.assertEqual(code, 4)
        self.assertTrue(result["index_scan_complete"])
        self.assertFalse(result["record_scan_complete"])

    def test_reverse_binding_mismatch(self):
        self.append()
        with sqlite3.connect(self.database) as connection:
            connection.execute("UPDATE evidence_intents_v1 SET session_id='wrong' WHERE sequence=2")
        code, result = self.run_cli()
        self.assertEqual(code, 5)
        self.assertEqual(result["record_counts"]["binding_conflict"], 1)

    def test_explicit_publication_closes_coverage_gap(self):
        self.append()
        self.assertEqual(self.run_cli()[0], 6)
        with CheckpointStore(self.store) as store:
            self.assertEqual(
                store.advance(self.database, expected_generation=1)["status"], "published"
            )
        code, result = self.run_cli()
        self.assertEqual(code, 0)
        self.assertEqual(result["uncheckpointed_counts"], {"intents": 0, "records": 0})
        self.assertEqual(result["checkpoint_generation"], 2)

    def test_partial_scan_does_not_estimate_coverage(self):
        self.append()
        code, result = self.run_cli("--max-records", "1")
        self.assertEqual(code, 4)
        self.assertIsNone(result["uncheckpointed_counts"])

    def test_pending_has_priority_over_uncheckpointed(self):
        ticks = iter([1.002, 1.02])
        with (
            CheckpointedEvidenceLedger(self.database, clock=lambda: next(ticks)) as ledger,
            self.assertRaises(ValueError),
        ):
            ledger.record(receipt())
        code, result = self.run_cli()
        self.assertEqual(code, 3)
        self.assertEqual(result["uncheckpointed_counts"], {"intents": 1, "records": 0})


if __name__ == "__main__":
    unittest.main()
