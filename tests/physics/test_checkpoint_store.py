import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from embodied_agent.checkpoint_store import (
    CheckpointStore,
    discover_from_store,
    read_checkpoint_store,
)
from embodied_agent.evidence_checkpoint import CheckpointedEvidenceLedger
from scripts.check_evidence_checkpoint import receipt


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database = Path(self.temp.name) / "evidence.sqlite"
        self.store = Path(self.temp.name) / "checkpoint.sqlite"
        self.append()
        with CheckpointStore(self.store, create=True) as store:
            self.assertEqual(store.enroll(self.database)["status"], "published")

    def append(self):
        with CheckpointedEvidenceLedger(self.database, clock=lambda: 1.002) as ledger:
            ledger.record(receipt())

    def test_reopen_advance_and_discover(self):
        self.append()
        with CheckpointStore(self.store) as store:
            self.assertEqual(store.advance(self.database, expected_generation=1)["generation"], 2)
        self.assertEqual(read_checkpoint_store(self.store)["checkpoint"]["records"]["count"], 2)
        page = discover_from_store(self.database, self.store)
        self.assertEqual(page["checkpoint_generation"], 2)
        self.assertEqual(len(page["entries"]), 2)
        self.assertFalse(page["robot_action_authorized"])

    def test_stale_publisher_cannot_replace_newer_head(self):
        with CheckpointStore(self.store) as first, CheckpointStore(self.store) as second:
            self.append()
            self.assertEqual(
                first.advance(self.database, expected_generation=1)["status"], "published"
            )
            before = read_checkpoint_store(self.store)
            self.assertEqual(
                second.advance(self.database, expected_generation=1)["status"],
                "generation_conflict",
            )
            self.assertEqual(read_checkpoint_store(self.store), before)

    def test_deleted_history_does_not_update_head(self):
        before = read_checkpoint_store(self.store)
        with sqlite3.connect(self.database) as connection:
            connection.execute("DELETE FROM recent_evidence_v1")
        with CheckpointStore(self.store) as store:
            result = store.advance(self.database, expected_generation=1)
        self.assertEqual(result["source"]["status"], "history_mismatch")
        self.assertEqual(read_checkpoint_store(self.store), before)
        self.assertEqual(discover_from_store(self.database, self.store)["entries"], [])

    def test_explicit_enrollment_cannot_overwrite(self):
        with CheckpointStore(self.store, create=True) as store:
            self.assertEqual(store.enroll(self.database)["status"], "already_enrolled")
        self.assertEqual(read_checkpoint_store(self.store)["generation"], 1)

    def test_missing_store_not_created(self):
        missing = self.store.with_name("missing.sqlite")
        self.assertEqual(read_checkpoint_store(missing)["status"], "unavailable")
        self.assertEqual(discover_from_store(self.database, missing)["entries"], [])
        with self.assertRaises(sqlite3.OperationalError):
            CheckpointStore(missing)
        self.assertFalse(missing.exists())

    def test_empty_store_requires_explicit_enrollment(self):
        empty = self.store.with_name("empty.sqlite")
        with CheckpointStore(empty, create=True) as store:
            self.assertEqual(
                store.advance(self.database, expected_generation=1)["status"], "unenrolled"
            )
        self.assertEqual(discover_from_store(self.database, empty)["status"], "unenrolled")

    def test_corruption_blocks_read_and_publication(self):
        with sqlite3.connect(self.store) as connection:
            connection.execute("UPDATE checkpoint_head_v1 SET checkpoint_json='{}'")
        self.assertEqual(read_checkpoint_store(self.store)["status"], "integrity_error")
        with CheckpointStore(self.store) as store:
            self.assertEqual(
                store.advance(self.database, expected_generation=1)["status"], "integrity_error"
            )
            self.assertEqual(store.enroll(self.database)["status"], "integrity_error")
        self.assertEqual(discover_from_store(self.database, self.store)["entries"], [])

    def test_alias_and_hardlink_rejected(self):
        alias = self.store.with_name("alias.sqlite")
        os.link(self.store, alias)
        with CheckpointStore(self.store) as store:
            for path in (self.store, alias):
                with self.assertRaises(ValueError):
                    store.enroll(path)
                with self.assertRaises(ValueError):
                    discover_from_store(path, self.store)

    def test_lock_failure_is_unconfirmed_and_preserves_head(self):
        other = sqlite3.connect(self.store, timeout=0)
        try:
            with CheckpointStore(self.store) as store:
                other.execute("BEGIN IMMEDIATE")
                result = store.advance(self.database, expected_generation=1)
                self.assertEqual(result["status"], "publication_unconfirmed")
        finally:
            other.close()
        self.assertEqual(read_checkpoint_store(self.store)["generation"], 1)

    def test_bad_generation_and_missing_source(self):
        with CheckpointStore(self.store) as store:
            for value in (True, 0, -1, 2**63 - 1):
                with self.assertRaises(ValueError):
                    store.advance(self.database, expected_generation=value)
            result = store.advance(self.database.with_name("missing.sqlite"), expected_generation=1)
        self.assertEqual(result["status"], "source_rejected")
        self.assertEqual(read_checkpoint_store(self.store)["generation"], 1)

    def test_discovery_does_not_write(self):
        before = (self.database.read_bytes(), self.store.read_bytes())
        discover_from_store(self.database, self.store)
        self.assertEqual(before, (self.database.read_bytes(), self.store.read_bytes()))

    def test_generation_corruption_is_detected(self):
        with sqlite3.connect(self.store) as connection:
            connection.execute("UPDATE checkpoint_head_v1 SET generation=7")
        self.assertEqual(read_checkpoint_store(self.store)["status"], "integrity_error")


if __name__ == "__main__":
    unittest.main()
