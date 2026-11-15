import sqlite3
import unittest

import test_verified_publication as fixtures

from embodied_agent.checkpoint_store import (
    CheckpointStore,
    preview_publication,
    reconcile_publication,
)


class PublicationReceiptTests(unittest.TestCase):
    setUp = fixtures.VerifiedPublicationTests.setUp
    append = fixtures.VerifiedPublicationTests.append

    def proposal(self):
        return preview_publication(self.database, self.store)["proposal"]

    def test_history_survives_later_publication(self):
        proposal = self.proposal()
        with CheckpointStore(self.store) as store:
            store.publish_reviewed(self.database, proposal)
        self.append()
        with CheckpointStore(self.store) as store:
            store.advance_verified(self.database, expected_generation=2)
        before = self.store.read_bytes()
        result = reconcile_publication(self.store, proposal)
        self.assertEqual(result["status"], "publication_recorded")
        self.assertEqual(result["generation"], 2)
        self.assertEqual(result["current_generation"], 3)
        self.assertTrue(result["historical_only"])
        self.assertEqual(self.store.read_bytes(), before)

    def test_no_record_is_not_success(self):
        proposal = self.proposal()
        self.assertEqual(
            reconcile_publication(self.store, proposal)["status"], "no_publication_record"
        )
        with CheckpointStore(self.store) as store:
            store.advance_verified(self.database, expected_generation=1)
        self.assertEqual(
            reconcile_publication(self.store, proposal)["status"], "no_publication_record"
        )

    def test_corrupt_receipt_is_rejected(self):
        proposal = self.proposal()
        with CheckpointStore(self.store) as store:
            store.publish_reviewed(self.database, proposal)
        with sqlite3.connect(self.store) as connection:
            connection.execute("UPDATE checkpoint_publications_v1 SET checksum='bad'")
        self.assertEqual(reconcile_publication(self.store, proposal)["status"], "integrity_error")

    def test_unknown_proposal_and_missing_store(self):
        proposal = self.proposal()
        missing = self.store.with_name("missing.sqlite")
        self.assertEqual(reconcile_publication(missing, proposal)["status"], "unavailable")
        self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
