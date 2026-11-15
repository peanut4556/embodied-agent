import unittest

import test_verified_publication as fixtures

from embodied_agent.checkpoint_store import (
    CheckpointStore,
    preview_publication,
    read_checkpoint_store,
)


class ReviewedPublicationTests(unittest.TestCase):
    setUp = fixtures.VerifiedPublicationTests.setUp
    append = fixtures.VerifiedPublicationTests.append

    def preview(self):
        return preview_publication(self.database, self.store)["proposal"]

    def publish(self, proposal):
        with CheckpointStore(self.store) as store:
            return store.publish_reviewed(self.database, proposal)

    def test_read_only_preview_and_exact_publication(self):
        before = self.database.read_bytes(), self.store.read_bytes()
        proposal = self.preview()
        self.assertEqual(before, (self.database.read_bytes(), self.store.read_bytes()))
        self.assertEqual(self.publish(proposal)["status"], "published")
        self.assertEqual(read_checkpoint_store(self.store)["checkpoint"], proposal["checkpoint"])

    def test_valid_append_rejects_old_proposal_without_writing(self):
        proposal = self.preview()
        self.append()
        before = read_checkpoint_store(self.store)
        self.assertEqual(self.publish(proposal)["source"]["status"], "proposal_changed")
        self.assertEqual(read_checkpoint_store(self.store), before)
        self.assertEqual(self.publish(self.preview())["status"], "published")

    def test_generation_change_and_repeat_are_refused(self):
        proposal = self.preview()
        self.publish(proposal)
        self.assertEqual(self.publish(proposal)["status"], "generation_conflict")

    def test_missing_store_preview_never_creates(self):
        missing = self.store.with_name("missing.sqlite")
        self.assertEqual(preview_publication(self.database, missing)["status"], "unavailable")
        self.assertFalse(missing.exists())

    def test_invalid_proposals_and_limit(self):
        proposal = self.preview()
        for value in ({}, proposal | {"generation": True}, proposal | {"checkpoint": {}}):
            with self.assertRaises(ValueError):
                self.publish(value)
        result = preview_publication(self.database, self.store, max_records=1)
        self.assertEqual(result["source"]["status"], "validation_incomplete")


if __name__ == "__main__":
    unittest.main()
