import json
import os
import sqlite3
import subprocess
import sys
import unittest

import test_verified_publication as fixtures

from embodied_agent.checkpoint_store import read_checkpoint_store


class CheckpointCLITests(unittest.TestCase):
    setUp = fixtures.VerifiedPublicationTests.setUp
    append = fixtures.VerifiedPublicationTests.append

    def invoke(self, operation, *extra):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "embodied_agent.checkpoint_cli",
                operation,
                "--database",
                str(self.database),
                "--store",
                str(self.store),
                *extra,
            ],
            env=os.environ | {"PYTHONPATH": "src:."},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.returncode, result.stdout

    def save_preview(self):
        code, output = self.invoke("preview")
        self.assertEqual(code, 0)
        path = self.store.with_name("proposal.json")
        path.write_text(output)
        return path

    def test_cross_process_publish_and_repeat(self):
        before = self.database.read_bytes(), self.store.read_bytes()
        proposal = self.save_preview()
        self.assertEqual(before, (self.database.read_bytes(), self.store.read_bytes()))
        code, output = self.invoke("publish", "--proposal", str(proposal))
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["generation"], 2)
        reconciled_code, reconciled = self.invoke("reconcile", "--proposal", str(proposal))
        self.assertEqual(reconciled_code, 0)
        self.assertEqual(json.loads(reconciled)["status"], "publication_recorded")
        self.assertEqual(self.invoke("publish", "--proposal", str(proposal))[0], 6)

    def test_source_growth_rejects_saved_proposal(self):
        proposal = self.save_preview()
        self.append()
        before = read_checkpoint_store(self.store)
        code, output = self.invoke("publish", "--proposal", str(proposal))
        self.assertEqual(code, 6)
        self.assertEqual(json.loads(output)["source"]["status"], "proposal_changed")
        self.assertEqual(read_checkpoint_store(self.store), before)

    def test_invalid_documents_never_publish(self):
        path = self.store.with_name("bad.json")
        for content in (
            "{",
            "[]",
            '{"status":"proposed","status":"proposed"}',
            "x" * 65537,
            '{"status":"proposed","robot_action_authorized":true}',
        ):
            path.write_text(content)
            self.assertEqual(self.invoke("publish", "--proposal", str(path))[0], 5)
            self.assertEqual(read_checkpoint_store(self.store)["generation"], 1)

    def test_missing_store_never_created(self):
        proposal = self.save_preview()
        self.store = self.store.with_name("missing.sqlite")
        self.assertEqual(self.invoke("preview")[0], 5)
        self.assertEqual(self.invoke("publish", "--proposal", str(proposal))[0], 5)
        self.assertFalse(self.store.exists())

    def test_required_proposal_and_invalid_limit(self):
        self.assertEqual(self.invoke("publish")[0], 2)
        self.assertEqual(self.invoke("preview", "--max-records", "0")[0], 5)

    def test_lock_contention_reports_unconfirmed(self):
        proposal = self.save_preview()
        connection = sqlite3.connect(self.store)
        try:
            connection.execute("BEGIN IMMEDIATE")
            code, output = self.invoke("publish", "--proposal", str(proposal))
            self.assertEqual(code, 7)
            self.assertEqual(json.loads(output)["status"], "publication_unconfirmed")
        finally:
            connection.close()
        self.assertEqual(read_checkpoint_store(self.store)["generation"], 1)


if __name__ == "__main__":
    unittest.main()
