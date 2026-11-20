import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from embodied_agent.budget_recovery_audit import audit_manifest
from embodied_agent.budget_start_intent import read_intent
from embodied_agent.recovery_json import read_recovery_json
from embodied_agent.wait_budget_store import WaitBudgetStore


class RecoveryJsonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "input.json"

    def test_duplicate_intent_identity_rejected(self):
        self.path.write_text(
            '{"version":1,"database":"/tmp/pool.db","pool_id":"p","request_id":"reviewed","request_id":"other"}'
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_intent(self.path)

    def test_nested_duplicate_manifest_field_rejected(self):
        self.path.write_text('{"version":1,"entries":[{"intent":"one","intent":"two"}]}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit_manifest(self.path)

    def test_constants_depth_and_limit_are_controlled_errors(self):
        for raw in ("NaN", "Infinity", "-Infinity", "[" * 2000 + "0" + "]" * 2000):
            self.path.write_text(raw)
            with self.assertRaises(ValueError):
                read_recovery_json(self.path, 8192)
        self.path.write_text(" " * 8193)
        with self.assertRaisesRegex(ValueError, "size limit"):
            read_recovery_json(self.path, 8192)

    def test_cli_ambiguous_or_oversized_checkpoint_unconfirmed(self):
        store = WaitBudgetStore.create(self.root / "pool.db", total_threshold_ns=10, slots=1)
        checkpoint = store.checkpoint()
        ambiguous = json.dumps(checkpoint)[:-1] + ',"reserved_slots":0}'
        for raw in (ambiguous, " " * 8193):
            self.path.write_text(raw)
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/lookup_wait_budget.py",
                    "--database",
                    str(store.path),
                    "--expected-pool-id",
                    store.pool_id,
                    "--request-id",
                    "a",
                    "--checkpoint",
                    str(self.path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "unconfirmed")
        self.assertEqual(store.snapshot()["reserved_slots"], 0)


if __name__ == "__main__":
    unittest.main()
