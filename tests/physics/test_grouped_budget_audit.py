import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embodied_agent.budget_recovery_audit import audit_manifest
from embodied_agent.budget_start_intent import save_intent
from embodied_agent.wait_budget_store import WaitBudgetStore


class GroupedAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pool = WaitBudgetStore.create(self.root / "pool.db", total_threshold_ns=10, slots=2)
        self.manifest = self.root / "manifest.json"
        for name, identity in (("a", "a"), ("b", "b"), ("alias", "a")):
            save_intent(self.root / f"{name}.json", self.pool, identity)
        self.pool.reserve("a")

    def audit(self, entries):
        self.manifest.write_text(json.dumps({"version": 1, "entries": entries}))
        return audit_manifest(self.manifest)

    def test_one_query_and_duplicate_identity_preserves_file_results(self):
        original = WaitBudgetStore.lookup_many
        calls = []

        def tracked(store, identities):
            calls.append(identities)
            return original(store, identities)

        with patch.object(WaitBudgetStore, "lookup_many", tracked):
            result = self.audit([{"intent": f"{n}.json"} for n in ("b", "alias", "a")])
        self.assertEqual(calls, [["b", "a"]])
        self.assertEqual(result["counts"], {"not_found": 1, "reserved": 2})
        self.assertTrue(result["single_database_snapshot"])
        self.assertEqual(result["snapshot_groups"][0]["request_count"], 2)

    def test_different_baselines_never_silently_merge(self):
        (self.root / "baseline.json").write_text(json.dumps(self.pool.checkpoint()))
        result = self.audit(
            [{"intent": "a.json"}, {"intent": "b.json", "checkpoint": "baseline.json"}]
        )
        self.assertEqual(len(result["snapshot_groups"]), 2)
        self.assertFalse(result["single_database_snapshot"])

    def test_bad_group_does_not_poison_valid_group(self):
        invalid = self.pool.checkpoint() | {"sha256": "0" * 64}
        (self.root / "bad.json").write_text(json.dumps(invalid))
        result = self.audit([{"intent": "a.json", "checkpoint": "bad.json"}, {"intent": "b.json"}])
        self.assertEqual(result["counts"], {"unconfirmed": 1, "not_found": 1})
        self.assertFalse(result["single_database_snapshot"])

    def test_identical_baseline_contents_share_snapshot(self):
        for name in ("one", "two"):
            (self.root / f"{name}.json").write_text(json.dumps(self.pool.checkpoint()))
        result = self.audit(
            [
                {"intent": "a.json", "checkpoint": "one.json"},
                {"intent": "b.json", "checkpoint": "two.json"},
            ]
        )
        self.assertTrue(result["single_database_snapshot"])
        self.assertEqual(len(result["snapshot_groups"]), 1)


if __name__ == "__main__":
    unittest.main()
