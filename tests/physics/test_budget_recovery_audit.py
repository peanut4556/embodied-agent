import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from embodied_agent.budget_recovery_audit import audit_manifest
from embodied_agent.budget_start_intent import save_intent
from embodied_agent.wait_budget_store import WaitBudgetStore


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = WaitBudgetStore.create(self.root / "pool.db", total_threshold_ns=10, slots=2)
        for name in ("a", "b"):
            save_intent(self.root / f"{name}.json", self.store, name)
        self.store.reserve("a")
        self.manifest = self.root / "manifest.json"

    def write(self, entries):
        self.manifest.write_text(json.dumps({"version": 1, "entries": entries}))

    def test_mixed_results_survive_corrupt_entry_and_do_not_mutate_pool(self):
        (self.root / "broken.json").write_text("{")
        self.write([{"intent": p} for p in ("a.json", "broken.json", "b.json")])
        before = self.store.snapshot()
        result = audit_manifest(self.manifest)
        self.assertEqual(result["counts"], {"reserved": 1, "unconfirmed": 1, "not_found": 1})
        self.assertEqual(result["exit_code"], 2)
        self.assertEqual(result["checked"], 3)
        self.assertEqual(before, self.store.snapshot())

    def test_duplicate_and_over_limit_rejected_before_any_query(self):
        for entries in (
            [{"intent": "a.json"}, {"intent": "./a.json"}],
            [{"intent": str(i)} for i in range(65)],
        ):
            self.write(entries)
            with (
                patch("embodied_agent.budget_recovery_audit.reconcile_intent") as lookup,
                self.assertRaises(ValueError),
            ):
                audit_manifest(self.manifest)
            lookup.assert_not_called()

    def test_missing_baseline_never_silently_downgrades(self):
        self.write([{"intent": "a.json", "checkpoint": "missing.json"}])
        result = audit_manifest(self.manifest)
        self.assertEqual(result["counts"], {"unconfirmed": 1})

    def test_cli_exit_codes_for_found_and_missing(self):
        for name, code in (("a", 0), ("b", 1)):
            self.write([{"intent": f"{name}.json"}])
            child = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_budget_recovery.py",
                    "--manifest",
                    str(self.manifest),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(child.returncode, code)
            self.assertFalse(json.loads(child.stdout)["execution_verified"])

    def test_oversize_and_empty_manifest_refused(self):
        self.write([])
        with self.assertRaises(ValueError):
            audit_manifest(self.manifest)
        self.manifest.write_text(" " * 65537)
        with self.assertRaisesRegex(ValueError, "size limit"):
            audit_manifest(self.manifest)


if __name__ == "__main__":
    unittest.main()
