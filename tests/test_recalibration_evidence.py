import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from embodied_agent.recalibration_evidence import verify_evidence
from scripts.validate_recalibration_lifecycle import validate


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.base.cleanup)
        cls.fixture = Path(cls.base.name) / "run"
        validate(cls.fixture)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "run"
        shutil.copytree(self.fixture, self.path)

    def alter_report(self, fn):
        path = self.path / "report.json"
        report = json.loads(path.read_text())
        fn(report)
        path.write_text(json.dumps(report))

    def test_valid_with_and_without_sources(self):
        before = {p.name: p.read_bytes() for p in self.path.glob("*.json")}
        result = verify_evidence(self.path)
        self.assertTrue(result["verified"])
        self.assertFalse(result["source_checked"])
        self.assertFalse(result["authenticated"])
        self.assertTrue(verify_evidence(self.path, source_root=Path.cwd())["verified"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.path.glob("*.json")})

    def test_missing_and_changed_artifacts(self):
        (self.path / "plan.json").unlink()
        (self.path / "budget.json").write_text("{}")
        result = verify_evidence(self.path)
        self.assertFalse(result["verified"])
        states = {r["path"]: r["status"] for r in result["artifacts"]}
        self.assertEqual(states["plan.json"], "missing")
        self.assertEqual(states["budget.json"], "changed")

    def test_wrong_stage_and_incomplete_manifest(self):
        self.alter_report(lambda r: r["stages"][0].update(exit_code=0))
        self.assertEqual(verify_evidence(self.path)["stage_issues"], ["diagnose-invalid"])
        self.alter_report(lambda r: r["artifact_sha256"].pop("initial.json"))
        with self.assertRaises(ValueError):
            verify_evidence(self.path)

    def test_path_escape_and_symlink(self):
        self.alter_report(lambda r: r["source_sha256"].update({"../outside": "0" * 64}))
        with self.assertRaises(ValueError):
            verify_evidence(self.path)
        shutil.copyfile(self.fixture / "report.json", self.path / "report.json")
        target = self.path / "initial.json"
        target.unlink()
        target.symlink_to(self.fixture / "initial.json")
        with self.assertRaises(ValueError):
            verify_evidence(self.path)

    def test_oversized_file_and_duplicate_json(self):
        (self.path / "plan.json").write_bytes(b"x" * 1_048_577)
        with self.assertRaises(ValueError):
            verify_evidence(self.path)
        (self.path / "report.json").write_text('{"passed":true,"passed":true}')
        with self.assertRaises(ValueError):
            verify_evidence(self.path)

    def test_missing_sources_is_not_matched(self):
        result = verify_evidence(self.path, source_root=self.path)
        self.assertFalse(result["verified"])
        self.assertTrue(result["source_checked"])
        self.assertTrue(all(r["status"] == "missing" for r in result["sources"]))

    def test_cli_exit_codes(self):
        def run():
            return subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "embodied_agent.recalibration_evidence",
                    "--directory",
                    str(self.path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

        self.assertEqual(run().returncode, 0)
        (self.path / "plan.json").unlink()
        self.assertEqual(run().returncode, 1)
        (self.path / "report.json").write_text("{}")
        child = run()
        self.assertEqual(child.returncode, 3)
        self.assertEqual(child.stdout, "")
