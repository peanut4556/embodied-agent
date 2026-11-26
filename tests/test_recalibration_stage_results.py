import json
import subprocess
import sys
import unittest

import test_recalibration_evidence as fixtures

from embodied_agent.recalibration_evidence import verify_evidence


class StageResultTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.EvidenceTests.setUpClass.__func__)
    setUp = fixtures.EvidenceTests.setUp
    alter_report = fixtures.EvidenceTests.alter_report

    def mutate(self, stage, change, channel="stdout"):
        def edit(report):
            row = next(r for r in report["stages"] if r["name"] == stage)
            payload = json.loads(row[channel])
            change(payload)
            row[channel] = json.dumps(payload)

        self.alter_report(edit)

    def verify(self):
        return verify_evidence(self.path, check_results=True)

    def test_valid_original_and_explicit_scope(self):
        result = self.verify()
        self.assertTrue(result["verified"])
        self.assertTrue(result["stage_results_checked"])
        self.assertFalse(verify_evidence(self.path)["stage_results_checked"])
        self.assertEqual(result["stage_result_issues"], [])

    def test_exit_codes_alone_miss_false_drift_status(self):
        self.mutate(
            "status-after-drift", lambda p: p["result"].update(currently_blocked_actions=[])
        )
        self.assertTrue(verify_evidence(self.path)["verified"])
        result = self.verify()
        self.assertFalse(result["verified"])
        self.assertEqual(result["stage_result_issues"][0]["stage"], "status-after-drift")

    def test_apply_must_match_preview(self):
        self.mutate("apply-current", lambda p: p["result"].update(after_revision=3))
        self.assertFalse(self.verify()["verified"])

    def test_error_reason_and_channel_are_checked(self):
        self.mutate("reject-stale-preview", lambda p: p.update(cause_type="ValueError"), "stderr")
        self.assertFalse(self.verify()["verified"])
        self.alter_report(lambda r: r["stages"][0].update(stderr="unexpected"))
        self.assertEqual(len(self.verify()["stage_result_issues"]), 2)

    def test_malformed_payload_fails_without_crashing(self):
        for raw in ("{}", "null", '{"ok":true,"ok":true}', '{"ok":true,"result":[]}'):
            self.alter_report(lambda r, raw=raw: r["stages"][0].update(stdout=raw))
            self.assertFalse(self.verify()["verified"])

    def test_preview_transition_and_diagnosis(self):
        self.mutate("preview-current", lambda p: p["result"].update(plan_sha256="0" * 64))
        codes = [r["code"] for r in self.verify()["stage_result_issues"]]
        self.assertIn("preview_transition_mismatch", codes)
        self.mutate("diagnose-fixed", lambda p: p["result"].update(ready_for_preview=False))
        self.assertIn("diagnose-fixed", [r["stage"] for r in self.verify()["stage_result_issues"]])

    def test_cli_option(self):
        self.mutate(
            "status-applied", lambda p: p["result"].update(atomic_application_verified=True)
        )
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "embodied_agent.recalibration_evidence",
                "--directory",
                str(self.path),
                "--check-stage-results",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(child.returncode, 1, child.stderr)
        self.assertTrue(json.loads(child.stdout)["result"]["stage_results_checked"])
