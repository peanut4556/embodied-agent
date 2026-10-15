"""Raw-data recomputation, exact pairing and separate import/evaluation outcomes."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from embodied_agent.external_joint_report import build_joint_report

AS_OF = "2026-10-14T10:00:01+08:00"
ROOT = Path("examples/external-observations")


class ExternalJointTests(unittest.TestCase):
    def setUp(self):
        self.observations = json.loads(
            Path("examples/external-joint/observations.json").read_text()
        )
        self.policy = json.loads((ROOT / "policy.json").read_text())
        self.visual = json.loads(Path("examples/external-joint/visual.json").read_text())

    def evaluate(self):
        return build_joint_report(
            self.observations, self.policy, ROOT / "evidence", self.visual, AS_OF
        )

    def assert_pair_rejected(self):
        r = self.evaluate()
        self.assertEqual(r["import"]["status"], "validated")
        self.assertEqual(r["pairing_status"], "rejected")
        self.assertEqual(r["results"], [])
        self.assertTrue(r["errors"])

    def test_valid_import_is_not_precision_acceptance(self):
        r = self.evaluate()
        self.assertEqual(r["status"], "evaluated")
        self.assertEqual(r["import"]["status"], "validated")
        self.assertEqual(r["summary"], {"passed": 1, "rejected": 1, "unknown": 0})
        good, bad = r["results"]
        self.assertTrue(good["passes_offline_checks"])
        self.assertTrue(bad["residual_check"]["passes"])
        self.assertTrue(bad["precision_check"]["passes"])
        self.assertFalse(bad["external_check"]["passes"])
        self.assertFalse(r["calibration_certified"])
        self.assertFalse(r["correction_applied"])

    def test_reordered_external_records_pair_by_id(self):
        first = self.evaluate()["results"]
        self.observations["observations"].reverse()
        self.assertEqual(first, self.evaluate()["results"])

    def test_missing_extra_and_reused_ids_are_atomic_rejections(self):
        original = copy.deepcopy(self.visual)
        self.visual["fits"].pop()
        self.assert_pair_rejected()
        self.visual = copy.deepcopy(original)
        self.visual["fits"][1]["observation_id"] = "missing"
        self.assert_pair_rejected()
        self.visual["fits"][1]["observation_id"] = self.visual["fits"][0]["observation_id"]
        self.assert_pair_rejected()
        self.visual = copy.deepcopy(original)
        self.visual["fits"][1]["fit_id"] = self.visual["fits"][0]["fit_id"]
        self.assert_pair_rejected()
        self.visual = copy.deepcopy(original)
        self.visual["fits"].append(copy.deepcopy(self.visual["fits"][0]))
        self.assert_pair_rejected()

    def test_metadata_and_nominal_origin_mismatch(self):
        for key in (
            "camera_id",
            "data_kind",
            "frame_id",
            "clock_id",
            "reference_origin_id",
            "position_semantics",
        ):
            old = self.visual["metadata"][key]
            self.visual["metadata"][key] = "wrong"
            self.assert_pair_rejected()
            self.visual["metadata"][key] = old
        self.visual["metadata"]["nominal_camera_origin_m"] = [0, 0, 0]
        self.assert_pair_rejected()
        self.policy["nominal_camera_origin_m"] = [0, 0, 0]
        self.assert_pair_rejected()  # Matching metadata still cannot change the solver's fixed origin.

    def test_camera_time_requires_exact_instant_but_allows_timezone_equivalence(self):
        self.visual["fits"][0]["camera_time"] = "2026-10-14T02:00:00.020Z"
        self.assertEqual(self.evaluate()["status"], "evaluated")
        self.visual["fits"][0]["camera_time"] = "2026-10-14T02:00:00.021Z"
        self.assert_pair_rejected()

    def test_calibration_cannot_overlap_validation(self):
        self.visual["calibration"]["end_time"] = self.visual["fits"][0]["camera_time"]
        self.assert_pair_rejected()
        self.visual["calibration"]["start_time"] = "2026-10-13T09:00:00+08:00"
        self.visual["calibration"]["end_time"] = "2026-10-13T09:59:50+08:00"
        self.assert_pair_rejected()

    def test_raw_visual_fields_shapes_and_false_pass_flags(self):
        original = copy.deepcopy(self.visual)
        self.visual["fits"][0]["passes"] = True
        self.assert_pair_rejected()
        self.visual = copy.deepcopy(original)
        self.visual["fits"][0]["depths_m"][0][0] = True
        self.assert_pair_rejected()
        self.visual = copy.deepcopy(original)
        self.visual["rays"][0][2] = 1
        self.assert_pair_rejected()
        self.visual = copy.deepcopy(original)
        self.visual["calibration"]["depths_m"][0][0][0] = float("nan")
        self.assert_pair_rejected()

    def test_fourth_plane_decision_is_recomputed_from_raw_depth(self):
        self.visual["fits"][0]["depths_m"][3] = [
            v + 0.01 for v in self.visual["fits"][0]["depths_m"][3]
        ]
        result = self.evaluate()["results"][0]
        self.assertEqual(result["status"], "rejected")
        self.assertFalse(result["residual_check"]["passes"])

    def test_unidentifiable_visual_fit_remains_unknown(self):
        self.visual["planes"][:3] = [copy.deepcopy(self.visual["planes"][0]) for _ in range(3)]
        r = self.evaluate()
        self.assertEqual(r["status"], "evaluated")
        self.assertEqual(r["summary"]["unknown"], 2)
        self.assertTrue(all(not x["passes_offline_checks"] for x in r["results"]))

    def test_external_policy_and_evidence_rechecked(self):
        self.policy["evidence"]["noise"]["sha256"] = "0" * 64
        r = self.evaluate()
        self.assertEqual(r["status"], "rejected")
        self.assertEqual(r["pairing_status"], "not_evaluated")
        self.assertEqual(r["results"], [])

    def test_cli_counts_and_source_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "joint.json"
            cmd = [
                sys.executable,
                "scripts/evaluate_external_joint.py",
                "--observations",
                "examples/external-joint/observations.json",
                "--visual",
                "examples/external-joint/visual.json",
                "--policy",
                str(ROOT / "policy.json"),
                "--evidence-root",
                str(ROOT / "evidence"),
                "--as-of",
                AS_OF,
                "--report",
                str(report),
            ]
            env = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            saved = json.loads(report.read_text())
            self.assertEqual(saved["summary"]["rejected"], 1)
            self.assertEqual(set(saved["source_sha256"]), {"observations", "policy", "visual"})
            cmd[cmd.index("--as-of") + 1] = "2026-10-15T10:00:01+08:00"
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(json.loads(report.read_text())["results"], [])


if __name__ == "__main__":
    unittest.main()
