"""Fail-closed offline data import; no simulation budgets silently fill missing evidence."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from embodied_agent.external_observation_contract import (
    ContractError,
    import_observations,
    strict_json,
)

ROOT = Path("examples/external-observations")
AS_OF = "2026-10-14T10:00:01+08:00"


class ObservationContractTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((ROOT / "policy.json").read_text())
        self.document = json.loads((ROOT / "observations.json").read_text())

    def run_import(self):
        return import_observations(self.document, self.policy, ROOT / "evidence", AS_OF)

    def assert_rejected(self, report):
        self.assertEqual(report["status"], "rejected")
        self.assertFalse(report["eligible_for_offline_check"])
        self.assertEqual(report["normalized"], [])
        self.assertTrue(report["errors"])

    def test_valid_units_and_complete_budget(self):
        r = self.run_import()
        self.assertEqual(r["status"], "validated")
        n = r["normalized"][0]
        self.assertEqual(n["translation_m"], [0.001, -0.002, 0.003])
        self.assertAlmostEqual(n["effective_skew_s"], 0.025)
        self.assertAlmostEqual(n["systematic_error_bound_m"], 0.000710025)
        self.assertFalse(r["calibration_certified"])
        self.assertFalse(r["correction_applied"])
        self.assertFalse(r["evidence_contents_authenticated"])
        self.document["observations"][0].update(units="m", position=[0.001, -0.002, 0.003])
        self.assertEqual(self.run_import()["normalized"][0]["translation_m"], n["translation_m"])

    def test_policy_missing_unknown_and_unsupported_version(self):
        for key in ("sigma_m", "evidence", "clock_error_bound_s"):
            with self.subTest(key=key):
                p = copy.deepcopy(self.policy)
                del p[key]
                self.assert_rejected(
                    import_observations(self.document, p, ROOT / "evidence", AS_OF)
                )
        self.policy["default_sigma"] = 0.0005
        self.assert_rejected(self.run_import())
        del self.policy["default_sigma"]
        for v in (True, 2, "1"):
            self.policy["schema_version"] = v
            self.assert_rejected(self.run_import())

    def test_identity_and_measurement_point_mismatch(self):
        for key in (
            "data_kind",
            "sensor_id",
            "camera_id",
            "frame_id",
            "clock_id",
            "mount_id",
            "measurement_point",
        ):
            with self.subTest(key=key):
                old = self.document["observations"][0][key]
                self.document["observations"][0][key] = "wrong"
                self.assert_rejected(self.run_import())
                self.document["observations"][0][key] = old

    def test_positions_and_numeric_budget_reject_nonfinite_bool_and_overflow(self):
        for position in ([1, 2], [1, 2, True], [0, 0, float("nan")], [0, 0, 10**1000]):
            self.document["observations"][0]["position"] = position
            self.assert_rejected(self.run_import())
        self.document["observations"][0]["position"] = [1, 2, 3]
        for value in (True, -1, float("inf"), 10**1000, "0.5"):
            self.policy["sigma_m"] = value
            self.assert_rejected(self.run_import())

    def test_strict_json_rejects_duplicate_and_nonfinite(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', "{bad"):
            with self.assertRaises(ContractError):
                strict_json(raw)

    def test_timestamp_expiry_future_stale_and_timezone(self):
        for value in (
            "2026-10-14T10:00:00",
            "2026-10-14T10:00:02+08:00",
            "2026-10-14T09:59:58+08:00",
            "invalid",
        ):
            self.document["observations"][0]["measurement_time"] = value
            self.assert_rejected(self.run_import())
        self.document["observations"][0]["measurement_time"] = "2026-10-14T10:00:00+08:00"
        self.policy["valid_until"] = "2026-10-14T10:00:00+08:00"
        self.assert_rejected(self.run_import())

    def test_sync_includes_clock_error_and_drift_reference(self):
        self.policy["max_skew_s"] = 0.024
        self.assert_rejected(self.run_import())
        self.policy["max_skew_s"] = 0.025
        self.assertEqual(self.run_import()["status"], "validated")
        self.policy["reference_time"] = "2026-10-14T10:00:00.010+08:00"
        self.assert_rejected(self.run_import())

    def test_evidence_missing_tampered_and_escape(self):
        for path in ("missing.txt", "../policy.json", "/etc/passwd"):
            self.policy["evidence"]["noise"]["path"] = path
            self.assert_rejected(self.run_import())
        self.policy["evidence"]["noise"]["path"] = "synthetic-contract.txt"
        self.policy["evidence"]["noise"]["sha256"] = "0" * 64
        self.assert_rejected(self.run_import())
        del self.policy["evidence"]["noise"]
        self.assert_rejected(self.run_import())

    def test_symlink_outside_bundle_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "evidence"
            root.mkdir()
            outside = Path(tmp) / "secret"
            outside.write_text("outside")
            (root / "link").symlink_to(outside)
            self.policy["evidence"]["noise"]["path"] = "link"
            self.assert_rejected(import_observations(self.document, self.policy, root, AS_OF))

    def test_batch_is_atomic_duplicate_or_invalid_second_record(self):
        self.document["observations"].append(copy.deepcopy(self.document["observations"][0]))
        self.assert_rejected(self.run_import())
        self.document["observations"][1]["observation_id"] = "other"
        self.document["observations"][1]["units"] = "cm"
        self.assert_rejected(self.run_import())
        self.document["observations"] = []
        self.assert_rejected(self.run_import())

    def test_cli_success_rejection_exit_code_and_input_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            cmd = [
                sys.executable,
                "scripts/import_external_observations.py",
                "--input",
                str(ROOT / "observations.json"),
                "--policy",
                str(ROOT / "policy.json"),
                "--evidence-root",
                str(ROOT / "evidence"),
                "--as-of",
                AS_OF,
                "--report",
                str(report),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = json.loads(report.read_text())
            self.assertEqual(set(output["source_sha256"]), {"input", "policy"})
            cmd[cmd.index("--as-of") + 1] = "2026-10-15T10:00:00+08:00"
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assert_rejected(json.loads(report.read_text()))

    def test_origin_semantics_and_timestamp_uncertainty(self):
        self.document["observations"][0]["position_semantics"] = "absolute_world_position"
        self.assert_rejected(self.run_import())
        self.document["observations"][0]["position_semantics"] = self.policy["position_semantics"]
        self.document["observations"][0]["reference_origin_id"] = "other_origin"
        self.assert_rejected(self.run_import())
        self.document["observations"][0]["reference_origin_id"] = self.policy["reference_origin_id"]
        self.policy["max_age_s"] = 1.002
        self.assert_rejected(self.run_import())
        self.policy["max_age_s"] = 2.0
        self.policy["nominal_camera_origin_m"] = [0, 0]
        self.assert_rejected(self.run_import())

    def test_cli_refuses_input_overwrite(self):
        cmd = [
            sys.executable,
            "scripts/import_external_observations.py",
            "--input",
            str(ROOT / "observations.json"),
            "--policy",
            str(ROOT / "policy.json"),
            "--evidence-root",
            str(ROOT / "evidence"),
            "--as-of",
            AS_OF,
            "--report",
            str(ROOT / "observations.json"),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("must not overwrite", result.stderr)


if __name__ == "__main__":
    unittest.main()
