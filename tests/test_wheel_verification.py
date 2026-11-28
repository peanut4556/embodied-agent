import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.verify_wheel import verify


class WheelVerificationTests(unittest.TestCase):
    def test_missing_wheel_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with self.assertRaises(ValueError):
                verify(Path(directory) / "missing.whl", output)
            self.assertFalse(output.exists())

    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "test.whl"
            wheel.write_bytes(b"fixture")
            with self.assertRaises(FileExistsError):
                verify(wheel, directory)
            self.assertEqual(wheel.read_bytes(), b"fixture")

    def test_install_failure_preserves_report(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "test.whl"
            wheel.write_bytes(b"fixture")
            output = Path(directory) / "output"
            with (
                patch("scripts.verify_wheel.venv.EnvBuilder.create"),
                patch(
                    "scripts.verify_wheel.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 1, "", "invalid wheel"),
                ),
                self.assertRaises(RuntimeError),
            ):
                verify(wheel, output)
            report = json.loads((output / "wheel-verification.json").read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["steps"][0]["name"], "install")
            self.assertEqual(report["steps"][0]["stderr"], "invalid wheel")

    def managed_fixture(self, state, fault=None):
        def run(name, args, cwd, expected=0):
            if name == "create-budget":
                (state / "budget.json").write_text("{}")
                return ""
            if name == "checkpointed-task":
                return json.dumps(
                    {"success": True, "completed_steps": 4, "operation": {"replayed": False}}
                )
            if name == "duplicate-task":
                if fault == "write":
                    (state / "budget.json").write_text('{"changed":true}')
                return json.dumps({"success": True, "operation": {"replayed": fault != "replay"}})
            if name == "conflicting-task":
                self.assertEqual(expected, 3)
                return json.dumps(
                    {"error": {"code": "wrong" if fault == "conflict" else "operation_conflict"}}
                )
            if name == "inspect-budget":
                return json.dumps({"valid": True, "degraded": []})
            self.fail(name)

        return run

    def test_managed_contract_and_preserved_state(self):
        from scripts.verify_wheel import verify_managed_task

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            verify_managed_task(self.managed_fixture(state), "python", root, state)
            self.assertEqual((state / "budget.json").read_text(), "{}")

    def test_managed_regressions_rejected(self):
        from scripts.verify_wheel import verify_managed_task

        for fault in ("write", "replay", "conflict"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                state = root / "state"
                with self.assertRaises(AssertionError):
                    verify_managed_task(self.managed_fixture(state, fault), "python", root, state)

    def recovery_fixture(self, state, fault=None):
        def run(name, args, cwd, expected=0):
            result = {}
            if name == "installed-diagnosis":
                result = {"ready_for_preview": fault != "eligibility"}
            elif name == "installed-preview":
                result = {
                    "before_degraded": ["pick", "verify"],
                    "after_degraded": [],
                    "checkpoint_sha256": "0" * 64,
                    "plan_sha256": "1" * 64,
                }
                if fault == "preview-write":
                    (state / "budget.json").write_text("changed")
            elif name == "installed-recovery":
                result = {"applied": True, "after_revision": 2}
                if fault == "receipt-write":
                    (state / "budget.json.operations/receipt.json").write_text("changed")
            elif name == "installed-recovery-status":
                result = {
                    "status": "all_matched",
                    "currently_blocked_actions": [],
                    "counts": {"applied_current": 2},
                }
            elif name == "installed-reject-repeat-recovery":
                self.assertEqual(expected, 3)
            return json.dumps({"result": result})

        return run

    def test_recovery_contract_and_receipt_preservation(self):
        from scripts.verify_wheel import verify_installed_recovery

        for fault in (None, "eligibility", "preview-write", "receipt-write"):
            with tempfile.TemporaryDirectory() as directory:
                state = Path(directory)
                (state / "budget.json").write_text("{}")
                (state / "budget.json.operations").mkdir()
                (state / "budget.json.operations/receipt.json").write_text("{}")
                if fault:
                    with self.assertRaises(AssertionError):
                        verify_installed_recovery(
                            self.recovery_fixture(state, fault), "python", state, state
                        )
                else:
                    verify_installed_recovery(self.recovery_fixture(state), "python", state, state)

    def test_recovery_requires_prior_execution_receipt(self):
        from scripts.verify_wheel import verify_installed_recovery

        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            with self.assertRaisesRegex(AssertionError, "receipt missing"):
                verify_installed_recovery(self.recovery_fixture(state), "python", state, state)
