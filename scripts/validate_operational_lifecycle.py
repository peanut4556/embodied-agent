"""Real CLI subprocess acceptance using isolated mock state, never hardware."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from embodied_agent.action_budget import ActionBudget
from embodied_agent.budget_checkpoint import load_budget, save_budget


def validate(output):
    output = Path(output).resolve()
    output.mkdir(parents=False, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src") + os.pathsep + str(root))
    state = output / "budget.json"
    save_budget(ActionBudget({}, drift_policy="quarantine"), state)
    records = []
    report = {"passed": False, "environment": "mock-only", "python": sys.version, "stages": records}

    def command(name, args, expected=0, decode=True):
        child = subprocess.run(
            [sys.executable, *map(str, args)],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        row = {
            "name": name,
            "exit_code": child.returncode,
            "expected_exit_code": expected,
            "stdout": child.stdout,
            "stderr": child.stderr,
        }
        records.append(row)
        if child.returncode != expected:
            raise AssertionError(f"{name}: unexpected exit code {child.returncode}")
        return json.loads(child.stdout) if decode else None

    def admin(name, *args, expected=0):
        return command(name, ["-m", "embodied_agent.operation_admin", *args], expected)["result"]

    def snapshot(name, sequence):
        path = output / (name + ".json")
        admin(
            name,
            "snapshot",
            "--checkpoint",
            state,
            "--output",
            path,
            "--source-id",
            "isolated-acceptance",
            "--capture-sequence",
            sequence,
        )
        return path

    try:
        initial = snapshot("initial", 1)
        args = [
            "-m",
            "embodied_agent",
            "red block box",
            "--budget-checkpoint",
            state,
            "--task-timeout",
            "5",
            "--operation-id",
            "normal-1",
            "--json",
        ]
        first = command("normal-task", args)
        if not first["success"] or first["completed_steps"] != 4 or first["operation"]["replayed"]:
            raise AssertionError("normal task did not execute four confirmed steps")
        saved = state.read_bytes()
        second = command("duplicate-request", args)
        if not second["operation"]["replayed"] or state.read_bytes() != saved:
            raise AssertionError("duplicate request did not preserve checkpoint")
        changed = list(args)
        changed[2] = "different instruction"
        conflict = command("request-conflict", changed, 3)
        if conflict["error"]["code"] != "operation_conflict":
            raise AssertionError("request conflict not detected")
        code = """import os, sys
from embodied_agent.operation_receipt import run_operation
class Runtime:
    def run(self,*args,**kwargs): os._exit(23)
run_operation(Runtime(),'interrupted',sys.argv[1],operation_id='crash-1',task_timeout_s=1,request_context={})
"""
        command("abrupt-process-exit", ["-c", code, state], 23, False)
        if not all(load_budget(state).blocked(a) for a in ("locate", "pick", "place", "verify")):
            raise AssertionError("crash did not preserve quarantine")
        retry = """import json, sys
from embodied_agent.operation_receipt import run_operation
class Runtime:
    def run(self,*args,**kwargs): raise AssertionError('must not execute')
print(json.dumps(run_operation(Runtime(),'interrupted',sys.argv[1],operation_id='crash-1',task_timeout_s=1,request_context={})))
"""
        pending = command("pending-retry", ["-c", retry, state])
        if pending["error"]["code"] != "operation_in_doubt":
            raise AssertionError("pending request retried")
        review = admin(
            "inspect-pending", "inspect", "--checkpoint", state, "--operation-id", "crash-1"
        )
        interrupted = snapshot("interrupted", 2)
        before_close = state.read_bytes()
        admin(
            "abandon-pending",
            "abandon",
            "--checkpoint",
            state,
            "--operation-id",
            "crash-1",
            "--expected-receipt-sha256",
            review["receipt_sha256"],
            "--reason",
            "acceptance test cancellation",
            "--evidence-ref",
            "mock-crash-fixture",
            "--actor",
            "test-runner",
        )
        if state.read_bytes() != before_close:
            raise AssertionError("abandon changed quarantine")
        abandoned = command("abandoned-retry", ["-c", retry, state])
        if abandoned["error"]["code"] != "operation_abandoned":
            raise AssertionError("abandoned operation replayed")
        final = snapshot("final", 3)
        admin("offline-verification", "verify-snapshot", "--snapshot", final)
        diff = admin(
            "pending-to-abandoned-diff",
            "compare-snapshots",
            "--before",
            interrupted,
            "--after",
            final,
        )
        if (
            not diff["comparison_performed"]
            or diff["needs_review"]
            or diff["counts"]["changed"] != 1
        ):
            raise AssertionError("valid abandonment transition not recognized")
        overall = admin(
            "whole-lifecycle-diff", "compare-snapshots", "--before", initial, "--after", final
        )
        if overall["counts"]["added"] != 2:
            raise AssertionError("operation inventory incomplete")
        inventory = admin("final-inventory", "list", "--checkpoint", state)
        if inventory["page_counts"] != {"pending": 0, "complete": 1, "abandoned": 1, "invalid": 0}:
            raise AssertionError("unexpected final inventory")
        report["passed"] = True
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        paths = [Path(__file__).resolve(), *sorted((root / "src/embodied_agent").glob("*.py"))]
        report["source_sha256"] = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        }
        report["artifact_sha256"] = {
            str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.rglob("*.json")
        }
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.output)
    print(json.dumps({"passed": result["passed"], "stages": len(result["stages"])}))
