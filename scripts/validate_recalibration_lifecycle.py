"""Reproduce the recalibration maintenance workflow in isolated synthetic state."""

import argparse
import copy
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
    state, plan = output / "budget.json", output / "plan.json"
    stages = []
    report = {
        "passed": False,
        "environment": "synthetic-only",
        "python": sys.version,
        "stages": stages,
    }

    def snapshot(name):
        (output / (name + ".json")).write_bytes(state.read_bytes())

    def command(name, operation, expected=0, preview=None):
        args = [
            sys.executable,
            "-m",
            "embodied_agent.budget_recalibration_plan",
            operation,
            "--checkpoint",
            str(state),
            "--plan",
            str(plan),
        ]
        if preview:
            args.extend(
                [
                    "--expected-checkpoint-sha256",
                    preview["checkpoint_sha256"],
                    "--expected-plan-sha256",
                    preview["plan_sha256"],
                ]
            )
        child = subprocess.run(
            args, cwd=root, env=env, capture_output=True, text=True, timeout=20, check=False
        )
        stages.append(
            {
                "name": name,
                "exit_code": child.returncode,
                "expected_exit_code": expected,
                "stdout": child.stdout,
                "stderr": child.stderr,
            }
        )
        if child.returncode != expected:
            raise AssertionError(f"{name}: unexpected exit code {child.returncode}")
        payload = json.loads(child.stderr if expected == 3 else child.stdout)
        return payload.get("result", payload)

    def require(condition, message):
        if not condition:
            raise AssertionError(message)

    try:
        budget = ActionBudget(
            {"pick": [0.01] * 20, "verify": [0.01] * 20}, drift_policy="quarantine"
        )
        budget.degraded.update(("pick", "verify", "place"))
        save_budget(budget, state)
        snapshot("initial")
        requests = [
            {
                "schema": 1,
                "action": action,
                "calibration": {"batch_id": action + "-cal", "durations_s": [0.03] * 20},
                "validation": {"batch_id": action + "-val", "durations_s": [0.032] * 20},
            }
            for action in ("pick", "verify")
        ]
        good = {"schema": 1, "requests": requests}
        bad = copy.deepcopy(good)
        bad["requests"][0]["calibration"]["durations_s"] = [0.03] * 2
        bad["requests"][1]["validation"]["durations_s"] = [0.09] * 20
        plan.write_text(json.dumps(bad))
        (output / "invalid-plan.json").write_bytes(plan.read_bytes())
        initial = state.read_bytes()
        diagnosis = command("diagnose-invalid", "diagnose", 1)
        require(
            diagnosis["eligible_count"] == 0 and len(diagnosis["results"]) == 2,
            "diagnosis did not report both failures",
        )
        command("preview-invalid", "preview", 3)
        require(state.read_bytes() == initial, "invalid plan changed checkpoint")
        plan.write_text(json.dumps(good))
        require(command("diagnose-fixed", "diagnose")["ready_for_preview"], "fixed plan not ready")
        preview = command("preview-fixed", "preview")
        # Simulate a cooperating runtime update between inspection and application.
        with_state = load_budget(state)
        with_state.degraded.add("locate")
        save_budget(with_state, state)
        snapshot("concurrent-change")
        concurrent = state.read_bytes()
        error = command("reject-stale-preview", "apply", 3, preview)
        require(error["cause_type"] == "StaleRecalibration", "stale preview accepted")
        require(state.read_bytes() == concurrent, "stale apply changed checkpoint")
        preview = command("preview-current", "preview")
        applied = command("apply-current", "apply", preview=preview)
        require(applied["applied"] and applied["after_revision"] == 2, "plan not applied")
        snapshot("applied")
        current = state.read_bytes()
        status = command("status-applied", "status")
        require(
            status["status"] == "all_matched" and status["counts"]["applied_current"] == 2,
            "applied history not recognized",
        )
        require(load_budget(state).degraded == {"locate", "place"}, "unrelated quarantine cleared")
        command("reject-repeat-apply", "apply", 3, preview)
        require(state.read_bytes() == current, "query or repeat apply changed checkpoint")
        budget = load_budget(state)
        budget.record("pick", 0.1, confirmed=True)
        save_budget(budget, state)
        snapshot("drifted")
        drifted = state.read_bytes()
        status = command("status-after-drift", "status")
        require(
            status["status"] == "all_matched"
            and status["counts"]["applied_then_degraded"] == 1
            and status["currently_blocked_actions"] == ["pick"],
            "new drift was hidden",
        )
        diagnosis = command("diagnose-reused-batches", "diagnose", 1)
        require(
            all("historical_batch_conflict" in row["reasons"] for row in diagnosis["results"]),
            "historical batch reuse not reported",
        )
        require(state.read_bytes() == drifted, "post-drift inspection changed state")
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
            for p in sorted(output.glob("*.json"))
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.output)
    print(json.dumps({"passed": result["passed"], "stages": len(result["stages"])}))
