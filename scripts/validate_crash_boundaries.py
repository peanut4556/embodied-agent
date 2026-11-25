"""Process-death matrix using isolated mock effects, not hardware or power failures."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from embodied_agent import checkpointed_runtime, operation_receipt
from embodied_agent.action_budget import ActionBudget
from embodied_agent.budget_checkpoint import load_budget, save_budget
from embodied_agent.models import Action, Observation, Plan, PlanStep
from embodied_agent.runtime import AgentRuntime

PHASES = (
    "before_registration",
    "after_registration",
    "after_prepare",
    "after_action",
    "after_budget_commit",
    "before_receipt_commit",
    "after_receipt_commit",
)


def child(directory, fault):
    def crash(phase):
        if fault == phase:
            os._exit(23)

    class Fixture:
        def create_plan(self, *args):
            return Plan("verify", [PlanStep("verify", {"object": "b", "destination": "box"})])

        def select_action(self, step, observation):
            return Action(step.action, step.arguments)

        def observe(self):
            return Observation(objects={"b": "box"})

        def execute(self, action):
            with (directory / "effects.jsonl").open("a") as stream:
                stream.write(json.dumps({"action": action.name}) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            crash("after_action")

    write_receipt, write_budget = operation_receipt._write, checkpointed_runtime.save_budget
    receipts = budgets = 0

    def receipt_hook(path, value):
        nonlocal receipts
        receipts += 1
        crash("before_registration" if receipts == 1 else "before_receipt_commit")
        write_receipt(path, value)
        crash("after_registration" if receipts == 1 else "after_receipt_commit")

    def budget_hook(budget, path):
        nonlocal budgets
        budgets += 1
        write_budget(budget, path)
        crash("after_prepare" if budgets == 1 else "after_budget_commit")

    fixture = Fixture()
    with (
        patch.object(operation_receipt, "_write", receipt_hook),
        patch.object(checkpointed_runtime, "save_budget", budget_hook),
    ):
        result = operation_receipt.run_operation(
            AgentRuntime(fixture, fixture, fixture),
            "verify",
            directory / "budget.json",
            operation_id="op-1",
            task_timeout_s=5,
            request_context={"fixture": "crash-matrix-v1"},
        )
    print(json.dumps(result))


def effects(directory):
    path = directory / "effects.jsonl"
    return len(path.read_text().splitlines()) if path.exists() else 0


def receipt_state(directory):
    file = operation_receipt.receipt_path(directory / "budget.json", "op-1")
    return operation_receipt.read_receipt(file)["state"] if file.exists() else "absent"


def validate(output):
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src") + os.pathsep + str(root))
    report = {"passed": False, "environment": "mock-process-death", "cases": []}
    try:
        for phase in PHASES:
            directory = output / phase
            directory.mkdir()
            save_budget(ActionBudget({}, drift_policy="quarantine"), directory / "budget.json")
            args = [sys.executable, str(Path(__file__).resolve()), "--child", str(directory)]
            first = subprocess.run(
                args + ["--fault", phase],
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            before_count, before_state = effects(directory), receipt_state(directory)
            row = {
                "phase": phase,
                "crash_exit": first.returncode,
                "crash_stderr": first.stderr,
                "effects_before_retry": before_count,
                "receipt_before_retry": before_state,
                "degraded_before_retry": sorted(load_budget(directory / "budget.json").degraded),
            }
            report["cases"].append(row)
            if first.returncode != 23:
                raise AssertionError("fault did not reach expected crash boundary")
            retry = subprocess.run(
                args, env=env, capture_output=True, text=True, timeout=20, check=False
            )
            row.update(retry_exit=retry.returncode, retry_stderr=retry.stderr)
            if retry.returncode != 0:
                raise AssertionError("retry process failed")
            result = json.loads(retry.stdout)
            row.update(
                result=result,
                effects_after_retry=effects(directory),
                receipt_after_retry=receipt_state(directory),
            )
            expected_effects = 0 if phase in ("after_registration", "after_prepare") else 1
            if row["effects_after_retry"] != expected_effects:
                raise AssertionError("unexpected mock effect count")
            if phase == "before_registration":
                if not result["success"] or result["operation"]["replayed"]:
                    raise AssertionError("unregistered request failed to execute once")
            elif phase == "after_receipt_commit":
                if not result["success"] or not result["operation"]["replayed"]:
                    raise AssertionError("committed request failed to replay historical result")
            elif result["error"]["code"] != "operation_in_doubt":
                raise AssertionError("uncertain operation permitted replay")
            if phase != "before_registration" and row["effects_after_retry"] != before_count:
                raise AssertionError("retry added a duplicate effect")
        report["passed"] = True
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["python"] = sys.version
        paths = [
            Path(__file__).resolve(),
            root / "src/embodied_agent/operation_receipt.py",
            root / "src/embodied_agent/checkpointed_runtime.py",
            root / "src/embodied_agent/budget_checkpoint.py",
            root / "src/embodied_agent/runtime.py",
        ]
        report["source_sha256"] = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        }
        report["artifact_sha256"] = {
            str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.rglob("*")
            if p.is_file() and p.suffix in (".json", ".jsonl")
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--child", type=Path)
    parser.add_argument("--fault", choices=PHASES)
    args = parser.parse_args()
    if args.child:
        child(args.child, args.fault)
    else:
        result = validate(args.output)
        print(json.dumps({"passed": result["passed"], "cases": len(result["cases"])}))
