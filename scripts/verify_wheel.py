"""Install one trusted local wheel in a fresh environment and smoke-test it."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import venv
from pathlib import Path


def verify_managed_task(run, python, cwd, state_dir):
    state_dir.mkdir()
    state = state_dir / "budget.json"
    run(
        "create-budget",
        [
            python,
            "-I",
            "-c",
            """import sys
from embodied_agent.action_budget import ActionBudget
from embodied_agent.budget_checkpoint import save_budget
save_budget(ActionBudget({}, drift_policy='quarantine'), sys.argv[1])
""",
            state,
        ],
        cwd,
    )
    args = [
        python,
        "-I",
        "-m",
        "embodied_agent",
        "red block box",
        "--json",
        "--budget-checkpoint",
        state,
        "--task-timeout",
        "5",
        "--operation-id",
        "installed-task-1",
    ]
    first = json.loads(run("checkpointed-task", args, cwd))
    if (
        first.get("success") is not True
        or first.get("completed_steps") != 4
        or first.get("operation", {}).get("replayed") is not False
    ):
        raise AssertionError("installed managed task did not execute normally")
    before = {str(p.relative_to(state_dir)): p.read_bytes() for p in state_dir.rglob("*.json")}
    second = json.loads(run("duplicate-task", args, cwd))
    if second.get("success") is not True or second.get("operation", {}).get("replayed") is not True:
        raise AssertionError("installed duplicate did not replay historical result")
    changed = list(args)
    changed[4] = "different instruction"
    conflict = json.loads(run("conflicting-task", changed, cwd, expected=3))
    if conflict.get("error", {}).get("code") != "operation_conflict":
        raise AssertionError("installed conflicting request not rejected")
    inspected = json.loads(
        run("inspect-budget", [python, "-I", "-m", "embodied_agent.budget_checkpoint", state], cwd)
    )
    if inspected.get("valid") is not True or inspected.get("degraded") != []:
        raise AssertionError("installed persisted budget invalid")
    after = {str(p.relative_to(state_dir)): p.read_bytes() for p in state_dir.rglob("*.json")}
    if before != after:
        raise AssertionError("replay, rejection or inspection changed persisted JSON")


def verify_installed_recovery(run, python, cwd, state_dir):
    state = state_dir / "budget.json"
    receipts = {str(p): p.read_bytes() for p in state_dir.glob("*.operations/*.json")}
    if not receipts:
        raise AssertionError("managed execution receipt missing")
    run(
        "inject-budget-drift",
        [
            python,
            "-I",
            "-c",
            """import sys
from embodied_agent.budget_checkpoint import load_budget, save_budget
budget = load_budget(sys.argv[1])
budget.record('pick', 0.2, confirmed=True)
budget.record('verify', 0.2, confirmed=True)
save_budget(budget, sys.argv[1])
""",
            state,
        ],
        cwd,
    )
    plan = state_dir / "recovery-plan.json"
    requests = [
        {
            "schema": 1,
            "action": action,
            "calibration": {"batch_id": action + "-cal", "durations_s": [0.03] * 20},
            "validation": {"batch_id": action + "-val", "durations_s": [0.032] * 20},
        }
        for action in ("pick", "verify")
    ]
    plan.write_text(json.dumps({"schema": 1, "requests": requests}))
    base = [python, "-I", "-m", "embodied_agent.budget_recalibration_plan"]

    def command(name, operation, extra=(), expected=0):
        raw = run(
            name,
            [*base, operation, "--checkpoint", state, "--plan", plan, *extra],
            cwd,
            expected=expected,
        )
        return json.loads(raw)

    before = state.read_bytes()
    diagnosis = command("installed-diagnosis", "diagnose")["result"]
    if diagnosis.get("ready_for_preview") is not True:
        raise AssertionError("installed recovery not eligible")
    preview = command("installed-preview", "preview")["result"]
    if (
        preview.get("before_degraded") != ["pick", "verify"]
        or preview.get("after_degraded") != []
        or state.read_bytes() != before
    ):
        raise AssertionError("installed preview changed state or missed isolation")
    args = [
        "--expected-checkpoint-sha256",
        preview["checkpoint_sha256"],
        "--expected-plan-sha256",
        preview["plan_sha256"],
    ]
    applied = command("installed-recovery", "apply", args)["result"]
    if applied.get("applied") is not True or applied.get("after_revision") != 2:
        raise AssertionError("installed recovery did not commit both actions")
    saved = state.read_bytes()
    status = command("installed-recovery-status", "status")["result"]
    if (
        status.get("status") != "all_matched"
        or status.get("currently_blocked_actions") != []
        or status.get("counts", {}).get("applied_current") != 2
    ):
        raise AssertionError("installed recovered state not recognized")
    # Error JSON is on stderr; run captures it in the report, not its return value.
    run(
        "installed-reject-repeat-recovery",
        [*base, "apply", "--checkpoint", state, "--plan", plan, *args],
        cwd,
        expected=3,
    )
    if saved != state.read_bytes() or receipts != {
        str(p): p.read_bytes() for p in state_dir.glob("*.operations/*.json")
    }:
        raise AssertionError("recovery modified receipts or repeated apply changed budget")


def verify(wheel, output):
    wheel = Path(wheel).resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise ValueError("a local wheel file is required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "passed": False,
        "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "steps": [],
    }
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"

    def run(name, args, cwd, expected=0):
        child = subprocess.run(
            [str(a) for a in args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        report["steps"].append(
            {
                "name": name,
                "exit_code": child.returncode,
                "expected_exit_code": expected,
                "stdout": child.stdout,
                "stderr": child.stderr,
            }
        )
        if child.returncode != expected:
            raise RuntimeError(f"{name} failed")
        return child.stdout

    try:
        with tempfile.TemporaryDirectory(prefix="embodied-wheel-") as directory:
            root = Path(directory)
            venv.EnvBuilder(with_pip=True).create(root / "venv")
            binary = root / "venv" / ("Scripts" if os.name == "nt" else "bin")
            python = binary / "python"
            run(
                "install",
                [python, "-I", "-m", "pip", "install", "--no-index", "--no-deps", wheel],
                root,
            )
            run(
                "package-and-asset",
                [
                    python,
                    "-I",
                    "-c",
                    """import sys, pathlib
import embodied_agent
from importlib.resources import files
assert pathlib.Path(embodied_agent.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve())
asset = files('embodied_agent').joinpath('assets/tabletop.xml')
assert '<mujoco' in asset.read_text()
print('Installed package and tabletop asset verified')
""",
                ],
                root,
            )
            for entry in ("embodied-agent", "embodied-evidence-audit", "embodied-checkpoint"):
                run(entry + "-help", [binary / entry, "--help"], root)
            result = json.loads(
                run("mock-task", [binary / "embodied-agent", "red block box", "--json"], root)
            )
            if result.get("success") is not True or result.get("completed_steps") != 4:
                raise AssertionError("installed mock task did not complete four steps")
            verify_managed_task(run, python, root, output / "managed-state")
            verify_installed_recovery(run, python, root, output / "managed-state")
            report["passed"] = True
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["managed_artifact_sha256"] = {
            str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((output / "managed-state").rglob("*.json"))
        }
        (output / "wheel-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps({"passed": verify(args.wheel, args.output)["passed"]}))
