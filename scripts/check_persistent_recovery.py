"""Separate application processes: failed capture, verified cleanup, restart, exhaustion."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from embodied_agent.capture_recovery import CaptureRecovery
from embodied_agent.wait_budget_store import WaitBudgetStore
from scripts.check_evidence_checkpoint import digest
from scripts.check_process_pair_capture import fast_factory
from scripts.check_recovery_handoff import drive
from scripts.check_recovery_handoff import sources as capture_sources
from scripts.check_supervised_recovery import blocked_factory


def stage(mode, directory):
    manager = CaptureRecovery(
        position_error_m=(0.001,) * 3,
        max_generations=2,
        capture_profile="reserved",
        supervised=True,
        runtime_limit_s=4,
        spin_window_s=0.0015,
        persistent_wait_budget=WaitBudgetStore(directory / "budget.db"),
    )
    result = {"mode": mode, "pid": os.getpid()}
    try:
        if mode == "fail":
            token = manager.start(blocked_factory)
            result["startup_failed"] = token is None
            result["timeout_observed"] = any(
                d["events"] and d["events"][-1].get("runtime_limit_exceeded")
                for d in manager.backend.result()["supervision"]
            )
        elif mode == "recover":
            old = json.loads((directory / "fail.json").read_text())["manager"]["session_id"]
            token = manager.start(fast_factory)
            result["old_session_attempt"] = manager.consume_confirmation(old)
            if token:
                drive(manager, token)
                receipt = manager.consume_confirmation(token)
                result["receipt"] = {k: v for k, v in receipt.items() if k != "evidence"}
                result["repeat"] = manager.consume_confirmation(token)
                result["confirmation"] = manager.runtime.controller.result()
            else:
                result["receipt"] = {"status": "not_attempted"}
                result["repeat"] = {}
                result["confirmation"] = {"status": "startup_failed", "requests": []}
        else:
            try:
                manager.start(fast_factory)
            except RuntimeError as exc:
                result["refusal"] = str(exc)
            else:
                result["refusal"] = None
            result["backend_created"] = manager.backend is not None
        result["closed"] = manager.close()
        result["manager"] = manager.result()
        result["pool"] = WaitBudgetStore(directory / "budget.db").snapshot()
        (directory / f"{mode}.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        manager.close()
    if mode == "fail":
        # Cleanup has been explicitly verified before ending the application.
        # This does not demonstrate orphan-device ownership recovery.
        os._exit(74)


def first_stage_verified(first, exitcode):
    history = first["manager"]["history"]
    return (
        exitcode == 74
        and first["startup_failed"]
        and first["timeout_observed"]
        and first["closed"]
        and len(history) == 1
        and history[0]["cleanup"]["workers_stopped"]
        and first["pool"]["reserved_slots"] == 1
    )


def summarize(first, recovered, refused, exitcodes):
    history = recovered["manager"]["history"]
    cleanup = history[-1]["cleanup"] if history else {}
    waits = [r["reservation_wait"] for r in cleanup.get("timing_trace", [])]
    confirmation = recovered["confirmation"]
    previous = 0
    trace_ok = bool(waits)
    for wait in waits:
        trace_ok = trace_ok and wait["wait_cpu_budget_s"] == 0.005
        trace_ok = trace_ok and wait["effective_spin_window_s"] == (
            0 if previous >= 0.005 else 0.0015
        )
        trace_ok = trace_ok and abs(wait["wait_cpu_spent_s"] - previous - wait["wait_cpu_s"]) < 1e-9
        previous = wait["wait_cpu_spent_s"]
    a = first["manager"]["wait_cpu_budget"]["last_persistent_reservation"]
    b = recovered["manager"]["wait_cpu_budget"]["last_persistent_reservation"]
    integrity = (
        first_stage_verified(first, exitcodes[0])
        and exitcodes[1:] == [0, 0]
        and a["pool_id"] == b["pool_id"] == refused["pool"]["pool_id"]
        and a["slot"] == 1
        and b["slot"] == 2
        and a["allowance_ns"] == b["allowance_ns"] == 5_000_000
        and a["request_id"] != b["request_id"]
        and recovered["closed"]
        and cleanup.get("workers_stopped")
        and recovered["old_session_attempt"].get("reason") == "session_mismatch"
        and refused["refusal"] == "persistent wait budget exhausted"
        and not refused["backend_created"]
        and refused["pool"]["reserved_slots"] == 2
        and refused["pool"]["remaining_ns"] == 0
        and len(waits) == len(confirmation["requests"])
        and trace_ok
    )
    full = (
        integrity
        and confirmation["status"] == "confirmed_recent_evidence"
        and len(confirmation["requests"]) == 36
        and recovered["receipt"]["status"] == "consumed"
        and recovered["repeat"].get("reason") == "already_consumed"
    )
    return {
        "integrity_verified": bool(integrity),
        "full_recovery_passed": bool(full),
        "exitcodes": exitcodes,
        "confirmation_status": confirmation["status"],
        "requests": len(confirmation["requests"]),
        "fallback_requests": [w["request_id"] for w in waits if w["effective_spin_window_s"] == 0],
        "wait_cpu_s": previous,
        "remaining_ns": refused["pool"]["remaining_ns"],
        "receipt": recovered["receipt"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["fail", "recover", "refuse"])
    args = parser.parse_args()
    if args.stage:
        stage(args.stage, args.output)
        return
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/check_persistent_recovery.py",
        "scripts/check_supervised_recovery.py",
        "src/embodied_agent/wait_budget_store.py",
        "src/embodied_agent/scheduled_pair_capture.py",
        "src/embodied_agent/capture_supervisor.py",
    ]
    sources = capture_sources() | {p: digest(p) for p in paths}
    (out / "plan.json").write_text(
        json.dumps(
            {"trials": 2, "stages": ["fail", "recover", "refuse"], "sources": sources}, indent=2
        )
        + "\n"
    )
    summaries = []
    for index in range(2):
        directory = out / str(index + 1)
        directory.mkdir()
        WaitBudgetStore.create(directory / "budget.db", total_threshold_ns=10_000_000, slots=2)
        results, codes = [], []
        for mode in ("fail", "recover", "refuse"):
            with (directory / f"{mode}.log").open("w") as log:
                process = subprocess.run(
                    [sys.executable, __file__, "--output", str(directory), "--stage", mode],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=20,
                    check=False,
                )
            codes.append(process.returncode)
            results.append(json.loads((directory / f"{mode}.json").read_text()))
            if mode == "fail" and not first_stage_verified(results[0], codes[0]):
                raise RuntimeError("old cleanup unverified; refusing application restart")
        summaries.append(summarize(*results, codes))
    if sources != capture_sources() | {p: digest(p) for p in paths}:
        raise RuntimeError("sources changed")
    files = {str(p): digest(p) for p in out.glob("*/*.json")}
    result = {"sources": sources, "stage_report_hashes": files, "cases": summaries}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
    if not all(c["integrity_verified"] for c in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
