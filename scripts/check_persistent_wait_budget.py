"""Real process contention and post-commit caller loss for a threshold pool."""

import argparse
import json
import multiprocessing
import os
from pathlib import Path

from embodied_agent.wait_budget_store import WaitBudgetStore
from scripts.check_evidence_checkpoint import digest


def contender(path, identity, start, reports):
    start.wait(5)
    try:
        receipt = WaitBudgetStore(path).reserve(identity)
    except RuntimeError as exc:
        reports.put({"identity": identity, "status": "refused", "reason": str(exc)})
    else:
        reports.put({"identity": identity, "status": "reserved", "receipt": receipt})


def lose_receipt(path):
    WaitBudgetStore(path).reserve("lost-receipt")
    os._exit(73)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/check_persistent_wait_budget.py",
        "src/embodied_agent/wait_budget_store.py",
        "src/embodied_agent/capture_recovery.py",
    ]
    sources = {p: digest(p) for p in paths}
    ctx = multiprocessing.get_context("spawn")
    store = WaitBudgetStore.create(out / "race.db", total_threshold_ns=10_000_000, slots=2)
    start, reports = ctx.Event(), ctx.Queue()
    processes = [
        ctx.Process(target=contender, args=(store.path, str(i), start, reports)) for i in range(4)
    ]
    exitcodes = []
    try:
        for p in processes:
            p.start()
        start.set()
        results = [reports.get(timeout=10) for _ in processes]
        for p in processes:
            p.join(5)
        exitcodes = [p.exitcode for p in processes]
    finally:
        for p in processes:
            if p.pid is not None:
                if p.is_alive():
                    p.kill()
                p.join(5)
                p.close()
        reports.close()
        reports.join_thread()
    loss = WaitBudgetStore.create(out / "loss.db", total_threshold_ns=5_000_000, slots=1)
    child = ctx.Process(target=lose_receipt, args=(loss.path,))
    try:
        child.start()
        child.join(5)
        loss_exit = child.exitcode
    finally:
        if child.is_alive():
            child.kill()
        child.join(5)
        child.close()
    reopened = WaitBudgetStore(loss.path)
    try:
        reopened.reserve("new-manager")
    except RuntimeError as exc:
        refusal = str(exc)
    else:
        refusal = None
    passed = (
        exitcodes == [0] * 4
        and sum(r["status"] == "reserved" for r in results) == 2
        and sum(r.get("reason") == "persistent wait budget exhausted" for r in results) == 2
        and store.snapshot()["remaining_ns"] == 0
        and loss_exit == 73
        and reopened.snapshot()["reserved_slots"] == 1
        and refusal == "persistent wait budget exhausted"
    )
    if sources != {p: digest(p) for p in paths}:
        raise RuntimeError("sources changed")
    summary = {
        "sources": sources,
        "passed": passed,
        "race_results": results,
        "race_exitcodes": exitcodes,
        "race_snapshot": store.snapshot(),
        "post_commit_exitcode": loss_exit,
        "loss_snapshot": reopened.snapshot(),
        "new_reservation_refusal": refusal,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
