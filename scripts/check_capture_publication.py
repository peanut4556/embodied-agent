"""Real-clock capture to reviewed publication; synthetic dual-view reader only."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from embodied_agent.capture_recovery import CaptureRecovery
from embodied_agent.checkpoint_store import CheckpointStore
from embodied_agent.evidence_checkpoint import CheckpointedEvidenceLedger
from embodied_agent.evidence_inbox import EvidenceInbox
from embodied_agent.evidence_ledger import reconcile_evidence
from scripts.check_evidence_checkpoint import digest
from scripts.check_process_pair_capture import fast_factory
from scripts.check_recovery_handoff import drive
from scripts.check_recovery_handoff import sources as capture_sources


def command(module, args):
    process = subprocess.run(
        [sys.executable, "-m", module, *args],
        env=os.environ | {"PYTHONPATH": "src:."},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    return {"exitcode": process.returncode, "result": json.loads(process.stdout)}


def run(directory, mode):
    directory.mkdir()
    database, store_path = directory / "evidence.sqlite", directory / "store.sqlite"
    manager = CaptureRecovery(position_error_m=(0.001,) * 3, capture_profile="reserved")
    inbox = EvidenceInbox()
    accepted = []
    consumer_calls = 0
    commands = {}
    with CheckpointedEvidenceLedger(database) as ledger:
        with CheckpointStore(store_path, create=True) as store:
            assert store.enroll(database)["status"] == "published"

        def consume(receipt):
            nonlocal consumer_calls
            consumer_calls += 1
            if mode == "slow_consumer":
                time.sleep(0.03)
            result = ledger.record(receipt)
            if mode == "lost_ack":
                raise RuntimeError("injected callback failure after committed evidence")
            accepted.append(
                result
                | {
                    "remaining_at_acceptance_s": receipt["expires_host_time"]
                    - result["accepted_host_time"]
                }
            )

        try:
            token = manager.start(fast_factory)
            if token is None:
                offered = {"status": "startup_failed"}
                dispatched = {"status": "not_attempted"}
            else:
                drive(manager, token)
                offered = manager.queue_confirmation(token, inbox)
                if mode == "delayed":
                    time.sleep(0.03)
                dispatched = inbox.dispatch(consume)
            repeated = inbox.dispatch(consume)
            assert repeated["reason"] == "no_pending_evidence"
        finally:
            inbox.close()
            closed = manager.close()
        snapshot = manager.result()
    reconciled = (
        reconcile_evidence(database, offered["receipt_id"])
        if "receipt_id" in offered
        else {"status": "not_attempted"}
    )
    # Decide from persisted evidence, never from callback success alone. This
    # resumes historical audit publication, not the failed consumer or an action.
    if reconciled["status"] == "recorded":
        args = ["--database", str(database), "--store", str(store_path)]
        commands["before"] = command("embodied_agent.evidence_audit", args)
        commands["preview"] = command("embodied_agent.checkpoint_cli", ["preview", *args])
        proposal = directory / "proposal.json"
        proposal.write_text(json.dumps(commands["preview"]["result"], indent=2) + "\n")
        commands["publish"] = command(
            "embodied_agent.checkpoint_cli", ["publish", *args, "--proposal", str(proposal)]
        )
        commands["after"] = command("embodied_agent.evidence_audit", args)
        commands["reconcile"] = command(
            "embodied_agent.checkpoint_cli", ["reconcile", *args, "--proposal", str(proposal)]
        )
    history = snapshot["history"]
    confirmation = history[-1]["result"]["confirmation"] if history else {}
    published = {k: v["exitcode"] for k, v in commands.items()} == {
        "before": 6,
        "preview": 0,
        "publish": 0,
        "after": 0,
        "reconcile": 0,
    }
    if mode == "immediate":
        passed = (
            dispatched["status"] == "delivered"
            and len(accepted) == 1
            and consumer_calls == 1
            and published
        )
    elif mode == "lost_ack":
        passed = (
            dispatched["status"] == "outcome_unknown"
            and not accepted
            and consumer_calls == 1
            and reconciled["status"] == "recorded"
            and published
        )
    elif mode == "slow_consumer":
        passed = (
            dispatched["status"] == "outcome_unknown"
            and not accepted
            and consumer_calls == 1
            and reconciled["status"] == "not_found"
            and not commands
        )
    else:
        passed = (
            offered["status"] == "queued"
            and dispatched.get("reason") == "expired_in_queue"
            and consumer_calls == 0
            and not accepted
            and not commands
        )
    passed = (
        passed
        and closed
        and bool(history)
        and all(h["cleanup"]["workers_stopped"] for h in history)
    )
    result = {
        "mode": mode,
        "passed": passed,
        "confirmation_status": confirmation.get("status"),
        "confirmation_reason": confirmation.get("reason"),
        "offered": offered,
        "dispatched": dispatched,
        "repeat": repeated,
        "accepted": accepted,
        "consumer_calls": consumer_calls,
        "evidence_reconciliation": reconciled,
        "commands": commands,
        "workers_stopped": closed,
        "files": {str(p): digest(p) for p in (database, store_path)},
        "manager": snapshot,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/check_capture_publication.py",
        "scripts/check_evidence_checkpoint.py",
        "src/embodied_agent/evidence_inbox.py",
        "src/embodied_agent/evidence_ledger.py",
        "src/embodied_agent/indexed_evidence_ledger.py",
        "src/embodied_agent/evidence_checkpoint.py",
        "src/embodied_agent/checkpoint_store.py",
        "src/embodied_agent/checkpoint_cli.py",
        "src/embodied_agent/evidence_audit.py",
    ]
    sources = capture_sources() | {p: digest(p) for p in paths}
    cases = [
        run(out / f"{mode}-{repeat}", mode)
        for repeat in range(2)
        for mode in ("immediate", "delayed", "lost_ack", "slow_consumer")
    ]
    assert sources == capture_sources() | {p: digest(p) for p in paths}
    report = out / "report.json"
    report.write_text(json.dumps({"sources": sources, "cases": cases}, indent=2) + "\n")
    summary = {
        "sources": sources,
        "report_path": str(report),
        "report_sha256": digest(report),
        "passed": sum(c["passed"] for c in cases),
        "total": len(cases),
        "cases": [{k: v for k, v in c.items() if k != "manager"} for c in cases],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("passed", "total")}))
    for c in cases:
        print(c["mode"], c["confirmation_status"], c["dispatched"], c["passed"])
    if not all(c["passed"] for c in cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
