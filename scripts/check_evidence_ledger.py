"""Crash-before/after-commit reconciliation with real processes and synthetic evidence."""

import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

from embodied_agent.capture_recovery import CaptureRecovery
from embodied_agent.evidence_inbox import EvidenceInbox
from embodied_agent.evidence_ledger import EvidenceLedger, reconcile_evidence
from scripts.check_evidence_inbox import sources as parent_sources
from scripts.check_process_pair_capture import fast_factory
from scripts.check_recovery_handoff import drive


def writer(connection, path, mode):
    try:
        with EvidenceLedger(path) as ledger:
            connection.send(("ready", None))
            receipt = connection.recv()
            if mode == "before_commit":

                def crash_before_commit(sql):
                    if sql.strip().upper() == "COMMIT":
                        os._exit(71)

                ledger._connection.set_trace_callback(crash_before_commit)
            result = ledger.record(receipt)
            if mode == "after_commit":
                os._exit(72)
            connection.send(("ack", result))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:  # noqa: BLE001 - report writer failure to caller
        connection.send(("error", f"{type(exc).__name__}: {exc}"[:512]))
    finally:
        connection.close()


def stop_process(process):
    process.join(0.25)
    if process.is_alive():
        process.terminate()
        process.join(0.25)
    if process.is_alive():
        process.kill()
        process.join(0.25)
    assert not process.is_alive()
    return process.exitcode


def run(output, mode, repetition):
    path = output / f"{mode}-{repetition}.sqlite"
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=writer, args=(child, str(path), mode))
    process.start()
    child.close()
    manager = CaptureRecovery(position_error_m=(0.001,) * 3, capture_profile="reserved")
    inbox = EvidenceInbox()
    acknowledgements = []

    def consume(receipt):
        parent.send(receipt)
        if not parent.poll(2):
            raise RuntimeError("writer acknowledgement timeout")
        try:
            kind, result = parent.recv()
        except EOFError as exc:
            raise RuntimeError("writer exited without acknowledgement") from exc
        if kind != "ack":
            raise RuntimeError(str(result))
        acknowledgements.append(result)

    try:
        assert parent.poll(5) and parent.recv()[0] == "ready"
        token = manager.start(fast_factory)
        assert token is not None
        drive(manager, token)
        offered = manager.queue_confirmation(token, inbox)
        dispatched = inbox.dispatch(consume)
        repeat = inbox.dispatch(consume)
        assert repeat["reason"] == "no_pending_evidence"
        code = stop_process(process)
        # SQLite may need a writable open to roll back a hot journal after a
        # crash. This repairs only our audit database, never replays the receipt.
        with EvidenceLedger(path):
            pass
        receipt_id = offered.get("receipt_id")
        reconciled = (
            reconcile_evidence(path, receipt_id) if receipt_id else {"status": "not_attempted"}
        )
        if code == 71:
            assert reconciled["status"] == "not_found"
            assert dispatched["status"] == "outcome_unknown"
        if code == 72:
            assert reconciled["status"] == "recorded"
            assert dispatched["status"] == "outcome_unknown"
        if acknowledgements:
            assert reconciled["status"] == "recorded"
        assert manager.close()
        return {
            "mode": mode,
            "repetition": repetition,
            "writer_exitcode": code,
            "offered": offered,
            "dispatched": dispatched,
            "repeat": repeat,
            "acknowledgements": acknowledgements,
            "reconciled": reconciled,
            "ledger_path": str(path),
            "ledger_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "manager": manager.result(),
        }
    finally:
        inbox.close()
        manager.close()
        parent.close()
        stop_process(process)
        process.close()


def sources():
    return parent_sources() | {
        p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in ("scripts/check_evidence_ledger.py", "src/embodied_agent/evidence_ledger.py")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = sources()
    cases = [
        run(args.output, mode, repetition)
        for repetition in range(2)
        for mode in ("normal", "before_commit", "after_commit")
    ]
    assert sources() == before
    report = args.output / "report.json"
    report.write_text(json.dumps({"sources": before, "cases": cases}, indent=2) + "\n")
    summary = {
        "kind": "sqlite_evidence_crash_reconciliation",
        "sources": before,
        "report_path": str(report),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "cases": [],
    }
    for case in cases:
        history = case["manager"]["history"]
        confirmation = history[0]["result"]["confirmation"]
        summary["cases"].append(
            {
                "mode": case["mode"],
                "repetition": case["repetition"],
                "confirmation_status": confirmation["status"],
                "confirmation_reason": confirmation["reason"],
                "offer_status": case["offered"]["status"],
                "dispatch_status": case["dispatched"]["status"],
                "writer_exitcode": case["writer_exitcode"],
                "reconciliation": case["reconciled"],
                "acknowledgements": len(case["acknowledgements"]),
                "repeat_reason": case["repeat"]["reason"],
                "ledger_path": case["ledger_path"],
                "ledger_sha256": case["ledger_sha256"],
                "capture_workers_stopped": all(h["cleanup"]["workers_stopped"] for h in history),
            }
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["cases"], indent=2))


if __name__ == "__main__":
    main()
