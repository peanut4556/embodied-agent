"""Lose the capture owner's memory; discover committed evidence from paths alone."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from embodied_agent.capture_recovery import CaptureRecovery
from embodied_agent.checkpoint_store import (
    CheckpointStore,
    discover_from_store,
    preview_publication,
)
from embodied_agent.evidence_checkpoint import CheckpointedEvidenceLedger
from embodied_agent.evidence_inbox import EvidenceInbox
from scripts.check_evidence_checkpoint import digest
from scripts.check_process_pair_capture import fast_factory
from scripts.check_recovery_handoff import drive
from scripts.check_recovery_handoff import sources as capture_sources


def owner(directory, mode):
    manager = CaptureRecovery(position_error_m=(0.001,) * 3, capture_profile="reserved")
    inbox = EvidenceInbox()
    marker = directory / "owner-status.json"

    def exit_at_boundary(code):
        # Reap capture resources explicitly before killing the owner. This tests
        # database transaction failure, not arbitrary orphan-worker cleanup.
        closed = manager.close()
        marker.write_text(
            json.dumps(
                {"capture_workers_reaped": closed, "confirmation": "confirmed_recent_evidence"}
            )
        )
        os._exit(code if closed else 99)

    try:
        with CheckpointedEvidenceLedger(directory / "evidence.sqlite") as ledger:
            token = manager.start(fast_factory)
            if token is None:
                raise RuntimeError("capture startup failed")
            drive(manager, token)

            def consume(receipt):
                commits = 0

                def trace(sql):
                    nonlocal commits
                    if sql.strip().upper() == "COMMIT":
                        commits += 1
                        # First COMMIT persists the index; second commits evidence.
                        if mode == "before_commit" and commits == 2:
                            exit_at_boundary(91)

                ledger._connection.set_trace_callback(trace)
                ledger.record(receipt)
                exit_at_boundary(92)

            offered = manager.queue_confirmation(token, inbox)
            dispatched = inbox.dispatch(consume)
            # Any normal return means the intended fault boundary wasn't reached.
            marker.write_text(
                json.dumps(
                    {
                        "offered": offered["status"],
                        "dispatched": dispatched,
                        "confirmation": manager.runtime.controller.status,
                        "reason": manager.runtime.controller.reason,
                    }
                )
            )
    finally:
        inbox.close()
        manager.close()
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=("before_commit", "after_commit"))
    args = parser.parse_args()
    if args.worker:
        return owner(args.output, args.worker)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/check_capture_owner_exit.py",
        "scripts/check_evidence_checkpoint.py",
        "src/embodied_agent/checkpoint_store.py",
        "src/embodied_agent/evidence_checkpoint.py",
        "src/embodied_agent/indexed_evidence_ledger.py",
        "src/embodied_agent/evidence_ledger.py",
        "src/embodied_agent/evidence_inbox.py",
    ]
    sources = capture_sources() | {p: digest(p) for p in paths}
    cases = []
    for repeat in range(2):
        for mode, expected_code in [("before_commit", 91), ("after_commit", 92)]:
            directory = out / f"{mode}-{repeat}"
            directory.mkdir()
            database, store_path = directory / "evidence.sqlite", directory / "store.sqlite"
            with CheckpointedEvidenceLedger(database):
                pass
            with CheckpointStore(store_path, create=True) as store:
                store.enroll(database)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "scripts.check_capture_owner_exit",
                    "--output",
                    str(directory),
                    "--worker",
                    mode,
                ],
                env=os.environ | {"PYTHONPATH": "src:."},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=12)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate(timeout=3)
            # Roll back hot SQLite journal; never replay a receipt or re-enroll.
            with CheckpointedEvidenceLedger(database):
                pass
            discovered = discover_from_store(database, store_path)
            preview = preview_publication(database, store_path)
            marker_path = directory / "owner-status.json"
            marker = json.loads(marker_path.read_text()) if marker_path.exists() else {}
            expected_status = "not_found" if mode == "before_commit" else "recorded"
            expected_preview = "source_rejected" if mode == "before_commit" else "proposed"
            passed = (
                process.returncode == expected_code
                and discovered["status"] == "scanned"
                and len(discovered["entries"]) == 1
                and discovered["entries"][0]["reconciliation"]["status"] == expected_status
                and preview["status"] == expected_preview
                and marker.get("capture_workers_reaped") is True
            )
            cases.append(
                {
                    "mode": mode,
                    "exitcode": process.returncode,
                    "passed": passed,
                    "owner_status": marker,
                    "discovery": discovered,
                    "preview_status": preview["status"],
                    "stdout": stdout[-2048:],
                    "stderr": stderr[-2048:],
                    "files": {str(p): digest(p) for p in (database, store_path)},
                }
            )
    assert sources == capture_sources() | {p: digest(p) for p in paths}
    report = {
        "sources": sources,
        "cases": cases,
        "passed": sum(c["passed"] for c in cases),
        "total": len(cases),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if all(c["passed"] for c in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
