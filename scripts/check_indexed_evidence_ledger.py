"""Lose all writer memory; discover audit outcomes from disk alone.

Receipts are synthetic audit fixtures, not live camera confirmations.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import time
import uuid
from pathlib import Path

from embodied_agent.indexed_evidence_ledger import IndexedEvidenceLedger, discover_evidence


def writer(path, mode):
    with IndexedEvidenceLedger(path) as ledger:
        commits = 0

        def crash(sql):
            nonlocal commits
            if sql.strip().upper() == "COMMIT":
                commits += 1
                if mode == "before_intent" and commits == 1:
                    os._exit(71)
                if mode == "before_evidence" and commits == 2:
                    os._exit(72)
            if sql.strip().upper() == "BEGIN IMMEDIATE" and commits == 1 and mode == "after_intent":
                os._exit(73)

        ledger._connection.set_trace_callback(crash)
        now = time.monotonic()
        receipt = {
            "receipt_id": uuid.uuid4().hex,
            "session_id": uuid.uuid4().hex,
            "status": "consumed",
            "robot_action_authorized": False,
            "evidence": {"status": "confirmed_recent_evidence", "fixture": True},
            "last_exposure_host_time": now,
            "consumed_host_time": now,
            "expires_host_time": now + 0.02,
        }
        ledger.record(receipt)
        if mode == "after_evidence":
            os._exit(74)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = [
        "src/embodied_agent/evidence_ledger.py",
        "src/embodied_agent/indexed_evidence_ledger.py",
        __file__,
    ]
    sources = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
    cases = []
    for repetition in range(2):
        for mode, code, status in [
            ("normal", 0, "recorded"),
            ("before_intent", 71, None),
            ("after_intent", 73, "not_found"),
            ("before_evidence", 72, "not_found"),
            ("after_evidence", 74, "recorded"),
        ]:
            path = args.output / f"{mode}-{repetition}.sqlite"
            process = multiprocessing.get_context("spawn").Process(
                target=writer, args=(str(path), mode)
            )
            process.start()
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join(1)
            actual_code = process.exitcode
            process.close()
            # Explicit writable open permits SQLite hot-journal rollback only.
            with IndexedEvidenceLedger(path):
                pass
            result = discover_evidence(path)
            expected = [] if status is None else [status]
            passed = (
                actual_code == code
                and result["status"] == "scanned"
                and [e["reconciliation"]["status"] for e in result["entries"]] == expected
            )
            cases.append(
                {
                    "mode": mode,
                    "repetition": repetition,
                    "exitcode": actual_code,
                    "passed": passed,
                    "discovery": result,
                    "database": str(path),
                    "database_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    assert sources == {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
    report = args.output / "report.json"
    report.write_text(json.dumps({"sources": sources, "cases": cases}, indent=2) + "\n")
    summary = {
        "kind": "caller_memory_loss_audit_fixture",
        "sources": sources,
        "report_path": str(report),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "cases": cases,
        "passed": sum(c["passed"] for c in cases),
        "total": len(cases),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{summary['passed']}/{summary['total']} crash discovery cases passed")
    if not all(c["passed"] for c in cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
