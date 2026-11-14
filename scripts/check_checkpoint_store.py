"""Real publisher crashes; synthetic receipts and virtual receipt clocks."""

import argparse
import json
import multiprocessing
import os
from pathlib import Path

from embodied_agent.checkpoint_store import (
    CheckpointStore,
    discover_from_store,
    read_checkpoint_store,
)
from embodied_agent.evidence_checkpoint import CheckpointedEvidenceLedger
from scripts.check_evidence_checkpoint import digest, receipt


def publish(database, store_path, mode, result_path):
    with CheckpointStore(store_path) as store:
        if mode == "before_commit":

            def crash(sql):
                if sql.strip().upper() == "COMMIT":
                    os._exit(81)

            store._connection.set_trace_callback(crash)
        result = store.advance(database, expected_generation=1)
        if mode == "after_commit":
            os._exit(82)
        Path(result_path).write_text(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/check_checkpoint_store.py",
        "scripts/check_evidence_checkpoint.py",
        "src/embodied_agent/checkpoint_store.py",
        "src/embodied_agent/evidence_checkpoint.py",
        "src/embodied_agent/indexed_evidence_ledger.py",
        "src/embodied_agent/evidence_ledger.py",
    ]
    sources = {p: digest(p) for p in paths}
    cases = []
    for repetition in range(2):
        for mode in ("normal", "before_commit", "after_commit"):
            directory = args.output / f"{mode}-{repetition}"
            directory.mkdir()
            database, store_path = directory / "evidence.sqlite", directory / "checkpoint.sqlite"
            with CheckpointedEvidenceLedger(database, clock=lambda: 1.002) as ledger:
                ledger.record(receipt())
            with CheckpointStore(store_path, create=True) as store:
                assert store.enroll(database)["status"] == "published"
            with CheckpointedEvidenceLedger(database, clock=lambda: 1.002) as ledger:
                ledger.record(receipt())
            ack = directory / "ack.json"
            process = multiprocessing.get_context("spawn").Process(
                target=publish, args=(str(database), str(store_path), mode, str(ack))
            )
            process.start()
            try:
                process.join(10)
                if process.is_alive():
                    process.kill()
                    process.join(1)
                code = process.exitcode
            finally:
                if process.is_alive():
                    process.kill()
                    process.join()
                process.close()
            # Explicit writable connection lets SQLite recover a hot journal;
            # no enroll/advance call, so no checkpoint is republished.
            with CheckpointStore(store_path):
                pass
            head = read_checkpoint_store(store_path)
            page = discover_from_store(database, store_path)
            expected_generation = 1 if mode == "before_commit" else 2
            expected_code = {"normal": 0, "before_commit": 81, "after_commit": 82}[mode]
            passed = (
                code == expected_code
                and head.get("generation") == expected_generation
                and head["checkpoint"]["records"]["count"] == expected_generation
                and page["status"] == "scanned"
                and len(page["entries"]) == 2
                and ack.exists() == (mode == "normal")
            )
            cases.append(
                {
                    "mode": mode,
                    "repetition": repetition,
                    "passed": passed,
                    "exitcode": code,
                    "acknowledged": ack.exists(),
                    "head": head,
                    "recovery": page,
                    "database_path": str(database),
                    "database_sha256": digest(database),
                    "store_path": str(store_path),
                    "store_sha256": digest(store_path),
                }
            )
    assert sources == {p: digest(p) for p in paths}
    report = args.output / "report.json"
    report.write_text(json.dumps({"sources": sources, "cases": cases}, indent=2) + "\n")
    summary = {
        "kind": "checkpoint_store_process_crashes",
        "sources": sources,
        "report_path": str(report),
        "report_sha256": digest(report),
        "cases": cases,
        "passed": sum(c["passed"] for c in cases),
        "total": len(cases),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{summary['passed']}/{summary['total']} checkpoint publication cases passed")
    if not all(c["passed"] for c in cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
