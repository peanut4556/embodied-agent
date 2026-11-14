"""Fresh-process recovery from independent JSON anchors and altered SQLite copies.

Synthetic audit receipts and a virtual receipt clock; real files and OS processes.
"""

import argparse
import hashlib
import json
import multiprocessing
import shutil
import sqlite3
import uuid
from pathlib import Path

from embodied_agent.evidence_checkpoint import (
    CheckpointedEvidenceLedger,
    checkpoint_evidence,
    discover_checkpointed,
)


def receipt():
    return {
        "receipt_id": uuid.uuid4().hex,
        "session_id": uuid.uuid4().hex,
        "status": "consumed",
        "robot_action_authorized": False,
        "last_exposure_host_time": 1.0,
        "consumed_host_time": 1.001,
        "expires_host_time": 1.02,
        "evidence": {"status": "confirmed_recent_evidence", "fixture": True},
    }


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recover(database, anchor, output):
    checkpoint = json.loads(Path(anchor).read_text())
    result = discover_checkpointed(database, checkpoint)
    Path(output).write_text(json.dumps(result, indent=2) + "\n")


def run(directory, mode):
    directory.mkdir()
    path = directory / "audit.sqlite"
    backup = directory / "older.sqlite"
    item = receipt()
    with CheckpointedEvidenceLedger(path, clock=lambda: 1.002):
        pass
    shutil.copyfile(path, backup)
    if mode == "pending_completion":
        ticks = iter([1.002, 1.02])
        with CheckpointedEvidenceLedger(path, clock=lambda: next(ticks)) as ledger:
            try:
                ledger.record(item)
            except ValueError:
                pass
    else:
        with CheckpointedEvidenceLedger(path, clock=lambda: 1.002) as ledger:
            ledger.record(item)
    checkpoint = checkpoint_evidence(path)["checkpoint"]
    anchor_path = directory / "external-checkpoint.json"
    anchor_path.write_text(json.dumps(checkpoint, indent=2) + "\n")
    expected = "scanned"
    if mode == "append":
        with CheckpointedEvidenceLedger(path, clock=lambda: 1.002) as ledger:
            ledger.record(receipt())
    elif mode == "pending_completion":
        with CheckpointedEvidenceLedger(path, clock=lambda: 1.002) as ledger:
            ledger.record(item)
    elif mode == "wrong_database":
        other = directory / "other.sqlite"
        with CheckpointedEvidenceLedger(other):
            pass
        shutil.copyfile(other, path)
        expected = "database_mismatch"
    elif mode == "restore_older":
        shutil.copyfile(backup, path)
        expected = "history_mismatch"
    elif mode in ("erase_record", "erase_intent"):
        table = "recent_evidence_v1" if mode == "erase_record" else "evidence_intents_v1"
        with sqlite3.connect(path) as connection:
            connection.execute(f"DELETE FROM {table}")
        expected = "history_mismatch"
    elif mode == "consistent_rewrite":
        payload = json.dumps(item | {"changed": True}, sort_keys=True, separators=(",", ":"))
        hashed = hashlib.sha256(payload.encode()).hexdigest()
        with sqlite3.connect(path) as connection:
            connection.execute(
                "UPDATE recent_evidence_v1 SET payload_json=?, payload_sha256=?", (payload, hashed)
            )
            connection.execute("UPDATE evidence_intents_v1 SET payload_sha256=?", (hashed,))
        expected = "history_mismatch"
    elif mode == "unanchored_tail_loss":
        shutil.copyfile(path, backup)
        with CheckpointedEvidenceLedger(path, clock=lambda: 1.002) as ledger:
            ledger.record(receipt())
        shutil.copyfile(backup, path)
        # Expected limitation: no anchor ever witnessed the lost tail.
    before = digest(path)
    result_path = directory / "recovery.json"
    process = multiprocessing.get_context("spawn").Process(
        target=recover, args=(str(path), str(anchor_path), str(result_path))
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
    result = (
        json.loads(result_path.read_text()) if result_path.exists() else {"status": "no_result"}
    )
    passed = code == 0 and result["status"] == expected and digest(path) == before
    if expected == "scanned":
        passed = (
            passed
            and len(result["entries"]) == (2 if mode == "append" else 1)
            and all(entry["reconciliation"]["status"] == "recorded" for entry in result["entries"])
        )
    else:
        passed = passed and result["entries"] == []
    return {
        "mode": mode,
        "expected": expected,
        "passed": passed,
        "exitcode": code,
        "recovery": result,
        "database_path": str(path),
        "database_sha256": digest(path),
        "anchor_path": str(anchor_path),
        "anchor_sha256": digest(anchor_path),
        "read_only_verified": digest(path) == before,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/check_evidence_checkpoint.py",
        "src/embodied_agent/evidence_checkpoint.py",
        "src/embodied_agent/indexed_evidence_ledger.py",
        "src/embodied_agent/evidence_ledger.py",
    ]
    sources = {path: digest(path) for path in paths}
    modes = [
        "unchanged",
        "append",
        "pending_completion",
        "wrong_database",
        "restore_older",
        "erase_record",
        "erase_intent",
        "consistent_rewrite",
        "unanchored_tail_loss",
    ]
    cases = [run(args.output / f"{mode}-{repeat}", mode) for repeat in range(2) for mode in modes]
    assert sources == {path: digest(path) for path in paths}
    report = args.output / "report.json"
    report.write_text(json.dumps({"sources": sources, "cases": cases}, indent=2) + "\n")
    summary = {
        "kind": "checkpoint_recovery_virtual_receipts_real_processes",
        "sources": sources,
        "report_path": str(report),
        "report_sha256": digest(report),
        "passed": sum(case["passed"] for case in cases),
        "total": len(cases),
        "cases": cases,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{summary['passed']}/{summary['total']} checkpoint recovery cases passed")
    if not all(case["passed"] for case in cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
