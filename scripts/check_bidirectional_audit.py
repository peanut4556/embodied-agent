"""Reproduce unindexed-tail detection with persisted synthetic audit fixtures."""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from embodied_agent.checkpoint_store import CheckpointStore, discover_from_store
from embodied_agent.evidence_checkpoint import CheckpointedEvidenceLedger
from scripts.check_evidence_checkpoint import digest, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        "src/embodied_agent/evidence_audit.py",
        "src/embodied_agent/checkpoint_store.py",
        "src/embodied_agent/evidence_checkpoint.py",
        "src/embodied_agent/indexed_evidence_ledger.py",
        "src/embodied_agent/evidence_ledger.py",
        "scripts/check_evidence_checkpoint.py",
        "scripts/check_bidirectional_audit.py",
    ]
    sources = {p: digest(p) for p in paths}
    database, store_path = out / "evidence.sqlite", out / "store.sqlite"
    with CheckpointedEvidenceLedger(database, clock=lambda: 1.002) as ledger:
        ledger.record(receipt())
    with CheckpointStore(store_path, create=True) as store:
        assert store.enroll(database)["status"] == "published"
    with CheckpointedEvidenceLedger(database, clock=lambda: 1.002) as ledger:
        ledger.record(receipt())
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM evidence_intents_v1 WHERE sequence=2")
    forward = discover_from_store(database, store_path)
    assert forward["status"] == "scanned"
    assert len(forward["entries"]) == 1
    hashes = {str(p): digest(p) for p in (database, store_path)}
    cases = []
    for limit, expected in [(10000, 5), (1, 4)]:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "embodied_agent.evidence_audit",
                "--database",
                str(database),
                "--store",
                str(store_path),
                "--max-records",
                str(limit),
            ],
            env=os.environ | {"PYTHONPATH": "src:."},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        result = json.loads(process.stdout)
        cases.append({"limit": limit, "exitcode": process.returncode, "result": result})
        assert process.returncode == expected
    assert hashes == {str(p): digest(p) for p in (database, store_path)}
    assert sources == {p: digest(p) for p in paths}
    report = {"sources": sources, "input_hashes": hashes, "forward_scan": forward, "cases": cases}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Unindexed tail blocked; limited reverse scan incomplete; files unchanged")


if __name__ == "__main__":
    main()
