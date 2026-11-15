"""Coverage gap, explicit publication, then stale-backup rejection."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from embodied_agent.checkpoint_store import CheckpointStore
from embodied_agent.evidence_checkpoint import CheckpointedEvidenceLedger
from scripts.check_evidence_checkpoint import digest, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/check_checkpoint_coverage.py",
        "scripts/check_evidence_checkpoint.py",
        "src/embodied_agent/evidence_audit.py",
        "src/embodied_agent/checkpoint_store.py",
        "src/embodied_agent/evidence_checkpoint.py",
        "src/embodied_agent/evidence_ledger.py",
        "src/embodied_agent/indexed_evidence_ledger.py",
    ]
    sources = {p: digest(p) for p in paths}
    database, store_path, backup = out / "evidence.sqlite", out / "store.sqlite", out / "old.sqlite"
    with CheckpointedEvidenceLedger(database, clock=lambda: 1.002) as ledger:
        ledger.record(receipt())
    with CheckpointStore(store_path, create=True) as store:
        assert store.enroll(database)["status"] == "published"
    shutil.copyfile(database, backup)
    with CheckpointedEvidenceLedger(database, clock=lambda: 1.002) as ledger:
        ledger.record(receipt())
    cases = []
    for stage, expected in [("gap", 6), ("published", 0), ("rollback", 5)]:
        if stage == "published":
            with CheckpointStore(store_path) as store:
                assert store.advance(database, expected_generation=1)["status"] == "published"
        elif stage == "rollback":
            shutil.copyfile(backup, database)
        hashes = {str(p): digest(p) for p in (database, store_path)}
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "embodied_agent.evidence_audit",
                "--database",
                str(database),
                "--store",
                str(store_path),
            ],
            env=os.environ | {"PYTHONPATH": "src:."},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        result = json.loads(process.stdout)
        assert process.returncode == expected
        assert hashes == {str(p): digest(p) for p in (database, store_path)}
        # Preserve each audited state, not merely the final mutated files.
        snapshots = {}
        for label, source in [("database", database), ("store", store_path)]:
            target = out / f"{stage}-{label}.sqlite"
            shutil.copyfile(source, target)
            snapshots[str(target)] = digest(target)
        cases.append(
            {
                "stage": stage,
                "exitcode": process.returncode,
                "result": result,
                "snapshots": snapshots,
            }
        )
    assert sources == {p: digest(p) for p in paths}
    (out / "report.json").write_text(
        json.dumps({"sources": sources, "cases": cases}, indent=2) + "\n"
    )
    print("Coverage gap -> explicit publication -> old-backup rejection: 3/3 passed")


if __name__ == "__main__":
    main()
