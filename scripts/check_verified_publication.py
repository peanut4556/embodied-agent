"""Strict checkpoint publication over valid and faulty synthetic tails."""

import argparse
import json
import sqlite3
from pathlib import Path

from embodied_agent.checkpoint_store import CheckpointStore, read_checkpoint_store
from embodied_agent.evidence_checkpoint import CheckpointedEvidenceLedger
from scripts.check_evidence_checkpoint import digest, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/check_verified_publication.py",
        "scripts/check_evidence_checkpoint.py",
        "src/embodied_agent/checkpoint_store.py",
        "src/embodied_agent/evidence_checkpoint.py",
        "src/embodied_agent/evidence_ledger.py",
        "src/embodied_agent/indexed_evidence_ledger.py",
    ]
    sources = {p: digest(p) for p in paths}
    cases = []
    faults = {
        "valid": None,
        "unindexed": "DELETE FROM evidence_intents_v1 WHERE sequence=2",
        "pending": "DELETE FROM recent_evidence_v1 WHERE rowid=2",
        "invalid_time": "UPDATE recent_evidence_v1 SET accepted_host_time=2 WHERE rowid=2",
        "corrupt": "UPDATE recent_evidence_v1 SET payload_json='{}' WHERE rowid=2",
        "limited": None,
    }
    for mode, sql in faults.items():
        directory = out / mode
        directory.mkdir()
        database, store_path = directory / "evidence.sqlite", directory / "store.sqlite"
        with CheckpointedEvidenceLedger(database, clock=lambda: 1.002) as ledger:
            ledger.record(receipt())
        with CheckpointStore(store_path, create=True) as store:
            store.enroll(database)
        with CheckpointedEvidenceLedger(database, clock=lambda: 1.002) as ledger:
            ledger.record(receipt())
        if sql:
            with sqlite3.connect(database) as connection:
                connection.execute(sql)
        before = read_checkpoint_store(store_path)
        source_hash = digest(database)
        with CheckpointStore(store_path) as store:
            result = store.advance_verified(
                database, expected_generation=1, max_records=1 if mode == "limited" else 10000
            )
        after = read_checkpoint_store(store_path)
        assert digest(database) == source_hash
        if mode == "valid":
            assert result["status"] == "published" and after["generation"] == 2
        else:
            assert result["status"] == "source_rejected" and after == before
        cases.append(
            {
                "mode": mode,
                "result": result,
                "head": after,
                "files": {str(p): digest(p) for p in (database, store_path)},
            }
        )
    assert sources == {p: digest(p) for p in paths}
    (out / "report.json").write_text(
        json.dumps({"sources": sources, "cases": cases}, indent=2) + "\n"
    )
    print("6/6 strict publication scenarios passed")


if __name__ == "__main__":
    main()
