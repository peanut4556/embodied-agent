"""Digest-consistent malformed tail records must fail read-only audit."""

import argparse
import hashlib
import json
import os
import sqlite3
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
        "scripts/check_evidence_semantics.py",
        "scripts/check_evidence_checkpoint.py",
        "src/embodied_agent/evidence_ledger.py",
        "src/embodied_agent/evidence_audit.py",
        "src/embodied_agent/checkpoint_store.py",
        "src/embodied_agent/evidence_checkpoint.py",
        "src/embodied_agent/indexed_evidence_ledger.py",
    ]
    sources = {p: digest(p) for p in paths}
    cases = []
    for mode in ("expired_acceptance", "body_session", "body_authorization"):
        directory = out / mode
        directory.mkdir()
        database, store_path = directory / "evidence.sqlite", directory / "store.sqlite"
        with CheckpointedEvidenceLedger(database, clock=lambda: 1.002) as ledger:
            ledger.record(receipt())
        with CheckpointStore(store_path, create=True) as store:
            assert store.enroll(database)["status"] == "published"
        item = receipt()
        with CheckpointedEvidenceLedger(database, clock=lambda: 1.002) as ledger:
            ledger.record(item)
        with sqlite3.connect(database) as connection:
            if mode == "expired_acceptance":
                connection.execute(
                    "UPDATE recent_evidence_v1 SET accepted_host_time=2 WHERE rowid=2"
                )
            else:
                changed = item | (
                    {"session_id": "other"}
                    if mode == "body_session"
                    else {"robot_action_authorized": True}
                )
                payload = json.dumps(changed, sort_keys=True, separators=(",", ":"))
                hashed = hashlib.sha256(payload.encode()).hexdigest()
                connection.execute(
                    "UPDATE recent_evidence_v1 SET payload_json=?, payload_sha256=? WHERE rowid=2",
                    (payload, hashed),
                )
                connection.execute(
                    "UPDATE evidence_intents_v1 SET payload_sha256=? WHERE sequence=2", (hashed,)
                )
            rows = connection.execute(
                "SELECT r.payload_json,r.payload_sha256,i.payload_sha256,r.session_id,i.session_id FROM recent_evidence_v1 r JOIN evidence_intents_v1 i USING(receipt_id)"
            ).fetchall()
            assert all(
                hashlib.sha256(p.encode()).hexdigest() == h == ih and s == ins
                for p, h, ih, s, ins in rows
            )
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
        assert process.returncode == 5 and result["counts"]["invalid_record"] == 1
        assert result["checkpoint_history_verified"]
        assert hashes == {str(p): digest(p) for p in (database, store_path)}
        cases.append(
            {
                "mode": mode,
                "digest_and_index_binding_match": True,
                "exitcode": process.returncode,
                "result": result,
                "input_hashes": hashes,
            }
        )
    assert sources == {p: digest(p) for p in paths}
    (out / "report.json").write_text(
        json.dumps({"sources": sources, "cases": cases}, indent=2) + "\n"
    )
    print("3/3 digest-consistent semantic faults rejected; files unchanged")


if __name__ == "__main__":
    main()
