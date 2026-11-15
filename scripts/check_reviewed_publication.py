"""Reject a reviewed proposal after valid source growth, then explicitly re-preview."""

import argparse
import json
from pathlib import Path

from embodied_agent.checkpoint_store import (
    CheckpointStore,
    preview_publication,
    read_checkpoint_store,
)
from embodied_agent.evidence_checkpoint import CheckpointedEvidenceLedger
from scripts.check_evidence_checkpoint import digest, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/check_reviewed_publication.py",
        "scripts/check_evidence_checkpoint.py",
        "src/embodied_agent/checkpoint_store.py",
        "src/embodied_agent/evidence_checkpoint.py",
        "src/embodied_agent/indexed_evidence_ledger.py",
        "src/embodied_agent/evidence_ledger.py",
    ]
    sources = {p: digest(p) for p in paths}
    database, store_path = out / "evidence.sqlite", out / "store.sqlite"
    with CheckpointedEvidenceLedger(database, clock=lambda: 1.002) as ledger:
        ledger.record(receipt())
    with CheckpointStore(store_path, create=True) as store:
        store.enroll(database)
    with CheckpointedEvidenceLedger(database, clock=lambda: 1.002) as ledger:
        ledger.record(receipt())
    original = preview_publication(database, store_path)["proposal"]
    with CheckpointedEvidenceLedger(database, clock=lambda: 1.002) as ledger:
        ledger.record(receipt())
    before = read_checkpoint_store(store_path)
    with CheckpointStore(store_path) as store:
        rejected = store.publish_reviewed(database, original)
    assert rejected["source"]["status"] == "proposal_changed"
    assert read_checkpoint_store(store_path) == before
    revised = preview_publication(database, store_path)["proposal"]
    with CheckpointStore(store_path) as store:
        published = store.publish_reviewed(database, revised)
        repeated = store.publish_reviewed(database, revised)
    assert published["status"] == "published"
    assert repeated["status"] == "generation_conflict"
    assert read_checkpoint_store(store_path)["checkpoint"] == revised["checkpoint"]
    assert sources == {p: digest(p) for p in paths}
    report = {
        "sources": sources,
        "original": original,
        "revised": revised,
        "rejected": rejected,
        "published": published,
        "repeated": repeated,
        "files": {str(p): digest(p) for p in (database, store_path)},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Changed proposal rejected; revised proposal published; repeated proposal refused")


if __name__ == "__main__":
    main()
