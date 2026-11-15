"""Atomic reviewed-publication receipts across real process exits."""

import argparse
import json
import multiprocessing
import os
from pathlib import Path

from embodied_agent.checkpoint_store import (
    CheckpointStore,
    preview_publication,
    read_checkpoint_store,
    reconcile_publication,
)
from embodied_agent.evidence_checkpoint import CheckpointedEvidenceLedger
from scripts.check_evidence_checkpoint import digest, receipt


def writer(database, store_path, proposal, mode, ack):
    with CheckpointStore(store_path) as store:
        if mode == "before_commit":

            def crash(sql):
                if sql.strip().upper() == "COMMIT":
                    os._exit(91)

            store._connection.set_trace_callback(crash)
        result = store.publish_reviewed(database, proposal)
        if mode == "after_commit":
            os._exit(92)
        Path(ack).write_text(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/check_publication_receipt.py",
        "scripts/check_evidence_checkpoint.py",
        "src/embodied_agent/checkpoint_store.py",
        "src/embodied_agent/checkpoint_cli.py",
        "src/embodied_agent/evidence_checkpoint.py",
        "src/embodied_agent/evidence_ledger.py",
        "src/embodied_agent/indexed_evidence_ledger.py",
    ]
    sources = {p: digest(p) for p in paths}
    cases = []
    for repeat in range(2):
        for mode, code in [("normal", 0), ("before_commit", 91), ("after_commit", 92)]:
            directory = out / f"{mode}-{repeat}"
            directory.mkdir()
            database, store_path, ack = (
                directory / "evidence.sqlite",
                directory / "store.sqlite",
                directory / "ack.json",
            )
            with CheckpointedEvidenceLedger(database, clock=lambda: 1.002) as ledger:
                ledger.record(receipt())
            with CheckpointStore(store_path, create=True) as store:
                store.enroll(database)
            with CheckpointedEvidenceLedger(database, clock=lambda: 1.002) as ledger:
                ledger.record(receipt())
            proposal = preview_publication(database, store_path)["proposal"]
            (directory / "proposal.json").write_text(json.dumps(proposal, indent=2))
            process = multiprocessing.get_context("spawn").Process(
                target=writer, args=(str(database), str(store_path), proposal, mode, str(ack))
            )
            process.start()
            try:
                process.join(10)
                if process.is_alive():
                    process.kill()
                    process.join(1)
                actual = process.exitcode
            finally:
                if process.is_alive():
                    process.kill()
                    process.join()
                process.close()
            # SQLite recovery only; never resubmit a proposal.
            with CheckpointStore(store_path):
                pass
            result = reconcile_publication(store_path, proposal)
            generation = read_checkpoint_store(store_path)["generation"]
            assert actual == code
            assert generation == (1 if mode == "before_commit" else 2)
            assert result["status"] == (
                "no_publication_record" if mode == "before_commit" else "publication_recorded"
            )
            assert ack.exists() == (mode == "normal")
            cases.append(
                {
                    "mode": mode,
                    "exitcode": actual,
                    "ack_received": ack.exists(),
                    "generation": generation,
                    "reconciliation": result,
                    "files": {
                        str(p): digest(p)
                        for p in (database, store_path, directory / "proposal.json")
                    },
                }
            )
    assert sources == {p: digest(p) for p in paths}
    (out / "report.json").write_text(
        json.dumps({"sources": sources, "cases": cases}, indent=2) + "\n"
    )
    print("6/6 crash cases passed; committed proposals reconciled without replay")


if __name__ == "__main__":
    main()
