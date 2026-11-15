"""Real concurrent publishers and lock-owner death; synthetic audit inputs."""

import argparse
import json
import multiprocessing
import os
from pathlib import Path

from embodied_agent.checkpoint_store import CheckpointStore, read_checkpoint_store
from embodied_agent.evidence_checkpoint import CheckpointedEvidenceLedger
from scripts.check_evidence_checkpoint import digest, receipt


def worker(pipe, database, store_path):
    with CheckpointStore(store_path) as store:
        pipe.send("ready")
        while True:
            command = pipe.recv()
            if command == "stop":
                return
            if command == "lock":
                store._connection.execute("BEGIN IMMEDIATE")
                pipe.send("locked")
            elif command == "die":
                os._exit(83)
            else:
                pipe.send(store.advance(database, expected_generation=command))


def receive(pipe):
    if not pipe.poll(5):
        raise TimeoutError("worker response timeout")
    return pipe.recv()


def run(directory, mode):
    directory.mkdir()
    database, store_path = directory / "evidence.sqlite", directory / "store.sqlite"
    with CheckpointedEvidenceLedger(database, clock=lambda: 1.002) as ledger:
        ledger.record(receipt())
    with CheckpointStore(store_path, create=True) as store:
        assert store.enroll(database)["status"] == "published"
    with CheckpointedEvidenceLedger(database, clock=lambda: 1.002) as ledger:
        ledger.record(receipt())
    ctx = multiprocessing.get_context("spawn")
    processes, pipes, events = [], [], []
    try:
        # Prewarm serially so initialization does not masquerade as publication contention.
        for _ in range(3):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=worker, args=(child, str(database), str(store_path)))
            process.start()
            child.close()
            processes.append(process)
            pipes.append(parent)
            assert receive(parent) == "ready"
        if mode == "race":
            for pipe in pipes:
                pipe.send(1)
            events = [receive(pipe) for pipe in pipes]
            assert sum(e["status"] == "published" for e in events) == 1
            assert all(
                e["status"] in ("published", "generation_conflict", "publication_unconfirmed")
                for e in events
            )
        else:
            pipes[0].send("lock")
            assert receive(pipes[0]) == "locked"
            pipes[1].send(1)
            events.append(receive(pipes[1]))
            assert events[-1]["status"] == "publication_unconfirmed"
            pipes[0].send("die")
            processes[0].join(5)
            assert processes[0].exitcode == 83
            # Explicitly reconcile before attempting a new publication.
            assert read_checkpoint_store(store_path)["generation"] == 1
            pipes[2].send(1)
            events.append(receive(pipes[2]))
            assert events[-1]["status"] == "published"
        head = read_checkpoint_store(store_path)
        assert head["generation"] == 2 and head["checkpoint"]["records"]["count"] == 2
        # A surviving stale publisher must not advance generation a second time.
        pipes[1].send(1)
        stale = receive(pipes[1])
        assert stale["status"] == "generation_conflict"
        return {
            "mode": mode,
            "events": events,
            "stale": stale,
            "head": head,
            "database_path": str(database),
            "database_sha256": digest(database),
            "store_path": str(store_path),
            "store_sha256": digest(store_path),
        }
    finally:
        for process, pipe in zip(processes, pipes, strict=True):
            if process.is_alive():
                try:
                    pipe.send("stop")
                except (BrokenPipeError, EOFError):
                    pass
                process.join(1)
            if process.is_alive():
                process.kill()
                process.join(1)
            assert not process.is_alive()
            pipe.close()
            process.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/check_checkpoint_contention.py",
        "scripts/check_evidence_checkpoint.py",
        "src/embodied_agent/checkpoint_store.py",
        "src/embodied_agent/evidence_checkpoint.py",
        "src/embodied_agent/indexed_evidence_ledger.py",
        "src/embodied_agent/evidence_ledger.py",
    ]
    sources = {p: digest(p) for p in paths}
    cases = []
    for repeat in range(3):
        for mode in ("race", "owner_death"):
            cases.append(run(args.output / f"{mode}-{repeat}", mode))
    assert sources == {p: digest(p) for p in paths}
    report = args.output / "report.json"
    report.write_text(json.dumps({"sources": sources, "cases": cases}, indent=2) + "\n")
    summary = {
        "sources": sources,
        "report_path": str(report),
        "report_sha256": digest(report),
        "cases": cases,
        "passed": len(cases),
        "workers_reaped": len(cases) * 3,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{len(cases)} contention scenarios passed; all workers reaped")


if __name__ == "__main__":
    main()
