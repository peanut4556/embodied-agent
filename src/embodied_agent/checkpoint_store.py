"""Explicitly enrolled, transactional local checkpoint storage with version checks."""

import hashlib
import json
import os
import sqlite3
from pathlib import Path

from embodied_agent.evidence_checkpoint import (
    _validate,
    checkpoint_evidence,
    discover_checkpointed,
)


def _checksum(generation, payload):
    return hashlib.sha256(f"{generation}:{payload}".encode()).hexdigest()


def _load(connection):
    row = connection.execute(
        "SELECT generation, checkpoint_json, sha256 FROM checkpoint_head_v1 WHERE singleton=1"
    ).fetchone()
    if row is None:
        return {"status": "unenrolled", "robot_action_authorized": False}
    generation, payload, digest = row
    try:
        if type(generation) is not int or generation < 1:
            raise ValueError("invalid generation")
        if _checksum(generation, payload) != digest:
            raise ValueError("checkpoint checksum mismatch")
        checkpoint = json.loads(payload)
        _validate(checkpoint)
    except (ValueError, TypeError, AttributeError) as exc:
        return {
            "status": "integrity_error",
            "robot_action_authorized": False,
            "error": str(exc)[:512],
        }
    return {
        "status": "loaded",
        "generation": generation,
        "checkpoint": checkpoint,
        "robot_action_authorized": False,
    }


def _separate(store, database):
    store, database = Path(store).resolve(), Path(database).resolve()
    if store == database or (
        store.exists() and database.exists() and os.path.samefile(store, database)
    ):
        raise ValueError("checkpoint store must be a separate database file")


class CheckpointStore:
    """Single-owner connection. create=True explicitly initializes an empty store.

    Enrollment is a separate trust decision; advance never enrolls implicitly.
    Source verification runs while holding the store write transaction, so another
    publisher cannot replace the head between validation and commit.
    """

    def __init__(self, path, *, create=False):
        self.path = Path(path).resolve()
        mode = "rwc" if create else "rw"
        self._connection = sqlite3.connect(
            self.path.as_uri() + f"?mode={mode}", uri=True, timeout=0
        )
        try:
            self._connection.execute("PRAGMA journal_mode=DELETE")
            self._connection.execute("PRAGMA synchronous=FULL")
            if create:
                self._connection.execute(
                    "CREATE TABLE IF NOT EXISTS checkpoint_head_v1 ("
                    "singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
                    "generation INTEGER NOT NULL, checkpoint_json TEXT NOT NULL, sha256 TEXT NOT NULL)"
                )
        except Exception:
            self.close()
            raise

    def enroll(self, database):
        return self._publish(database, expected=None)

    def advance(self, database, *, expected_generation):
        if type(expected_generation) is not int or not 1 <= expected_generation < 2**63 - 1:
            raise ValueError("positive generation below SQLite maximum required")
        return self._publish(database, expected=expected_generation)

    def advance_verified(self, database, *, expected_generation, max_records=10000):
        """Publish only fully indexed, valid, committed evidence in the source snapshot."""
        if type(expected_generation) is not int or not 1 <= expected_generation < 2**63 - 1:
            raise ValueError("positive generation below SQLite maximum required")
        if type(max_records) is not int or not 1 <= max_records <= 100000:
            raise ValueError("max_records must be 1..100000")
        return self._publish(
            database, expected_generation, require_complete=True, max_records=max_records
        )

    def _publish(self, database, expected, *, require_complete=False, max_records=10000):
        _separate(self.path, database)
        base = {"robot_action_authorized": False}
        try:
            with self._connection:
                self._connection.execute("BEGIN IMMEDIATE")
                current = _load(self._connection)
                if current["status"] == "integrity_error":
                    return current
                if expected is None:
                    if current["status"] != "unenrolled":
                        return base | {
                            "status": "already_enrolled",
                            "generation": current["generation"],
                        }
                    captured = checkpoint_evidence(database)
                    generation = 1
                else:
                    if current["status"] == "unenrolled":
                        return current
                    if current["generation"] != expected:
                        return base | {
                            "status": "generation_conflict",
                            "generation": current["generation"],
                        }
                    captured = checkpoint_evidence(
                        database,
                        previous=current["checkpoint"],
                        require_complete=require_complete,
                        max_records=max_records,
                    )
                    generation = expected + 1
                if captured["status"] != "checkpointed":
                    return base | {"status": "source_rejected", "source": captured}
                payload = json.dumps(
                    captured["checkpoint"], sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                digest = _checksum(generation, payload)
                self._connection.execute(
                    "INSERT INTO checkpoint_head_v1 VALUES (1, ?, ?, ?) "
                    "ON CONFLICT(singleton) DO UPDATE SET generation=excluded.generation, "
                    "checkpoint_json=excluded.checkpoint_json, sha256=excluded.sha256",
                    (generation, payload, digest),
                )
            return base | {"status": "published", "generation": generation}
        except sqlite3.Error as exc:
            # Commit errors can have uncertain outcomes. Reopen/read to reconcile;
            # never interpret this response as proof that no publication occurred.
            return base | {"status": "publication_unconfirmed", "error": str(exc)[:512]}

    def close(self):
        self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def read_checkpoint_store(path):
    """Read existing store only; missing or corrupt storage cannot re-enroll."""
    return _read_store(path, _load)


def _read_store(path, operation):
    try:
        connection = sqlite3.connect(
            Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=0
        )
        try:
            connection.execute("BEGIN")
            return operation(connection)
        finally:
            connection.close()
    except sqlite3.Error as exc:
        return {"status": "unavailable", "robot_action_authorized": False, "error": str(exc)[:512]}


def discover_from_store(database, store, *, after=0, limit=100):
    """Recover against the head observed in this store read transaction.

    A concurrent later publication is not included. The two databases are not a
    distributed atomic snapshot; the source may append after checkpoint capture.
    """
    _separate(store, database)

    def discover(connection):
        head = _load(connection)
        if head["status"] != "loaded":
            return head | {"entries": []}
        return discover_checkpointed(database, head["checkpoint"], after=after, limit=limit) | {
            "checkpoint_generation": head["generation"]
        }

    result = _read_store(store, discover)
    return {"entries": []} | result
