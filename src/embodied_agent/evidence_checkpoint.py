"""Externally retained audit checkpoints detect replacement and history loss.

These are consistency anchors, not signed credentials or execution leases.
"""

import hashlib
import json
import re
import sqlite3
import uuid
from pathlib import Path

from embodied_agent.indexed_evidence_ledger import IndexedEvidenceLedger, _discover_connection


class CheckpointedEvidenceLedger(IndexedEvidenceLedger):
    """Opt-in writer; initialize before capturing time-limited evidence."""

    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        try:
            with self._connection:
                self._connection.execute(
                    "CREATE TABLE IF NOT EXISTS evidence_database_v1 ("
                    "singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
                    "database_id TEXT NOT NULL)"
                )
                self._connection.execute(
                    "INSERT OR IGNORE INTO evidence_database_v1 VALUES (1, ?)",
                    (uuid.uuid4().hex,),
                )
        except Exception:
            self.close()
            raise


def _database_id(connection):
    rows = connection.execute("SELECT database_id FROM evidence_database_v1").fetchall()
    if (
        len(rows) != 1
        or not isinstance(rows[0][0], str)
        or not re.fullmatch(r"[0-9a-f]{32}", rows[0][0])
    ):
        raise ValueError("invalid database identity")
    return rows[0][0]


def _prefix(connection, kind, high_water=None):
    # Fixed queries only. Hash length-framed canonical rows, including payload
    # text, so editing both a payload and its own digest cannot hide history loss.
    if kind == "intents":
        table, key = "evidence_intents_v1", "sequence"
        columns = "sequence, receipt_id, session_id, payload_sha256"
    else:
        table, key = "recent_evidence_v1", "rowid"
        columns = "rowid, receipt_id, session_id, payload_sha256, accepted_host_time, payload_json"
    if high_water is None:
        high_water = connection.execute(f"SELECT COALESCE(MAX({key}), 0) FROM {table}").fetchone()[
            0
        ]
    digest = hashlib.sha256()
    count = 0
    for row in connection.execute(
        f"SELECT {columns} FROM {table} WHERE {key}<=? ORDER BY {key}", (high_water,)
    ):
        encoded = json.dumps(row, separators=(",", ":"), allow_nan=False).encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        count += 1
    return {"high_water": high_water, "count": count, "sha256": digest.hexdigest()}


def _read(path, operation):
    try:
        connection = sqlite3.connect(
            Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=0
        )
        try:
            connection.execute("BEGIN")
            return operation(connection)
        finally:
            connection.close()
    except (sqlite3.Error, ValueError, TypeError, OverflowError) as exc:
        return {
            "status": "unavailable",
            "entries": [],
            "robot_action_authorized": False,
            "error": str(exc)[:512],
        }


def _publication_gate(connection, max_records):
    # Count first to bound the number of payloads parsed, not SQL execution time.
    sizes = [
        connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("evidence_intents_v1", "recent_evidence_v1")
    ]
    base = {"robot_action_authorized": False}
    if max(sizes) > max_records:
        return base | {"status": "validation_incomplete", "row_counts": sizes}
    cursor = 0
    while True:
        page = _discover_connection(connection, cursor, 100)
        for entry in page["entries"]:
            if entry["reconciliation"]["status"] != "recorded":
                return base | {
                    "status": "evidence_rejected",
                    "receipt_id": entry["receipt_id"],
                    "reason": entry["reconciliation"]["status"],
                }
        if not page["entries"]:
            break
        cursor = page["next_after"]
    orphan = connection.execute(
        "SELECT r.receipt_id FROM recent_evidence_v1 r WHERE NOT EXISTS "
        "(SELECT 1 FROM evidence_intents_v1 i WHERE i.receipt_id=r.receipt_id) LIMIT 1"
    ).fetchone()
    if orphan:
        return base | {
            "status": "evidence_rejected",
            "receipt_id": orphan[0],
            "reason": "unindexed",
        }
    return None


def checkpoint_evidence(
    path, *, previous=None, require_complete=False, max_records=10000, expected_candidate=None
):
    """Return a snapshot anchor to retain outside the database's restore domain.

    This records what exists now, not proof that the baseline is complete or
    trustworthy. Pass the retained previous anchor when advancing it; validation
    and capture then use the same snapshot. Omitting previous is initial trust.
    """
    if type(require_complete) is not bool:
        raise ValueError("require_complete must be boolean")
    if type(max_records) is not int or not 1 <= max_records <= 100000:
        raise ValueError("max_records must be 1..100000")
    if expected_candidate is not None:
        _validate(expected_candidate)
    if previous is not None:
        _validate(previous)

    def capture(connection):
        if previous is not None:
            failure = _verify(connection, previous)
            if failure is not None:
                return failure
        if require_complete:
            failure = _publication_gate(connection, max_records)
            if failure is not None:
                return failure
        result = {
            "status": "checkpointed",
            "robot_action_authorized": False,
            "checkpoint": {
                "version": 1,
                "database_id": _database_id(connection),
                "intents": _prefix(connection, "intents"),
                "records": _prefix(connection, "records"),
            },
        }

        if expected_candidate is not None and result["checkpoint"] != expected_candidate:
            return {"status": "proposal_changed", "robot_action_authorized": False}
        return result

    return _read(path, capture)


def _validate(checkpoint):
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "version",
        "database_id",
        "intents",
        "records",
    }:
        raise ValueError("invalid checkpoint fields")
    if type(checkpoint["version"]) is not int or checkpoint["version"] != 1:
        raise ValueError("unsupported checkpoint version")
    if not isinstance(checkpoint["database_id"], str) or not re.fullmatch(
        r"[0-9a-f]{32}", checkpoint["database_id"]
    ):
        raise ValueError("invalid checkpoint database identity")
    for kind in ("intents", "records"):
        prefix = checkpoint[kind]
        if not isinstance(prefix, dict) or set(prefix) != {"high_water", "count", "sha256"}:
            raise ValueError("invalid checkpoint prefix")
        if any(
            type(prefix[k]) is not int or not 0 <= prefix[k] <= 2**63 - 1
            for k in ("high_water", "count")
        ):
            raise ValueError("invalid checkpoint counters")
        if not isinstance(prefix["sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", prefix["sha256"]
        ):
            raise ValueError("invalid checkpoint digest")


def _verify(connection, checkpoint):
    base = {"entries": [], "robot_action_authorized": False}
    if _database_id(connection) != checkpoint["database_id"]:
        return base | {"status": "database_mismatch"}
    for kind in ("intents", "records"):
        if _prefix(connection, kind, checkpoint[kind]["high_water"]) != checkpoint[kind]:
            return base | {"status": "history_mismatch", "component": kind}
    return None


def discover_checkpointed(path, checkpoint, *, after=0, limit=100):
    """Validate historical prefixes and discover in ONE read snapshot.

    New appended records are allowed, including records for earlier pending
    intents. Failure returns no entries and never silently enrolls a new DB.
    """
    _validate(checkpoint)
    if type(after) is not int or not 0 <= after <= 2**63 - 1:
        raise ValueError("invalid page cursor")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("invalid page limit")

    def discover(connection):
        failure = _verify(connection, checkpoint)
        if failure is not None:
            return failure | {"next_after": after}
        return _discover_connection(connection, after, limit) | {"checkpoint_verified": True}

    return _read(path, discover)
