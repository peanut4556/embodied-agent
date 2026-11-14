"""SQLite audit sink and read-only reconciliation, never motion authorization."""

import hashlib
import json
import math
import re
import sqlite3
import time
from pathlib import Path


def _identity(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise ValueError("32-character receipt identity required")
    return value


def _lookup(connection, receipt_id):
    row = connection.execute(
        "SELECT session_id, payload_sha256, accepted_host_time, payload_json "
        "FROM recent_evidence_v1 WHERE receipt_id=?",
        (receipt_id,),
    ).fetchone()
    base = {"receipt_id": receipt_id, "robot_action_authorized": False}
    if row is None:
        return base | {"status": "not_found"}
    session, digest, accepted, payload = row
    if hashlib.sha256(payload.encode()).hexdigest() != digest:
        return base | {"status": "integrity_error"}
    return base | {
        "status": "recorded",
        "session_id": session,
        "payload_sha256": digest,
        "accepted_host_time": accepted,
    }


def reconcile_evidence(path, receipt_id):
    """Read an existing DB only. Missing/inaccessible DB is not 'not_found'.

    A recorded row means this local transaction committed, not that any external
    effect happened. A missing row is never permission to replay an action.
    """
    _identity(receipt_id)
    try:
        connection = sqlite3.connect(
            Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=0
        )
        try:
            return _lookup(connection, receipt_id)
        finally:
            connection.close()
    except sqlite3.Error as exc:
        return {
            "status": "unavailable",
            "receipt_id": receipt_id,
            "error": str(exc)[:512],
            "robot_action_authorized": False,
        }


class EvidenceLedger:
    """Single-owner local audit writer. Serialization and lock acquisition count
    against the receipt's age budget. Duplicate identical receipts are read-only
    acknowledgements; conflicting payloads cannot overwrite a committed record.
    """

    def __init__(self, path, *, clock=time.monotonic):
        self.clock = clock
        self._connection = sqlite3.connect(path, timeout=0)
        try:
            self._connection.execute("PRAGMA journal_mode=DELETE")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS recent_evidence_v1 ("
                "receipt_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
                "payload_sha256 TEXT NOT NULL, accepted_host_time REAL NOT NULL, "
                "payload_json TEXT NOT NULL)"
            )
        except Exception:
            self._connection.close()
            raise

    def _now(self, receipt, floor):
        value = self.clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < floor
            or value >= receipt["expires_host_time"]
        ):
            raise ValueError("receipt expired or ledger clock invalid")
        return value

    def record(self, receipt):
        identity = _identity(receipt.get("receipt_id"))
        session = receipt.get("session_id")
        if (
            receipt.get("status") != "consumed"
            or receipt.get("robot_action_authorized") is not False
            or not isinstance(session, str)
            or not 1 <= len(session) <= 256
            or receipt.get("evidence", {}).get("status") != "confirmed_recent_evidence"
        ):
            raise ValueError("completed evidence receipt required")
        times = [
            receipt.get(k)
            for k in ("last_exposure_host_time", "consumed_host_time", "expires_host_time")
        ]
        if any(
            isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t)
            for t in times
        ):
            raise ValueError("finite receipt timestamps required")
        if not times[0] <= times[1] < times[2]:
            raise ValueError("ordered receipt timestamps required")
        payload = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(payload.encode()) > 1024 * 1024:
            raise ValueError("receipt exceeds 1 MiB audit limit")
        digest = hashlib.sha256(payload.encode()).hexdigest()
        with self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            existing = _lookup(self._connection, identity)
            if existing["status"] != "not_found":
                if existing.get("payload_sha256") != digest:
                    raise ValueError("receipt identity conflicts with committed payload")
                return existing | {"duplicate": True}
            accepted = self._now(receipt, times[1])
            self._connection.execute(
                "INSERT INTO recent_evidence_v1 VALUES (?, ?, ?, ?, ?)",
                (identity, session, digest, accepted, payload),
            )
            self._now(receipt, accepted)
        return _lookup(self._connection, identity) | {"duplicate": False}

    def close(self):
        self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
