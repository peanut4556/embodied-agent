"""Durable receipt discovery; recovery is read-only and never replays evidence."""

import sqlite3
from pathlib import Path

from embodied_agent.evidence_ledger import EvidenceLedger, _lookup, _receipt_payload


class IndexedEvidenceLedger(EvidenceLedger):
    """Opt-in audit sink usable as EvidenceInbox.dispatch(ledger.record).

    Commit identity before the evidence transaction. A crash between transactions
    leaves an observable intent, not proof of attempted or successful insertion.
    """

    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        try:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS evidence_intents_v1 ("
                "sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
                "receipt_id TEXT NOT NULL UNIQUE, session_id TEXT NOT NULL, "
                "payload_sha256 TEXT NOT NULL)"
            )
        except Exception:
            self.close()
            raise

    def record(self, receipt):
        identity, session, times, _, digest = _receipt_payload(receipt)
        with self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            row = self._connection.execute(
                "SELECT session_id, payload_sha256 FROM evidence_intents_v1 WHERE receipt_id=?",
                (identity,),
            ).fetchone()
            if row is not None and row != (session, digest):
                raise ValueError("receipt conflicts with durable intent")
            if row is None:
                self._now(receipt, times[1])
                self._connection.execute(
                    "INSERT INTO evidence_intents_v1 "
                    "(receipt_id, session_id, payload_sha256) VALUES (?, ?, ?)",
                    (identity, session, digest),
                )
        # Ledger checks time again after intent serialization/commit. Never use
        # the persisted intent to renew the receipt's original deadline.
        return super().record(receipt)


def discover_evidence(path, *, after=0, limit=100):
    """Page durable identities without caller memory or evidence payloads.

    Each page is a read snapshot. Concurrent later inserts appear on later
    scans. An empty page is not a claim that no future intent can be inserted.
    """
    if type(after) is not int or not 0 <= after <= 2**63 - 1:
        raise ValueError("nonnegative SQLite cursor required")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("page limit must be 1..1000")
    base = {"robot_action_authorized": False, "next_after": after, "entries": []}
    try:
        connection = sqlite3.connect(
            Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=0
        )
        try:
            connection.execute("BEGIN")
            rows = connection.execute(
                "SELECT sequence, receipt_id, session_id, payload_sha256 "
                "FROM evidence_intents_v1 WHERE sequence>? ORDER BY sequence LIMIT ?",
                (after, limit),
            ).fetchall()
            entries = []
            for sequence, identity, session, digest in rows:
                result = _lookup(connection, identity)
                if result["status"] == "recorded" and (
                    result["session_id"] != session or result["payload_sha256"] != digest
                ):
                    result = {"status": "intent_conflict", "robot_action_authorized": False}
                entries.append(
                    {
                        "sequence": sequence,
                        "receipt_id": identity,
                        "session_id": session,
                        "reconciliation": result,
                    }
                )
            return base | {
                "status": "scanned",
                "entries": entries,
                "next_after": rows[-1][0] if rows else after,
            }
        finally:
            connection.close()
    except sqlite3.Error as exc:
        return base | {"status": "unavailable", "error": str(exc)[:512]}
