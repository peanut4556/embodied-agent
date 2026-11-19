"""Explicit SQLite threshold pool; committed reservations are never refunded/reused."""

import copy
import hashlib
import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path


def validate_checkpoint(expected_checkpoint):
    if expected_checkpoint is not None:
        c = expected_checkpoint
        if (
            not isinstance(c, dict)
            or set(c) != {"version", "pool_id", "reserved_slots", "sha256"}
            or type(c["version"]) is not int
            or c["version"] != 1
            or not isinstance(c["pool_id"], str)
            or not 1 <= len(c["pool_id"]) <= 256
            or type(c["reserved_slots"]) is not int
            or not 0 <= c["reserved_slots"] <= 3
            or not isinstance(c["sha256"], str)
            or len(c["sha256"]) != 64
            or any(ch not in "0123456789abcdef" for ch in c["sha256"])
        ):
            raise ValueError("invalid budget checkpoint")


class WaitBudgetStore:
    """Local durable allocation, not process ownership or a hard CPU quota.

    Create once, then reopen the same path. Missing stores are never recreated.
    Keep on a local filesystem; deletion/rollback/replacement is out of scope.
    """

    @classmethod
    def create(cls, path, *, total_threshold_ns, slots):
        if type(total_threshold_ns) is not int or not 0 <= total_threshold_ns <= 1_000_000_000:
            raise ValueError("threshold must be integer nanoseconds in [0, 1e9]")
        if type(slots) is not int or not 1 <= slots <= 3:
            raise ValueError("slots must be 1..3")
        path = Path(path).resolve()
        # Exclusive creation avoids accidentally resetting an existing pool.
        with path.open("xb"):
            pass
        with closing(sqlite3.connect(path)) as db:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                db.execute(
                    "CREATE TABLE config (id INTEGER PRIMARY KEY CHECK(id=1), pool TEXT NOT NULL, total INTEGER NOT NULL, slots INTEGER NOT NULL)"
                )
                db.execute(
                    "CREATE TABLE reservations (slot INTEGER PRIMARY KEY, request TEXT UNIQUE NOT NULL, allowance INTEGER NOT NULL)"
                )
                db.execute(
                    "INSERT INTO config VALUES (1, ?, ?, ?)",
                    (uuid.uuid4().hex, total_threshold_ns, slots),
                )
        return cls(path)

    def __init__(self, path, *, expected_pool_id=None, expected_checkpoint=None):
        if expected_pool_id is not None and (
            not isinstance(expected_pool_id, str) or not 1 <= len(expected_pool_id) <= 256
        ):
            raise ValueError("bounded expected pool identity required")
        validate_checkpoint(expected_checkpoint)
        self._expected_checkpoint = copy.deepcopy(expected_checkpoint)
        self.path = Path(path).resolve()
        with closing(self._connect()) as db, db:
            db.execute("BEGIN")
            self.pool_id, self.total_ns, self.slots = self._config(db)
            if expected_pool_id is not None and self.pool_id != expected_pool_id:
                raise ValueError("unexpected budget pool identity")
            self._rows(db)

    def _connect(self):
        db = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=0.25)
        db.execute("PRAGMA synchronous=FULL")
        return db

    @staticmethod
    def _config(db):
        rows = db.execute("SELECT pool, total, slots FROM config WHERE id=1").fetchall()
        if len(rows) != 1:
            raise ValueError("invalid budget config")
        pool, total, slots = rows[0]
        if (
            not isinstance(pool, str)
            or not pool
            or type(total) is not int
            or not 0 <= total <= 1_000_000_000
            or type(slots) is not int
            or not 1 <= slots <= 3
        ):
            raise ValueError("invalid budget config")
        return pool, total, slots

    def _rows(self, db):
        if self._config(db) != (self.pool_id, self.total_ns, self.slots):
            raise ValueError("budget store identity changed")
        rows = db.execute(
            "SELECT slot, request, allowance FROM reservations ORDER BY slot"
        ).fetchall()
        if len(rows) > self.slots:
            raise ValueError("invalid reservations")
        for i, (slot, request, allowance) in enumerate(rows, 1):
            if (
                type(slot) is not int
                or slot != i
                or not isinstance(request, str)
                or not 1 <= len(request) <= 256
                or type(allowance) is not int
                or allowance != self._share(i)
            ):
                raise ValueError("invalid reservations")
        if self._expected_checkpoint is not None:
            checkpoint = self._expected_checkpoint
            count = checkpoint["reserved_slots"]
            if len(rows) < count or self._checkpoint(rows[:count]) != checkpoint:
                raise ValueError("budget history differs from checkpoint")
        return rows

    def _checkpoint(self, rows):
        payload = [self.pool_id, self.total_ns, self.slots, rows]
        encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode()
        return {
            "version": 1,
            "pool_id": self.pool_id,
            "reserved_slots": len(rows),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }

    def checkpoint(self):
        """Export a history baseline for independent storage, never an allocation grant."""
        with closing(self._connect()) as db, db:
            db.execute("BEGIN")
            return self._checkpoint(self._rows(db))

    def _share(self, slot):
        return self.total_ns // self.slots + (1 if slot <= self.total_ns % self.slots else 0)

    def reserve(self, request_id, *, reviewed_checkpoint=None):
        """Optionally require the entire reviewed state, checked inside the write transaction."""
        validate_checkpoint(reviewed_checkpoint)
        reviewed_checkpoint = copy.deepcopy(reviewed_checkpoint)
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 256:
            raise ValueError("bounded request identity required")
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            rows = self._rows(db)
            if reviewed_checkpoint is not None and self._checkpoint(rows) != reviewed_checkpoint:
                raise RuntimeError("reviewed budget checkpoint is stale")
            if any(row[1] == request_id for row in rows):
                raise RuntimeError("reservation already spent; replay prohibited")
            if len(rows) >= self.slots:
                raise RuntimeError("persistent wait budget exhausted")
            slot = len(rows) + 1
            allowance = self._share(slot)
            db.execute("INSERT INTO reservations VALUES (?, ?, ?)", (slot, request_id, allowance))
            # Return is only observed after context-manager commit succeeds.
            result = {
                "pool_id": self.pool_id,
                "slot": slot,
                "request_id": request_id,
                "allowance_ns": allowance,
                "remaining_ns": self.total_ns - sum(r[2] for r in rows) - allowance,
            }
        return result

    def lookup(self, request_id):
        """Reconcile a historical allocation without replaying or granting execution."""
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 256:
            raise ValueError("bounded request identity required")
        with closing(self._connect()) as db, db:
            db.execute("BEGIN")
            rows = self._rows(db)
            remaining = self.total_ns
            for slot, identity, allowance in rows:
                remaining -= allowance
                if identity == request_id:
                    return {
                        "status": "reserved",
                        "execution_verified": False,
                        "receipt": {
                            "pool_id": self.pool_id,
                            "slot": slot,
                            "request_id": identity,
                            "allowance_ns": allowance,
                            "remaining_ns": remaining,
                        },
                    }
            return {"status": "not_found", "execution_verified": False, "receipt": None}

    def snapshot(self):
        with closing(self._connect()) as db, db:
            db.execute("BEGIN")
            rows = self._rows(db)
            return {
                "pool_id": self.pool_id,
                "total_threshold_ns": self.total_ns,
                "slots": self.slots,
                "reserved_slots": len(rows),
                "remaining_ns": self.total_ns - sum(r[2] for r in rows),
                "reservations": [
                    {"slot": r[0], "request_id": r[1], "allowance_ns": r[2]} for r in rows
                ],
            }
