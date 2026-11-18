"""Explicit SQLite threshold pool; committed reservations are never refunded/reused."""

import sqlite3
import uuid
from contextlib import closing
from pathlib import Path


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

    def __init__(self, path):
        self.path = Path(path).resolve()
        with closing(self._connect()) as db:
            self.pool_id, self.total_ns, self.slots = self._config(db)

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
            if slot != i or not request or allowance != self._share(i):
                raise ValueError("invalid reservations")
        return rows

    def _share(self, slot):
        return self.total_ns // self.slots + (1 if slot <= self.total_ns % self.slots else 0)

    def reserve(self, request_id):
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 256:
            raise ValueError("bounded request identity required")
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            rows = self._rows(db)
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
