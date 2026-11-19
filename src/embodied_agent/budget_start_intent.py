"""Persist allocation identity before submission; reconciliation never replays work."""

import json
import os
from pathlib import Path

from .wait_budget_store import WaitBudgetStore


def save_intent(path, store, request_id):
    record = {
        "version": 1,
        "database": str(store.path),
        "pool_id": store.pool_id,
        "request_id": request_id,
    }
    # Exclusive create: a partial failed write is retained and never overwritten.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(record, stream)
        stream.flush()
        os.fsync(stream.fileno())
    # Sync the directory entry before permitting allocation.
    descriptor = os.open(Path(path).resolve().parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def reconcile_intent(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = stream.read(8193)
    if len(raw) > 8192:
        raise ValueError("intent too large")
    value = json.loads(raw)
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "database", "pool_id", "request_id"}
        or type(value["version"]) is not int
        or value["version"] != 1
        or any(
            not isinstance(value[k], str) or not value[k]
            for k in ("database", "pool_id", "request_id")
        )
        or not Path(value["database"]).is_absolute()
    ):
        raise ValueError("invalid startup intent")
    return WaitBudgetStore(value["database"], expected_pool_id=value["pool_id"]).lookup(
        value["request_id"]
    )
