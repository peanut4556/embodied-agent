"""Persist allocation identity before submission; reconciliation never replays work."""

import json
import os
from pathlib import Path

from .recovery_json import read_recovery_json
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


def read_intent(path):
    value = read_recovery_json(path, 8192)
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "database", "pool_id", "request_id"}
        or type(value["version"]) is not int
        or value["version"] != 1
        or any(
            not isinstance(value[k], str) or not value[k]
            for k in ("database", "pool_id", "request_id")
        )
        or len(value["pool_id"]) > 256
        or len(value["request_id"]) > 256
        or not Path(value["database"]).is_absolute()
    ):
        raise ValueError("invalid startup intent")
    return value


def reconcile_intent(path, *, expected_checkpoint=None):
    value = read_intent(path)
    return WaitBudgetStore(
        value["database"],
        expected_pool_id=value["pool_id"],
        expected_checkpoint=expected_checkpoint,
    ).lookup(value["request_id"])
