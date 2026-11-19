"""Bounded, explicit-manifest reconciliation. Never reserves or starts capture."""

import json
import sqlite3
from collections import Counter
from pathlib import Path

from .budget_start_intent import reconcile_intent


def read_json(path, limit):
    with Path(path).open(encoding="utf-8") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("audit input exceeds size limit")
    return json.loads(raw)


def audit_manifest(path):
    path = Path(path).resolve()
    manifest = read_json(path, 65536)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"version", "entries"}
        or type(manifest["version"]) is not int
        or manifest["version"] != 1
        or not isinstance(manifest["entries"], list)
        or not 1 <= len(manifest["entries"]) <= 64
    ):
        raise ValueError("manifest requires version 1 and 1..64 explicit entries")
    prepared, seen = [], set()
    # Validate the whole manifest before reading any intent or database.
    for entry in manifest["entries"]:
        if (
            not isinstance(entry, dict)
            or "intent" not in entry
            or set(entry) - {"intent", "checkpoint"}
            or any(not isinstance(v, str) or not 1 <= len(v) <= 4096 for v in entry.values())
        ):
            raise ValueError("invalid audit entry")
        intent = (path.parent / entry["intent"]).resolve()
        if intent in seen:
            raise ValueError("duplicate intent path")
        seen.add(intent)
        checkpoint = (
            (path.parent / entry["checkpoint"]).resolve() if "checkpoint" in entry else None
        )
        prepared.append((intent, checkpoint))
    rows = []
    for intent, checkpoint in prepared:
        try:
            baseline = read_json(checkpoint, 8192) if checkpoint else None
            result = reconcile_intent(intent, expected_checkpoint=baseline)
        except (OSError, sqlite3.Error, ValueError) as exc:
            result = {"status": "unconfirmed", "execution_verified": False, "error": str(exc)}
        rows.append(
            {"intent": str(intent), "checkpoint": str(checkpoint) if checkpoint else None, **result}
        )
    counts = dict(Counter(r["status"] for r in rows))
    return {
        "version": 1,
        "entries": rows,
        "counts": counts,
        "checked": len(rows),
        "all_allocations_found": counts.get("reserved", 0) == len(rows),
        "single_database_snapshot": False,
        "execution_verified": False,
        "exit_code": 2 if counts.get("unconfirmed") else 1 if counts.get("not_found") else 0,
    }
