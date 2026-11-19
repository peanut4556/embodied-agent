"""Bounded, explicit-manifest reconciliation. Never reserves or starts capture."""

import json
import sqlite3
from collections import Counter
from pathlib import Path

from .budget_start_intent import read_intent
from .wait_budget_store import WaitBudgetStore, validate_checkpoint


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
    rows, grouped, snapshots = [], {}, []
    for intent, checkpoint in prepared:
        row = {"intent": str(intent), "checkpoint": str(checkpoint) if checkpoint else None}
        rows.append(row)
        try:
            baseline = read_json(checkpoint, 8192) if checkpoint else None
            validate_checkpoint(baseline)
            identity = read_intent(intent)
            database = str(Path(identity["database"]).resolve())
            key = (database, identity["pool_id"], json.dumps(baseline, sort_keys=True))
            group = grouped.setdefault(key, {"baseline": baseline, "members": []})
            group["members"].append((row, identity["request_id"]))
        except (OSError, ValueError) as exc:
            row.update(status="unconfirmed", execution_verified=False, error=str(exc))
    for (database, pool_id, _), group in grouped.items():
        group_id = len(snapshots) + 1
        identities = list(dict.fromkeys(identity for _, identity in group["members"]))
        summary = {
            "group_id": group_id,
            "database": database,
            "pool_id": pool_id,
            "entry_count": len(group["members"]),
            "request_count": len(identities),
        }
        try:
            pool = WaitBudgetStore(
                database, expected_pool_id=pool_id, expected_checkpoint=group["baseline"]
            )
            batch = pool.lookup_many(identities)
            by_id = {r["request_id"]: r["result"] for r in batch["results"]}
            summary.update(status="queried", checkpoint=batch["checkpoint"])
            for row, identity in group["members"]:
                row.update(group_id=group_id, **by_id[identity])
        except (OSError, sqlite3.Error, ValueError) as exc:
            summary.update(status="unconfirmed", error=str(exc))
            for row, _ in group["members"]:
                row.update(
                    group_id=group_id,
                    status="unconfirmed",
                    execution_verified=False,
                    error=str(exc),
                )
        snapshots.append(summary)
    counts = dict(Counter(r["status"] for r in rows))
    return {
        "version": 1,
        "entries": rows,
        "counts": counts,
        "checked": len(rows),
        "all_allocations_found": counts.get("reserved", 0) == len(rows),
        "snapshot_groups": snapshots,
        "single_database_snapshot": len(snapshots) == 1
        and snapshots[0]["status"] == "queried"
        and not counts.get("unconfirmed"),
        "execution_verified": False,
        "exit_code": 2 if counts.get("unconfirmed") else 1 if counts.get("not_found") else 0,
    }
