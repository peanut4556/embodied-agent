"""Bounded, read-only receipt pages; cursors are keys, not a cross-page snapshot."""

import hashlib
import os
import re

from .budget_checkpoint import canonical, digest, load_budget
from .checkpointed_runtime import checkpoint_lock
from .operation_receipt import read_receipt

MAX_ENTRIES = 4096
MAX_PAGE = 25


class InventoryLimitExceeded(ValueError):
    pass


def budget_summary(path):
    try:
        budget = load_budget(path)
    except (OSError, ValueError, TypeError) as exc:
        return {"valid": False, "cause_type": type(exc).__name__}
    return {
        "valid": True,
        "revision": budget.revision,
        "drift_policy": budget.drift_policy,
        "degraded": sorted(budget.degraded),
    }


def inspect_receipt_key(path, key):
    digest(key)
    with checkpoint_lock(path) as resolved:
        receipt = read_receipt(resolved.with_name(resolved.name + ".operations") / (key + ".json"))
        return {
            "receipt_key": key,
            "receipt": receipt,
            "receipt_sha256": hashlib.sha256(canonical(receipt).encode()).hexdigest(),
            "budget": budget_summary(resolved),
            "historical_result_available": receipt["state"] == "complete",
            "physical_outcome_verified": False,
        }


def list_operations(path, *, after=None, limit=20):
    if type(limit) is not int or not 1 <= limit <= MAX_PAGE:
        raise ValueError("page limit must be 1..25")
    if after is not None:
        digest(after)
    with checkpoint_lock(path) as resolved:
        directory = resolved.with_name(resolved.name + ".operations")
        keys, ignored, scanned = scan_receipt_keys(directory)
        eligible = [key for key in keys if after is None or key > after]
        selected = eligible[:limit]
        rows = []
        counts = {state: 0 for state in ("pending", "complete", "abandoned", "invalid")}
        for key in selected:
            row = {"receipt_key": key}
            try:
                receipt = read_receipt(directory / (key + ".json"))
            except (OSError, ValueError, TypeError) as exc:
                row.update(state="invalid", cause_type=type(exc).__name__)
            else:
                row.update(
                    state=receipt["state"],
                    request_sha256=receipt["request_sha256"],
                    receipt_sha256=hashlib.sha256(canonical(receipt).encode()).hexdigest(),
                )
                if receipt["state"] == "complete":
                    row.update(
                        success=receipt["result"]["success"],
                        completed_steps=receipt["result"]["completed_steps"],
                    )
            counts[row["state"]] += 1
            rows.append(row)
        budget = budget_summary(resolved)
        return {
            "items": rows,
            "page_counts": counts,
            "budget": budget,
            "has_errors": counts["invalid"] > 0 or not budget["valid"],
            "matched_entries": len(keys),
            "scanned_entries": scanned,
            "ignored_entries": ignored,
            "has_more": len(eligible) > limit,
            "next_cursor": selected[-1] if len(eligible) > limit else None,
            "physical_outcome_verified": False,
        }


def scan_receipt_keys(directory):
    keys, ignored, scanned = [], 0, 0
    try:
        entries = os.scandir(directory)
    except FileNotFoundError:
        entries = None
    if entries is not None:
        with entries:
            for entry in entries:
                scanned += 1
                if scanned > MAX_ENTRIES:
                    raise InventoryLimitExceeded(
                        "receipt directory exceeds scan limit; no partial inventory returned"
                    )
                if re.fullmatch("[0-9a-f]{64}\\.json", entry.name) and entry.is_file(
                    follow_symlinks=False
                ):
                    keys.append(entry.name[:-5])
                else:
                    ignored += 1
    keys.sort()
    return keys, ignored, scanned
