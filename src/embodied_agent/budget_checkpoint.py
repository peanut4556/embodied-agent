"""Strict local budget snapshots; checksums detect corruption, not malicious edits."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from .action_budget import ActionBudget, seconds
from .recovery_json import read_recovery_json

LIMIT = 1_048_576


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def fields(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys.split()):
        raise ValueError("invalid checkpoint fields")


def label(value):
    if not isinstance(value, str) or not value or value.strip() != value or len(value) > 128:
        raise ValueError("invalid checkpoint label")


def integer(value, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid checkpoint integer")


def digest(value):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("invalid checkpoint digest")


def restore(state):
    fields(
        state,
        "schema fallback_s margin_s minimum_samples drift_policy estimates degraded revision history",
    )
    if type(state["schema"]) is not int or state["schema"] != 1:
        raise ValueError("unsupported budget checkpoint schema")
    budget = ActionBudget(
        {},
        fallback_s=state["fallback_s"],
        margin_s=state["margin_s"],
        minimum_samples=state["minimum_samples"],
        drift_policy=state["drift_policy"],
    )
    integer(state["revision"], 0)
    estimates, degraded, history = state["estimates"], state["degraded"], state["history"]
    if (
        not isinstance(estimates, dict)
        or not isinstance(degraded, list)
        or not isinstance(history, list)
    ):
        raise ValueError("invalid checkpoint collections")  # noqa: TRY004 - malformed document
    if len(estimates) > 128 or len(degraded) > 128 or len(history) > 1000:
        raise ValueError("checkpoint collection limit exceeded")
    for name, value in estimates.items():
        label(name)
        seconds(value)
    for name in degraded:
        label(name)
    if len(set(degraded)) != len(degraded) or state["revision"] != len(history):
        raise ValueError("inconsistent checkpoint state")
    batches, latest = set(), {}
    basic = "revision action batch_id sample_count previous_required_s required_s samples_sha256"
    validation = "validation_batch_id validation_sample_count validation_max_s validation_sha256"
    for index, row in enumerate(history, 1):
        fields(
            row,
            basic
            + (" " + validation if isinstance(row, dict) and "validation_batch_id" in row else ""),
        )
        integer(row["revision"], 1)
        if row["revision"] != index:
            raise ValueError("nonsequential calibration history")
        label(row["action"])
        integer(row["sample_count"], budget.minimum_samples)
        seconds(row["previous_required_s"])
        seconds(row["required_s"])
        digest(row["samples_sha256"])
        ids = [row["batch_id"]]
        if "validation_batch_id" in row:
            ids.append(row["validation_batch_id"])
            integer(row["validation_sample_count"], budget.minimum_samples)
            seconds(row["validation_max_s"])
            digest(row["validation_sha256"])
            if row["validation_max_s"] > row["required_s"]:
                raise ValueError("validation exceeds restored budget")
        for batch in ids:
            label(batch)
            if batch in batches:
                raise ValueError("reused calibration batch")
            batches.add(batch)
        latest[row["action"]] = row["required_s"]
    if any(estimates.get(name) != value for name, value in latest.items()):
        raise ValueError("estimates disagree with calibration history")
    budget.estimates = dict(estimates)
    budget.degraded = set(degraded)
    budget.revision = state["revision"]
    # JSON roundtrip detaches the audit trail from the input object.
    budget._recalibrations = json.loads(canonical(history))
    return budget


def save_budget(budget, path):
    state = {
        "schema": 1,
        "fallback_s": budget.fallback_s,
        "margin_s": budget.margin_s,
        "minimum_samples": budget.minimum_samples,
        "drift_policy": budget.drift_policy,
        "estimates": dict(budget.estimates),
        "degraded": sorted(budget.degraded),
        "revision": budget.revision,
        "history": budget.recalibration_history(),
    }
    restore(state)
    envelope = {"state": state, "sha256": hashlib.sha256(canonical(state).encode()).hexdigest()}
    content = canonical(envelope) + "\n"
    if len(content) > LIMIT:
        raise ValueError("budget checkpoint exceeds size limit")
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".budget-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_budget(path):
    envelope = read_recovery_json(path, LIMIT)
    fields(envelope, "state sha256")
    digest(envelope["sha256"])
    if hashlib.sha256(canonical(envelope["state"]).encode()).hexdigest() != envelope["sha256"]:
        raise ValueError("budget checkpoint checksum mismatch")
    return restore(envelope["state"])


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Read-only validation of an action budget checkpoint"
    )
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        budget = load_budget(args.path)
    except (ValueError, OSError, TypeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "valid": True,
                "revision": budget.revision,
                "drift_policy": budget.drift_policy,
                "degraded": sorted(budget.degraded),
                "required_s": {
                    name: budget.required(name)
                    for name in sorted(set(budget.estimates) | budget.degraded)
                },
                "history_entries": len(budget.recalibration_history()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
