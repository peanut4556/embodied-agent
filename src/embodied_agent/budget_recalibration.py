"""Explicit preview/apply of validated recalibration under the execution lock."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .action_budget import seconds
from .budget_checkpoint import LIMIT, digest, fields, label, load_budget, restore, save_budget
from .checkpointed_runtime import checkpoint_lock
from .recovery_json import parse_recovery_json


class StaleRecalibration(ValueError):
    pass


def _read(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(LIMIT + 1)
    if len(raw) > LIMIT:
        raise ValueError("recalibration input exceeds size limit")
    return raw


def _request(request_path):
    raw = _read(request_path)
    return raw, _parse_request(raw)


def _parse_request(raw):
    request = parse_recovery_json(raw.decode("utf-8"))
    fields(request, "schema action calibration validation")
    if type(request["schema"]) is not int or request["schema"] != 1:
        raise ValueError("unsupported recalibration request")
    label(request["action"])
    for key in ("calibration", "validation"):
        fields(request[key], "batch_id durations_s")
        label(request[key]["batch_id"])
        values = request[key]["durations_s"]
        if not isinstance(values, list) or len(values) > 4096:
            raise ValueError("sample list required, maximum 4096 per batch")
    for key in ("calibration", "validation"):
        values = request[key]["durations_s"]
        if len(values) < 2:
            raise ValueError("at least two samples required")
        for value in values:
            seconds(value)
    if request["calibration"]["batch_id"] == request["validation"]["batch_id"]:
        raise ValueError("distinct batches required")
    return request


def inspect_recalibration(checkpoint, request_path):
    """Reconcile request content with current history, without retrying a write."""
    with checkpoint_lock(checkpoint) as resolved:
        budget = load_budget(resolved)
        raw, request = _request(request_path)
        return _inspect_request(budget, raw, request)


def _inspect_request(budget, raw, request):
    calibration, validation = request["calibration"], request["validation"]
    expected = {
        "action": request["action"],
        "batch_id": calibration["batch_id"],
        "validation_batch_id": validation["batch_id"],
        "sample_count": len(calibration["durations_s"]),
        "validation_sample_count": len(validation["durations_s"]),
    }
    for key, batch in (("samples_sha256", calibration), ("validation_sha256", validation)):
        expected[key] = hashlib.sha256(
            json.dumps(batch["durations_s"], separators=(",", ":")).encode()
        ).hexdigest()
    history = budget.recalibration_history()
    matching = [row for row in history if all(row.get(k) == v for k, v in expected.items())]
    ids = {calibration["batch_id"], validation["batch_id"]}
    collisions = [
        row["revision"]
        for row in history
        if ids.intersection((row["batch_id"], row.get("validation_batch_id")))
    ]
    action = request["action"]
    matched = matching[0] if matching else None
    later = [
        row["revision"]
        for row in history
        if row["action"] == action and matched and row["revision"] > matched["revision"]
    ]
    if matched:
        status = (
            "superseded"
            if later
            else ("applied_then_degraded" if action in budget.degraded else "applied_current")
        )
    else:
        status = "batch_conflict" if collisions else "not_found"
    return {
        "status": status,
        "evidence_scope": "current_checkpoint_history",
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "action": action,
        "current_revision": budget.revision,
        "matched_revision": matched["revision"] if matched else None,
        "matched_record": matched,
        "later_action_revisions": later,
        "conflicting_revisions": collisions if not matched else [],
        "currently_degraded": action in budget.degraded,
        "currently_blocked": budget.blocked(action),
        "current_required_s": budget.required(action),
        "drift_policy": budget.drift_policy,
    }


def _validated_candidate(budget, request):
    candidate = budget.validated_recalibrated(
        request["action"],
        request["calibration"]["durations_s"],
        batch_id=request["calibration"]["batch_id"],
        validation_samples=request["validation"]["durations_s"],
        validation_batch_id=request["validation"]["batch_id"],
    )
    # Apply the same structural limits as persisted checkpoints, without writing.
    restore(
        {
            "schema": 1,
            "fallback_s": candidate.fallback_s,
            "margin_s": candidate.margin_s,
            "minimum_samples": candidate.minimum_samples,
            "drift_policy": candidate.drift_policy,
            "estimates": candidate.estimates,
            "degraded": sorted(candidate.degraded),
            "revision": candidate.revision,
            "history": candidate.recalibration_history(),
        }
    )
    return candidate


def _candidate(checkpoint, request_path):
    original = _read(checkpoint)
    budget = load_budget(checkpoint)
    if budget.drift_policy != "quarantine":
        raise ValueError("maintenance requires quarantine policy")
    raw, request = _request(request_path)
    candidate = _validated_candidate(budget, request)
    summary = {
        "checkpoint_sha256": hashlib.sha256(original).hexdigest(),
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "action": request["action"],
        "before_revision": budget.revision,
        "after_revision": candidate.revision,
        "before_required_s": budget.required(request["action"]),
        "after_required_s": candidate.required(request["action"]),
        "before_degraded": sorted(budget.degraded),
        "after_degraded": sorted(candidate.degraded),
        "calibration": candidate.recalibration_history()[-1],
        "applied": False,
    }
    return candidate, summary


def preview_recalibration(checkpoint, request_path):
    with checkpoint_lock(checkpoint) as resolved:
        _, summary = _candidate(resolved, request_path)
        return summary


def apply_recalibration(
    checkpoint, request_path, *, expected_checkpoint_sha256, expected_request_sha256
):
    digest(expected_checkpoint_sha256)
    digest(expected_request_sha256)
    with checkpoint_lock(checkpoint) as resolved:
        # Check inspected bytes before allowing any candidate restoration.
        if hashlib.sha256(_read(resolved)).hexdigest() != expected_checkpoint_sha256:
            raise StaleRecalibration("checkpoint changed since preview")
        candidate, summary = _candidate(resolved, request_path)
        if (
            summary["checkpoint_sha256"] != expected_checkpoint_sha256
            or summary["request_sha256"] != expected_request_sha256
        ):
            raise StaleRecalibration("recalibration inputs changed since preview")
        save_budget(candidate, resolved)
        summary["applied"] = True
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preview", "apply", "status"):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--request", required=True)
        if name == "apply":
            command.add_argument("--expected-checkpoint-sha256", required=True)
            command.add_argument("--expected-request-sha256", required=True)
    args = parser.parse_args()
    try:
        if args.command == "status":
            result = inspect_recalibration(args.checkpoint, args.request)
        elif args.command == "preview":
            result = preview_recalibration(args.checkpoint, args.request)
        else:
            result = apply_recalibration(
                args.checkpoint,
                args.request,
                expected_checkpoint_sha256=args.expected_checkpoint_sha256,
                expected_request_sha256=args.expected_request_sha256,
            )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(json.dumps({"ok": False, "cause_type": type(exc).__name__}), file=sys.stderr)
        return 3
    print(json.dumps({"ok": True, "result": result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
