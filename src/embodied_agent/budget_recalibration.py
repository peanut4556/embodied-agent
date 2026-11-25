"""Explicit preview/apply of validated recalibration under the execution lock."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

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


def _candidate(checkpoint, request_path):
    original = _read(checkpoint)
    budget = load_budget(checkpoint)
    if budget.drift_policy != "quarantine":
        raise ValueError("maintenance requires quarantine policy")
    raw = _read(request_path)
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
    for name in ("preview", "apply"):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--request", required=True)
        if name == "apply":
            command.add_argument("--expected-checkpoint-sha256", required=True)
            command.add_argument("--expected-request-sha256", required=True)
    args = parser.parse_args()
    try:
        if args.command == "preview":
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
