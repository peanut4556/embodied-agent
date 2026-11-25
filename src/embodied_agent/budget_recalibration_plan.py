"""Validate all actions before publishing one recalibration checkpoint."""

import argparse
import hashlib
import json
import sys

from .budget_checkpoint import canonical, digest, fields, load_budget, save_budget
from .budget_recalibration import (
    StaleRecalibration,
    _inspect_request,
    _parse_request,
    _read,
    _validated_candidate,
)
from .checkpointed_runtime import checkpoint_lock
from .recovery_json import parse_recovery_json


def _plan(path):
    raw = _read(path)
    plan = parse_recovery_json(raw.decode("utf-8"))
    fields(plan, "schema requests")
    if type(plan["schema"]) is not int or plan["schema"] != 1:
        raise ValueError("unsupported recalibration plan schema")
    requests = plan["requests"]
    if not isinstance(requests, list) or not 1 <= len(requests) <= 16:
        raise ValueError("plan requires one to sixteen requests")
    actions, batches = set(), set()
    for request in requests:
        request = _parse_request(canonical(request).encode())
        if request["action"] in actions:
            raise ValueError("duplicate plan action")
        actions.add(request["action"])
        for role in ("calibration", "validation"):
            batch = request[role]["batch_id"]
            if batch in batches:
                raise ValueError("batch reused across plan actions")
            batches.add(batch)
    return raw, requests


def _candidate(checkpoint, plan_path):
    original = _read(checkpoint)
    budget = load_budget(checkpoint)
    if budget.drift_policy != "quarantine":
        raise ValueError("maintenance requires quarantine policy")
    raw, requests = _plan(plan_path)
    candidate = budget
    changes = []
    for request in requests:
        previous = candidate
        candidate = _validated_candidate(previous, request)
        action = request["action"]
        changes.append(
            {
                "action": action,
                "before_required_s": previous.required(action),
                "after_required_s": candidate.required(action),
                "calibration": candidate.recalibration_history()[-1],
            }
        )
    return candidate, {
        "checkpoint_sha256": hashlib.sha256(original).hexdigest(),
        "plan_sha256": hashlib.sha256(raw).hexdigest(),
        "before_revision": budget.revision,
        "after_revision": candidate.revision,
        "before_degraded": sorted(budget.degraded),
        "after_degraded": sorted(candidate.degraded),
        "changes": changes,
        "applied": False,
    }


def inspect_plan(checkpoint, plan_path):
    """Read all request outcomes from one locked checkpoint, without publishing."""
    with checkpoint_lock(checkpoint) as resolved:
        budget = load_budget(resolved)
        raw, requests = _plan(plan_path)
        results = [
            _inspect_request(budget, canonical(request).encode(), request) for request in requests
        ]
        # The standalone status hashes a file; nested entries have no such file.
        for result in results:
            result.pop("request_sha256")
        matched = sum(result["matched_revision"] is not None for result in results)
        conflicts = sum(result["status"] == "batch_conflict" for result in results)
        counts = {
            name: sum(result["status"] == name for result in results)
            for name in (
                "applied_current",
                "applied_then_degraded",
                "superseded",
                "batch_conflict",
                "not_found",
            )
        }
        status = (
            "conflict"
            if conflicts
            else "all_matched"
            if matched == len(results)
            else "partially_matched"
            if matched
            else "not_found"
        )
        return {
            "status": status,
            "evidence_scope": "current_checkpoint_history",
            "atomic_application_verified": False,
            "plan_sha256": hashlib.sha256(raw).hexdigest(),
            "current_revision": budget.revision,
            "matched_count": matched,
            "request_count": len(results),
            "counts": counts,
            "currently_blocked_actions": [r["action"] for r in results if r["currently_blocked"]],
            "results": results,
        }


def preview_plan(checkpoint, plan_path):
    with checkpoint_lock(checkpoint) as resolved:
        _, result = _candidate(resolved, plan_path)
        return result


def apply_plan(checkpoint, plan_path, *, expected_checkpoint_sha256, expected_plan_sha256):
    digest(expected_checkpoint_sha256)
    digest(expected_plan_sha256)
    with checkpoint_lock(checkpoint) as resolved:
        if hashlib.sha256(_read(resolved)).hexdigest() != expected_checkpoint_sha256:
            raise StaleRecalibration("checkpoint changed since plan preview")
        candidate, result = _candidate(resolved, plan_path)
        if (
            result["checkpoint_sha256"] != expected_checkpoint_sha256
            or result["plan_sha256"] != expected_plan_sha256
        ):
            raise StaleRecalibration("plan inputs changed since preview")
        save_budget(candidate, resolved)
        result["applied"] = True
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preview", "apply", "status"):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--plan", required=True)
        if name == "apply":
            command.add_argument("--expected-checkpoint-sha256", required=True)
            command.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args()
    try:
        if args.command == "status":
            result = inspect_plan(args.checkpoint, args.plan)
        elif args.command == "preview":
            result = preview_plan(args.checkpoint, args.plan)
        else:
            result = apply_plan(
                args.checkpoint,
                args.plan,
                expected_checkpoint_sha256=args.expected_checkpoint_sha256,
                expected_plan_sha256=args.expected_plan_sha256,
            )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(json.dumps({"ok": False, "cause_type": type(exc).__name__}), file=sys.stderr)
        return 3
    print(json.dumps({"ok": True, "result": result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
