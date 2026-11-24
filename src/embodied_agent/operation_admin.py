"""Inspect or explicitly close a pending receipt; never authorizes motion or replay."""

import argparse
import hashlib
import json
import sys

from .budget_checkpoint import canonical, digest, label, load_budget
from .checkpointed_runtime import checkpoint_lock
from .operation_inventory import inspect_receipt_key, list_operations
from .operation_receipt import _write, read_receipt, receipt_path
from .operation_snapshot import export_snapshot, verify_snapshot


def inspect_operation(path, operation_id):
    label(operation_id)
    with checkpoint_lock(path) as resolved:
        receipt = read_receipt(receipt_path(resolved, operation_id))
        report = {
            "operation_id": operation_id,
            "receipt": receipt,
            "receipt_sha256": hashlib.sha256(canonical(receipt).encode()).hexdigest(),
        }
        try:
            budget = load_budget(resolved)
        except (OSError, ValueError, TypeError) as exc:
            report["budget"] = {"valid": False, "cause_type": type(exc).__name__}
        else:
            report["budget"] = {
                "valid": True,
                "revision": budget.revision,
                "drift_policy": budget.drift_policy,
                "degraded": sorted(budget.degraded),
            }
        report["historical_result_available"] = receipt["state"] == "complete"
        report["physical_outcome_verified"] = False
        return report


def abandon_operation(path, operation_id, *, expected_receipt_sha256, reason, evidence_ref, actor):
    label(operation_id)
    for value in (reason, evidence_ref, actor):
        label(value)
    digest(expected_receipt_sha256)
    with checkpoint_lock(path) as resolved:
        file = receipt_path(resolved, operation_id)
        receipt = read_receipt(file)
        checksum = hashlib.sha256(canonical(receipt).encode()).hexdigest()
        if checksum != expected_receipt_sha256:
            raise ValueError("receipt changed since inspection")
        if receipt["state"] != "pending":
            raise ValueError("only pending operations may be abandoned")
        closed = {
            "schema": 2,
            "request_sha256": receipt["request_sha256"],
            "state": "abandoned",
            "result": None,
            "resolution": {
                "reason": reason,
                "evidence_ref": evidence_ref,
                "actor": actor,
                "previous_receipt_sha256": checksum,
            },
        }
        _write(file, closed)
        return closed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "abandon"):
        child = sub.add_parser(command)
        child.add_argument("--checkpoint", required=True)
        child.add_argument("--operation-id", required=True)
        if command == "abandon":
            for name in ("expected-receipt-sha256", "reason", "evidence-ref", "actor"):
                child.add_argument("--" + name, required=True)
    inventory = sub.add_parser("list")
    inventory.add_argument("--checkpoint", required=True)
    inventory.add_argument("--limit", type=int, default=20)
    inventory.add_argument("--after")
    detail = sub.add_parser("inspect-key")
    detail.add_argument("--checkpoint", required=True)
    detail.add_argument("--receipt-key", required=True)
    export = sub.add_parser("snapshot")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    verify = sub.add_parser("verify-snapshot")
    verify.add_argument("--snapshot", required=True)
    args = parser.parse_args()
    try:
        if args.command == "snapshot":
            result = export_snapshot(args.checkpoint, args.output)
        elif args.command == "verify-snapshot":
            result = verify_snapshot(args.snapshot)
        elif args.command == "list":
            result = list_operations(args.checkpoint, after=args.after, limit=args.limit)
        elif args.command == "inspect-key":
            result = inspect_receipt_key(args.checkpoint, args.receipt_key)
        elif args.command == "inspect":
            result = inspect_operation(args.checkpoint, args.operation_id)
        else:
            result = abandon_operation(
                args.checkpoint,
                args.operation_id,
                expected_receipt_sha256=args.expected_receipt_sha256,
                reason=args.reason,
                evidence_ref=args.evidence_ref,
                actor=args.actor,
            )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(json.dumps({"ok": False, "cause_type": type(exc).__name__}), file=sys.stderr)
        return 3
    print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
    return (
        1 if args.command in ("list", "snapshot", "verify-snapshot") and result["has_errors"] else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
