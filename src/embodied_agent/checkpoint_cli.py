"""Cross-process checkpoint proposals and explicit reviewed publication."""

import argparse
import json
import sqlite3
from pathlib import Path

from embodied_agent.checkpoint_store import (
    CheckpointStore,
    _separate,
    _validate_proposal,
    preview_publication,
    reconcile_publication,
)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _read_proposal(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("proposal file exceeds 64 KiB")
    document = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(document, dict) or document.get("status") != "proposed":
        raise ValueError("successful preview document required")
    if document.get("robot_action_authorized") is not False:
        raise ValueError("proposal cannot authorize robot actions")
    proposal = document.get("proposal")
    _validate_proposal(proposal)
    return proposal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("preview", "publish", "reconcile"))
    parser.add_argument("--database", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--proposal", help="saved preview JSON; required for publish")
    parser.add_argument("--max-records", type=int, default=10000)
    args = parser.parse_args(argv)
    if (args.operation != "preview") != (args.proposal is not None):
        parser.error("--proposal is required for publish and reconcile")
    try:
        # Validate all caller inputs before opening a writable store connection.
        _separate(args.store, args.database)
        if not 1 <= args.max_records <= 100000:
            raise ValueError("max_records must be 1..100000")
        if args.operation == "preview":
            result = preview_publication(args.database, args.store, max_records=args.max_records)
        else:
            proposal = _read_proposal(args.proposal)
            if args.operation == "reconcile":
                result = reconcile_publication(args.store, proposal)
            else:
                with CheckpointStore(args.store) as store:
                    result = store.publish_reviewed(
                        args.database, proposal, max_records=args.max_records
                    )
    except (ValueError, OSError, sqlite3.Error, RecursionError) as exc:
        result = {
            "status": "request_failed",
            "error": str(exc)[:512],
            "robot_action_authorized": False,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    status = result["status"]
    if status in ("proposed", "published", "publication_recorded"):
        return 0
    if status == "no_publication_record":
        return 8
    if status == "publication_unconfirmed":
        return 7
    if (
        status == "generation_conflict"
        or result.get("source", {}).get("status") == "proposal_changed"
    ):
        return 6
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
