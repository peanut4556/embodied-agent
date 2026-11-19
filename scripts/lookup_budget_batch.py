"""Reconcile multiple request identities from one pool snapshot."""

import argparse
import json
import sqlite3

from embodied_agent.budget_recovery_audit import read_json
from embodied_agent.wait_budget_store import WaitBudgetStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--expected-pool-id", required=True)
    parser.add_argument("--requests", required=True, help="JSON array of 1..64 unique request ids")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    try:
        ids = read_json(args.requests, 65536)
        baseline = read_json(args.checkpoint, 8192) if args.checkpoint else None
        pool = WaitBudgetStore(
            args.database, expected_pool_id=args.expected_pool_id, expected_checkpoint=baseline
        )
        result = pool.lookup_many(ids)
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(json.dumps({"status": "unconfirmed", "error": str(exc), "execution_verified": False}))
        return 2
    print(json.dumps(result, indent=2))
    return 0 if all(r["result"]["status"] == "reserved" for r in result["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
