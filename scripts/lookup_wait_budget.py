"""Read-only allocation reconciliation; never starts capture or reserves a slot."""

import argparse
import json
import sqlite3
from pathlib import Path

from embodied_agent.wait_budget_store import WaitBudgetStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--expected-pool-id", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    try:
        checkpoint = json.loads(args.checkpoint.read_text()) if args.checkpoint else None
        store = WaitBudgetStore(
            args.database, expected_pool_id=args.expected_pool_id, expected_checkpoint=checkpoint
        )
        result = store.lookup(args.request_id)
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(
            json.dumps(
                {"status": "unconfirmed", "error": str(exc), "execution_verified": False},
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "reserved" else 1


if __name__ == "__main__":
    raise SystemExit(main())
