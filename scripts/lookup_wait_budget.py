"""Read-only allocation reconciliation; never starts capture or reserves a slot."""

import argparse
import json
import sqlite3
from pathlib import Path

from embodied_agent.budget_start_intent import reconcile_intent
from embodied_agent.wait_budget_store import WaitBudgetStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--intent", type=Path)
    parser.add_argument("--expected-pool-id")
    parser.add_argument("--request-id")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    manual = (args.database, args.expected_pool_id, args.request_id)
    if args.intent is not None:
        if any(v is not None for v in manual):
            parser.error("--intent cannot be combined with manual identity arguments")
    elif any(v is None for v in manual):
        parser.error("provide --intent or all three manual identity arguments")
    try:
        checkpoint = json.loads(args.checkpoint.read_text()) if args.checkpoint else None
        if args.intent is not None:
            result = reconcile_intent(args.intent, expected_checkpoint=checkpoint)
        else:
            store = WaitBudgetStore(
                args.database,
                expected_pool_id=args.expected_pool_id,
                expected_checkpoint=checkpoint,
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
