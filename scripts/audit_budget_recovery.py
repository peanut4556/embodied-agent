"""Audit an explicit list of startup intents without replaying any work."""

import argparse
import json

from embodied_agent.budget_recovery_audit import audit_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    try:
        result = audit_manifest(args.manifest)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "invalid_manifest", "error": str(exc)}))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
