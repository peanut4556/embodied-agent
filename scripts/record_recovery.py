"""Collect or revalidate outcome-labelled local recovery data; never uploads."""

import argparse
import json
import os
from pathlib import Path

from embodied_agent.recovery_data import record_recovery, validate_recovery


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "outputs/hf-cache"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("outputs/models/context-bc-v1"))
    parser.add_argument("--scenarios", type=Path, default=Path("config/slip-scenarios.json"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_recovery(args.output)
    else:
        # These scenarios have already been inspected; never label this batch as held-out test data.
        cases = json.loads(args.scenarios.read_text())["test"]
        record_recovery(args.output, args.model, cases)


if __name__ == "__main__":
    main()
