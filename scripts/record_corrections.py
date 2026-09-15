"""Record or replay actual learner-to-expert correction episodes, entirely locally."""

import argparse
import json
import os
from pathlib import Path


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(Path("outputs/hf-cache").resolve()))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, default=Path("outputs/models/memory-bc-v2/epoch-2000")
    )
    parser.add_argument("--scenarios", type=Path, default=Path("config/correction-scenarios.json"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    from embodied_agent.correction_data import record, validate

    report = (
        validate(args.output)
        if args.validate_only
        else record(args.output, args.model, args.scenarios)
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
