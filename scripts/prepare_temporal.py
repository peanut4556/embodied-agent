"""Prepare causal windows from validated recovery recordings, entirely offline."""

import argparse
import json
import os
from pathlib import Path

from embodied_agent.temporal_data import prepare


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(Path("outputs/hf-cache").resolve()))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("config/temporal-curation-split.json"))
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()
    report = prepare(
        args.dataset, args.output, json.loads(args.split.read_text()), args.history, args.horizon
    )
    print(json.dumps({k: v for k, v in report.items() if k != "windows"}, indent=2))


if __name__ == "__main__":
    main()
