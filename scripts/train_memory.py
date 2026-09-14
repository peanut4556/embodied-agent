"""Train a recurrent policy, select on development physics, then test the frozen winner."""

import argparse
import json
import os
from pathlib import Path


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(Path("outputs/hf-cache").resolve()))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "select", "evaluate"))
    parser.add_argument(
        "--dataset", type=Path, default=Path("outputs/datasets/reactive-development-v2")
    )
    parser.add_argument(
        "--experiment", type=Path, default=Path("config/memory-extended-experiment.json")
    )
    parser.add_argument("--run", type=Path, default=Path("outputs/models/memory-bc-v2"))
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("outputs/evaluations/memory-bc-v2-development/selection.json"),
    )
    parser.add_argument("--baseline", type=Path, default=Path("outputs/models/context-bc-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "train":
        from embodied_agent.memory_policy import train

        result = train(args.dataset, args.experiment, args.output)
    elif args.mode == "select":
        from embodied_agent.memory_evaluation import select

        result = select(args.run, args.output)
    else:
        from embodied_agent.memory_evaluation import evaluate

        result = evaluate(args.run, args.selection, args.baseline, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
