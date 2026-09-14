"""Train or physically evaluate the local continuous-feedback baseline."""

import argparse
import json
import os
from pathlib import Path


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(Path("outputs/hf-cache").resolve()))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train", "evaluate"))
    parser.add_argument(
        "--dataset", type=Path, default=Path("outputs/datasets/reactive-development-v2")
    )
    parser.add_argument(
        "--index", type=Path, default=Path("outputs/datasets/reactive-development-v2-temporal.json")
    )
    parser.add_argument("--experiment", type=Path, default=Path("config/reactive-experiment.json"))
    parser.add_argument("--model", type=Path, default=Path("outputs/models/reactive-bc-v1"))
    parser.add_argument("--baseline", type=Path, default=Path("outputs/models/context-bc-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        from embodied_agent.temporal_data import prepare

        result = prepare(
            args.dataset, args.output, json.loads(args.experiment.read_text())["split"]
        )
        result = {key: value for key, value in result.items() if key != "windows"}
    elif args.mode == "train":
        from embodied_agent.reactive import train

        result = train(args.index, args.experiment, args.output)
    else:
        from embodied_agent.reactive_evaluation import evaluate

        result = evaluate(args.model, args.baseline, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
