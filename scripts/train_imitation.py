"""Train or evaluate the first visual imitation baseline entirely locally."""

import argparse
import os
from pathlib import Path

from embodied_agent.imitation import evaluate, train


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "outputs/hf-cache"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "evaluate"])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--split", type=Path, default=Path("config/imitation-split.json"))
    args = parser.parse_args()
    if args.mode == "train":
        train(args.dataset, args.output, args.split)
    else:
        if args.model is None:
            parser.error("evaluate requires --model")
        evaluate(args.dataset, args.model, args.output)


if __name__ == "__main__":
    main()
