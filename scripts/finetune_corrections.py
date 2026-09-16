"""Fine-tune the frozen learner on validated expert corrections."""

import argparse
import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", "outputs/hf-cache")

from embodied_agent.correction_finetune import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="outputs/datasets/reactive-development-v2")
    parser.add_argument(
        "--corrections",
        nargs="+",
        default=["outputs/datasets/corrections-v2", "outputs/datasets/corrections-expanded-v1"],
    )
    parser.add_argument("--pretrained", default="outputs/models/memory-bc-v2/epoch-2000")
    parser.add_argument("--experiment", default="config/correction-finetune-experiment.json")
    parser.add_argument("--output", default="outputs/models/memory-correction-v1")
    args = parser.parse_args()
    print(
        json.dumps(
            train(args.base, args.corrections, args.pretrained, args.experiment, args.output),
            indent=2,
        )
    )
