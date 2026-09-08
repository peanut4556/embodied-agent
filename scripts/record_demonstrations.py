"""Record and validate a local LeRobot dataset, without uploading it."""

import argparse
import os
from pathlib import Path

from embodied_agent.demonstrations import record_dataset, validate_dataset


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "outputs/hf-cache"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positions", type=float, nargs="+", default=[0.28, 0.32, 0.40])
    parser.add_argument("--fps", type=int, choices=[25, 50], default=25)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_dataset(args.output)
    else:
        record_dataset(args.output, args.positions, args.fps)


if __name__ == "__main__":
    main()
