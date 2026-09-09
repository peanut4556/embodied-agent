"""Evaluate visual and contact feedback around the frozen imitation baseline."""

import argparse
from pathlib import Path

from embodied_agent.feedback_evaluation import evaluate_feedback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("outputs/models/context-bc-v1"))
    parser.add_argument("--scenarios", type=Path, default=Path("config/feedback-scenarios.json"))
    parser.add_argument("--group", choices=["development", "test"], default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate_feedback(args.model, args.scenarios, args.output, args.group)


if __name__ == "__main__":
    main()
