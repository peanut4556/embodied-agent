from __future__ import annotations

import argparse
import logging

from .mock import MockPolicy, MockRobot
from .planner import RuleBasedPlanner
from .runtime import AgentRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the embodied-agent mock loop")
    parser.add_argument("instruction", nargs="?", default="把桌上的红色积木放进盒子")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    runtime = AgentRuntime(RuleBasedPlanner(), MockPolicy(), MockRobot())
    result = runtime.run(args.instruction)
    print(
        f"success={result.success} completed_steps={result.completed_steps} "
        f"message={result.message}"
    )
