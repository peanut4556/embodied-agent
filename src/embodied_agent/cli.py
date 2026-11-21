from __future__ import annotations

import argparse
import logging

from .adapters.rai_planner import RAISubprocessPlanner
from .mock import MockPolicy, MockRobot, TimestampedMockRobot
from .planner import RuleBasedPlanner
from .runtime import AgentRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the embodied-agent loop")
    parser.add_argument("instruction", nargs="?", default="把桌上的红色积木放进盒子")
    parser.add_argument(
        "--planner",
        choices=("rule", "rai"),
        default="rule",
        help="planning backend (default: rule)",
    )
    parser.add_argument(
        "--confirmation-timeout",
        type=float,
        default=0,
        help="seconds to poll action outcomes (0..5; default immediate)",
    )
    parser.add_argument(
        "--max-observation-age",
        type=float,
        default=None,
        help="require timestamped mock observations of at most this age (0..5 seconds)",
    )
    parser.add_argument("--confirmation-samples", type=int, default=1)
    parser.add_argument("--confirmation-span", type=float, default=0)
    parser.add_argument("--max-confirmation-gap", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    planner = RAISubprocessPlanner() if args.planner == "rai" else RuleBasedPlanner()
    runtime = AgentRuntime(
        planner,
        MockPolicy(),
        TimestampedMockRobot() if args.max_observation_age is not None else MockRobot(),
        confirmation_timeout_s=args.confirmation_timeout,
        max_observation_age_s=args.max_observation_age,
        confirmation_samples=args.confirmation_samples,
        confirmation_span_s=args.confirmation_span,
        max_confirmation_gap_s=args.max_confirmation_gap,
    )
    result = runtime.run(args.instruction)
    print(
        f"success={result.success} completed_steps={result.completed_steps} "
        f"message={result.message}"
    )
