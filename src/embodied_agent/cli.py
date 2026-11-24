from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict

from .adapters.rai_planner import RAISubprocessPlanner
from .checkpointed_runtime import run_checkpointed
from .mock import MockPolicy, MockRobot, TimestampedMockRobot
from .planner import RuleBasedPlanner
from .runtime import AgentRuntime


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite positive seconds")
    return number


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
    parser.add_argument(
        "--task-timeout",
        type=positive_seconds,
        default=None,
        help="total task budget in seconds, including planning (default: unlimited)",
    )
    parser.add_argument(
        "--budget-checkpoint",
        help="existing quarantine budget checkpoint (requires --task-timeout)",
    )
    parser.add_argument("--confirmation-samples", type=int, default=1)
    parser.add_argument("--confirmation-span", type=float, default=0)
    parser.add_argument("--max-confirmation-gap", type=float, default=None)
    parser.add_argument(
        "--json", action="store_true", help="print structured task and confirmation results"
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.budget_checkpoint and args.task_timeout is None:
        parser.error("--budget-checkpoint requires --task-timeout")
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
    if args.budget_checkpoint:
        result = run_checkpointed(
            runtime, args.instruction, args.budget_checkpoint, task_timeout_s=args.task_timeout
        )
    else:
        result = runtime.run(args.instruction, task_timeout_s=args.task_timeout)
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False))
        return 0 if result.success else 1
    print(
        f"success={result.success} completed_steps={result.completed_steps} "
        f"message={result.message}"
    )
    return 0 if result.success else 1
