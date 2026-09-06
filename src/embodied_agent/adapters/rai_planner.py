from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..models import Observation, Plan, PlanStep

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class RAIPlannerAdapter:
    """Boundary for a configured RAI agent.

    RAI model/provider setup evolves independently from this app. Supply a callable
    that returns validated structured plan data, then decode it into our Plan type.
    """

    def __init__(
        self,
        invoke_agent: Callable[[str, Observation], Mapping[str, Any]],
        decode_plan: Callable[[Mapping[str, Any]], Plan],
    ) -> None:
        self.invoke_agent = invoke_agent
        self.decode_plan = decode_plan

    def create_plan(self, instruction: str, observation: Observation) -> Plan:
        payload = self.invoke_agent(instruction, observation)
        return self.decode_plan(payload)


class RAISubprocessPlanner:
    """Run RAI in its isolated environment and exchange only JSON payloads."""

    def __init__(
        self,
        *,
        python_executable: Path | str = PROJECT_ROOT / ".venv-rai/bin/python",
        worker_script: Path | str = PROJECT_ROOT / "scripts/rai_plan.py",
        config_path: Path | str = PROJECT_ROOT / "config/rai.ollama.toml",
        timeout_seconds: float = 180,
    ) -> None:
        self.python_executable = Path(python_executable)
        self.worker_script = Path(worker_script)
        self.config_path = Path(config_path)
        self.timeout_seconds = timeout_seconds

    def create_plan(self, instruction: str, observation: Observation) -> Plan:
        request = {
            "instruction": instruction,
            "observation": asdict(observation),
        }
        command = [
            str(self.python_executable),
            str(self.worker_script),
            "--config",
            str(self.config_path),
        ]
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(request, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "RAI environment is missing; create .venv-rai and install requirements/rai.txt"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("RAI planner timed out") from exc

        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"RAI planner failed: {details}")

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("RAI planner returned invalid JSON") from exc
        return decode_plan(payload)


def decode_plan(payload: Mapping[str, Any]) -> Plan:
    goal = payload.get("goal")
    raw_steps = payload.get("steps")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("RAI plan goal must be a non-empty string")
    if not isinstance(raw_steps, list):
        raise TypeError("RAI plan steps must be a list")

    steps: list[PlanStep] = []
    for item in raw_steps:
        if not isinstance(item, Mapping):
            raise TypeError("each RAI plan step must be an object")
        action = item.get("action")
        arguments = item.get("arguments", {})
        success_condition = item.get("success_condition", "")
        if not isinstance(action, str) or not action:
            raise ValueError("each RAI plan action must be a non-empty string")
        if not isinstance(arguments, dict):
            raise TypeError("RAI plan step arguments must be an object")
        if not isinstance(success_condition, str):
            raise TypeError("RAI success_condition must be a string")
        steps.append(PlanStep(action, arguments, success_condition))

    return Plan(goal.strip(), steps)
