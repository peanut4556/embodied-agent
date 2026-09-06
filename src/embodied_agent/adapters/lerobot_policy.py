from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..models import Action, Observation, PlanStep


class LeRobotPolicyAdapter:
    """Thin wrapper around an initialized LeRobot-compatible policy.

    Observation preprocessing and action decoding are robot-specific, so callers
    inject both functions instead of hiding unsafe assumptions in this adapter.
    """

    def __init__(
        self,
        policy: Any,
        encode_observation: Callable[[PlanStep, Observation], Mapping[str, Any]],
        decode_action: Callable[[Any], Action],
    ) -> None:
        self.policy = policy
        self.encode_observation = encode_observation
        self.decode_action = decode_action

    def select_action(self, step: PlanStep, observation: Observation) -> Action:
        model_input = self.encode_observation(step, observation)
        raw_action = self.policy.select_action(model_input)
        return self.decode_action(raw_action)
