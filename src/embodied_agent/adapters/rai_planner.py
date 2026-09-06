from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..models import Observation, Plan


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
