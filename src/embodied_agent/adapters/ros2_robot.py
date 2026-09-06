from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..models import Action, Observation


class ROS2RobotAdapter:
    """Robot adapter built from ROS-specific observation and publisher callbacks.

    Keep rclpy and message types in the composition root because every robot uses
    different topics, services, actions, QoS, and safety controllers.
    """

    def __init__(
        self,
        read_observation: Callable[[], Observation],
        publish_json: Callable[[str], Any],
    ) -> None:
        self.read_observation = read_observation
        self.publish_json = publish_json

    def observe(self) -> Observation:
        return self.read_observation()

    def execute(self, action: Action) -> None:
        self.publish_json(json.dumps({"name": action.name, "parameters": action.parameters}))
