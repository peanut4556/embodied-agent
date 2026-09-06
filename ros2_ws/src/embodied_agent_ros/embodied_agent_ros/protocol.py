from __future__ import annotations

import json
from typing import Any

ALLOWED_ACTIONS = frozenset({"locate", "pick", "place", "verify", "stop"})


def validate_action_json(raw: str) -> dict[str, Any]:
    """Parse and validate the semantic action envelope sent over ROS 2."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {error.msg}") from error

    if not isinstance(payload, dict):
        raise TypeError("action payload must be a JSON object")

    name = payload.get("name")
    parameters = payload.get("parameters", {})
    if name not in ALLOWED_ACTIONS:
        raise ValueError(f"action is not allowed: {name}")
    if not isinstance(parameters, dict):
        raise TypeError("parameters must be a JSON object")

    return {"name": name, "parameters": parameters}


def status_json(*, accepted: bool, action: str = "", reason: str = "") -> str:
    return json.dumps(
        {"accepted": accepted, "action": action, "reason": reason},
        ensure_ascii=False,
        sort_keys=True,
    )
