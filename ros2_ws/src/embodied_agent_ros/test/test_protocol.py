import json

import pytest
from embodied_agent_ros.protocol import status_json, validate_action_json


def test_accepts_known_action():
    action = validate_action_json('{"name":"pick","parameters":{"object":"red_block"}}')
    assert action == {"name": "pick", "parameters": {"object": "red_block"}}


def test_rejects_unknown_action():
    with pytest.raises(ValueError, match="not allowed"):
        validate_action_json('{"name":"disable_safety"}')


def test_status_is_machine_readable():
    assert json.loads(status_json(accepted=True, action="pick"))["accepted"] is True
