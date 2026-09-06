"""Integration checks against running sim-ros and sim-web services."""

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen


def request(port, path="", payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    with urlopen(
        Request(
            f"http://127.0.0.1:{port}/{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        ),
        timeout=20,
    ) as response:
        return json.load(response)


def wait_task():
    deadline = time.monotonic() + 35
    while time.monotonic() < deadline:
        state = request(8765, "api/state")
        if not state["busy"]:
            return state
        time.sleep(0.15)
    raise AssertionError("task timeout")


def main():
    run_id = uuid.uuid4().hex
    # Do not interrupt a task started by a person using the console.
    assert not request(8765, "api/state")["busy"], "console has an active task"
    request(8765, "api/reset", {})
    request(8765, "api/run", {"instruction": "把桌上的红色积木放进盒子", "planner": "rule"})
    result = wait_task()
    assert result["task"]["phase"] == "success", result
    assert result["world"]["inside_box"] and not result["world"]["holding"]
    print("PASS rule plan → ROS actions/status → geometric goal verification")
    request(8765, "api/reset", {})
    failed = request(8766, payload={"name": "verify", "parameters": {"object": "red_block"}})
    assert not failed["success"]
    print("PASS verify rejects block outside box")
    with ThreadPoolExecutor() as executor:
        motion = executor.submit(
            request,
            8766,
            "",
            {
                "name": "pick",
                "parameters": {"object": "red_block"},
                "run_id": run_id,
            },
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not request(8766).get("active"):
            time.sleep(0.05)
        assert request(8766).get("active"), "motion did not start"
        stopped = request(8766, payload={"name": "stop", "run_id": run_id})
        assert stopped["success"]
        assert not motion.result()["success"]
    before = request(8766)["tip"]
    time.sleep(0.3)
    assert request(8766)["tip"] == before
    cancelled = request(
        8766,
        payload={"name": "pick", "run_id": run_id, "parameters": {"object": "red_block"}},
    )
    assert not cancelled["success"]
    print("PASS stop interrupts trajectory and rejects subsequent commands for cancelled task")
    request(8765, "api/reset", {})
    print("PASS reset; scene ready")


if __name__ == "__main__":
    main()
