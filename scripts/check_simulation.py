"""Integration checks against running sim-ros and sim-web services."""

import argparse
import json
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen


def request(port, path="", payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    with urlopen(
        Request(
            f"http://127.0.0.1:{port}/{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        ),
        timeout=25,
    ) as response:
        return json.load(response)


def wait_task(timeout=70):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = request(8765, "api/state")
        if not state["busy"]:
            return state
        time.sleep(0.15)
    raise AssertionError("task timeout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rai", action="store_true", help="Also run the Qwen Chinese task")
    parser.add_argument("--report", type=Path, help="Save successful task states as JSON")
    parser.add_argument("--expect-perception", choices=["truth", "rgbd"])
    parser.add_argument("--execution", choices=["scripted", "feedback"], default="scripted")
    args = parser.parse_args()
    report = {}
    run_id = uuid.uuid4().hex
    # Do not interrupt a task started by a person using the console.
    initial = request(8765, "api/state")
    assert not initial["busy"], "console has an active task"
    if args.expect_perception:
        assert initial["world"].get("perception") == args.expect_perception, initial
    request(8765, "api/reset", {})
    task = {
        "instruction": "把桌上的红色积木放进盒子",
        "planner": "rule",
        "execution": args.execution,
    }
    request(8765, "api/run", task)
    result = wait_task(timeout=110)
    assert result["task"]["phase"] == "success", result
    assert result["world"]["inside_box"] and not result["world"]["holding"]
    report["rule"] = result
    if args.execution == "feedback":
        assert result["world"]["execution"]["state"] == "completed", result
        assert any(s["action"] == "learned_pick_place" for s in result["task"]["plan"])
        # A second task without reset must reject the scene (target no longer visible at home).
        request(8765, "api/run", task)
        rejected = wait_task(timeout=110)
        assert rejected["task"]["phase"] == "failed", rejected
        assert not rejected["world"]["active"]
        report["invalid_start"] = rejected
        print("PASS learned task rejects unreset scene without fallback", flush=True)
    if args.expect_perception == "rgbd":
        assert result["world"]["detection"]["source"] == "rgbd"
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
    # Physical joints settle under position control; the ideal 2D model stops instantly.
    time.sleep(0.5)
    before = request(8766)["tip"]
    time.sleep(0.3)
    assert math.dist(request(8766)["tip"], before) < 0.01
    cancelled = request(
        8766,
        payload={"name": "pick", "run_id": run_id, "parameters": {"object": "red_block"}},
    )
    assert not cancelled["success"]
    print("PASS stop interrupts trajectory and rejects subsequent commands for cancelled task")
    request(8765, "api/reset", {})
    print("PASS reset; scene ready")
    report["stop_and_reset"] = "passed"
    if args.execution == "feedback":
        request(8765, "api/run", task)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = request(8765, "api/state")
            if state["world"].get("active") == "learned_pick_place":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("learned motion did not start")
        cancelled_run = state["task"]["run_id"]
        request(8765, "api/stop", {})
        stopped = wait_task()
        assert stopped["task"]["phase"] == "stopped", stopped
        assert stopped["world"]["execution"]["state"] == "stopped", stopped
        assert not stopped["world"]["active"]
        time.sleep(0.5)
        before = request(8766)["tip"]
        time.sleep(0.3)
        assert math.dist(request(8766)["tip"], before) < 0.01
        cancelled = request(
            8766,
            payload={
                "name": "learned_pick_place",
                "run_id": cancelled_run,
                "parameters": {"object": "red_block", "destination": "box"},
            },
        )
        assert not cancelled["success"]
        report["learned_stop"] = stopped
        request(8765, "api/reset", {})
        assert request(8766)["execution"] is None
        print(
            "PASS console stop interrupts learned motion and cancellation stays latched", flush=True
        )
    if args.rai:
        request(8765, "api/run", dict(task, planner="rai"))
        result = wait_task(timeout=250)
        assert result["task"]["phase"] == "success", result
        assert result["world"]["inside_box"] and not result["world"]["holding"]
        report["rai"] = result
        if args.expect_perception == "rgbd":
            assert result["world"]["detection"]["source"] == "rgbd"
        print("PASS Qwen Chinese plan → ROS actions/status → goal verification")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"Report saved: {args.report}")


if __name__ == "__main__":
    main()
