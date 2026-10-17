"""Exercise the real console → ROS → RGB-D MuJoCo path; no mocked bridge."""

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError

from check_simulation import request, wait_task

TASK = {"instruction": "把桌上的红色积木放进盒子", "planner": "rule", "execution": "scripted"}


def wait_until(predicate, timeout=35):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = request(8765, "api/state")
        if predicate(state):
            return state
        if not state["busy"] and state["task"]["phase"] in {"failed", "stopped"}:
            raise AssertionError(state)
        time.sleep(0.05)
    raise AssertionError("expected task state was not reached")


def rejected(path, payload):
    try:
        request(8765, path, payload)
    except HTTPError as error:
        body = json.load(error)
        assert error.code == 400 and body.get("error"), body
        return body
    raise AssertionError(f"{path} unexpectedly accepted")


def pause_during_pick():
    request(8765, "api/reset", {})
    request(8765, "api/run", TASK)
    wait_until(lambda state: state["world"]["active"] == "pick")
    request(8765, "api/pause", {})
    paused = wait_until(lambda state: state["task"]["phase"] == "paused")
    assert paused["busy"] and not paused["world"]["active"], paused
    assert paused["world"]["holding"], paused
    assert paused["task"]["completed"] == 2, paused
    # Observe several ROS updates. No next command, no progress, no completion.
    time.sleep(0.8)
    later = request(8765, "api/state")
    assert later["task"]["phase"] == "paused" and later["busy"], later
    assert later["task"]["completed"] == 2 and not later["world"]["active"], later
    assert later["world"]["holding"], later
    return later


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("outputs/console-interactions.json"))
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("refusing to overwrite report")
    initial = request(8765, "api/state")
    assert not initial["busy"], "do not interrupt an existing console task"
    assert initial["world"] and not initial["world"]["active"], "simulator not idle"
    assert initial["world"]["engine"] == "mujoco", initial
    assert initial["world"]["perception"] == "rgbd", initial
    report = {"status": "running", "engine": "mujoco", "perception": "rgbd", "checks": {}}
    try:
        report["paused"] = pause_during_pick()
        report["duplicate_start"] = rejected("api/run", TASK)
        report["reset_while_paused"] = rejected("api/reset", {})
        request(8765, "api/resume", {})
        success = wait_task()
        assert success["task"]["phase"] == "success", success
        assert success["world"]["inside_box"] and not success["world"]["holding"], success
        assert success["task"]["completed"] == len(success["task"]["plan"]), success
        report["resumed"] = success
        report["checks"]["pause_resume_to_goal"] = True
        report["checks"]["busy_rejects_start_and_reset"] = True
        paused = pause_during_pick()
        request(8765, "api/stop", {})
        stopped = wait_task()
        assert stopped["task"]["phase"] == "stopped", stopped
        assert not stopped["world"]["active"], stopped
        assert stopped["task"]["completed"] == paused["task"]["completed"], stopped
        report["stopped"] = stopped
        report["checks"]["stop_from_pause"] = True
        rejected("api/resume", {})
        report["checks"]["cannot_resume_stopped_run"] = True
        request(8765, "api/reset", {})
        # A deliberately unavailable model tests the real failure presentation path.
        if not initial["world"].get("feedback_available"):
            request(8765, "api/run", dict(TASK, execution="feedback"))
            failed = wait_task()
            assert failed["task"]["phase"] == "failed", failed
            assert failed["task"]["failure"]["title"] == "学习模型不可用", failed
            assert failed["task"]["failure"]["detail"] and failed["task"]["failure"]["recovery"]
            assert not failed["world"]["active"]
            report["failed"] = failed
            report["checks"]["actionable_failure"] = True
        else:
            report["checks"]["actionable_failure"] = "skipped: model configured"
        request(8765, "api/reset", {})
        final = request(8765, "api/state")
        assert final["task"]["phase"] == "idle" and not final["busy"], final
        report["checks"]["reset_to_idle"] = True
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        try:
            request(8765, "api/stop", {})
        except (OSError, ValueError) as stop_error:
            report["cleanup_error"] = str(stop_error)
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps({"status": report["status"], "checks": report["checks"]}, ensure_ascii=False)
        )


if __name__ == "__main__":
    main()
