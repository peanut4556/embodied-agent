"""Real learning-skill boundary pause and cancellation during local Qwen planning."""

import argparse
import json
import time
from pathlib import Path

from check_console_interactions import TASK, rejected, wait_until
from check_simulation import request, wait_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("outputs/learned-interactions-v1.json"))
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("refusing to overwrite report")
    state = request(8765, "api/state")
    assert not state["busy"] and not state["world"]["active"], (
        "existing task must not be interrupted"
    )
    assert state["world"]["engine"] == "mujoco" and state["world"]["perception"] == "rgbd"
    assert state["world"]["feedback_available"], "learning checkpoint must be configured"
    report = {"status": "running", "checks": {}}
    try:
        request(8765, "api/reset", {})
        request(8765, "api/run", dict(TASK, execution="feedback"))
        active = wait_until(lambda s: s["world"]["active"] == "learned_pick_place")
        request(8765, "api/pause", {})
        paused = wait_until(lambda s: s["task"]["phase"] == "paused", timeout=110)
        assert paused["task"]["completed"] == 2 and paused["busy"], paused
        assert not paused["world"]["active"] and paused["world"]["inside_box"], paused
        assert paused["world"]["execution"]["state"] == "completed", paused
        assert paused["world"]["execution"]["run_id"] == active["task"]["run_id"]
        time.sleep(0.6)
        still = request(8765, "api/state")
        assert still["task"]["phase"] == "paused" and still["task"]["completed"] == 2
        report["paused_after_skill"] = still
        request(8765, "api/resume", {})
        resumed = wait_task()
        assert resumed["task"]["phase"] == "success", resumed
        report["resumed"] = resumed
        report["checks"]["learning_pause_at_skill_boundary"] = True
        report["checks"]["resume_runs_remaining_verification"] = True
        request(8765, "api/reset", {})
        request(8765, "api/run", dict(TASK, execution="feedback", planner="rai"))
        planning = request(8765, "api/state")
        assert planning["task"]["phase"] == "planning", "planner completed before stop test"
        request(8765, "api/stop", {})
        stopped = wait_task(timeout=210)
        assert stopped["task"]["phase"] == "stopped", stopped
        assert stopped["task"]["completed"] == 0 and not stopped["world"]["active"], stopped
        assert stopped["world"]["execution"] is None, stopped
        assert not any(e["text"].startswith("ROS →") for e in stopped["task"]["events"]), stopped
        rejected("api/resume", {})
        report["stopped_during_planning"] = stopped
        report["checks"]["planning_stop_prevents_motion"] = True
        request(8765, "api/reset", {})
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
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
