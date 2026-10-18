"""Measure cancellation of live RAI workers, then run a fresh Qwen learning task."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from check_console_interactions import TASK
from check_simulation import request, wait_task

ROOT = Path(__file__).resolve().parents[1]


def workers():
    output = subprocess.run(
        ["ps", "-ax", "-o", "pid=,command="], capture_output=True, text=True, check=True
    ).stdout
    marker = str(ROOT / "scripts/rai_plan.py") + " --config"
    return {int(line.strip().split(None, 1)[0]) for line in output.splitlines() if marker in line}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=Path("outputs/planning-cancellation-v1.json")
    )
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("refusing to overwrite report")
    initial = request(8765, "api/state")
    assert not initial["busy"] and initial["world"] and not initial["world"]["active"]
    assert initial["world"]["feedback_available"] and initial["world"]["engine"] == "mujoco"
    assert initial["world"]["perception"] == "rgbd"
    assert not workers(), "do not interfere with an existing RAI worker"
    report = {
        "status": "running",
        "trials": [],
        "limit_seconds": 3,
        "scope": "Local RAI worker cancellation, not proof of Ollama GPU inference cancellation.",
    }
    try:
        for delay in (0.3, 1.0, 2.0):
            request(8765, "api/reset", {})
            request(8765, "api/run", dict(TASK, planner="rai", execution="feedback"))
            deadline = time.monotonic() + 10
            found = set()
            while time.monotonic() < deadline:
                found = workers()
                if found:
                    break
                time.sleep(0.02)
            assert len(found) == 1, "must observe exactly one live RAI process before cancelling"
            time.sleep(delay)
            before = request(8765, "api/state")
            assert before["task"]["phase"] == "planning" and found <= workers(), before
            start = time.monotonic()
            request(8765, "api/stop", {})
            acknowledged = time.monotonic() - start
            stopped = wait_task(timeout=5)
            elapsed = time.monotonic() - start
            remaining = workers()
            trial = {
                "delay_after_worker_seconds": delay,
                "observed_worker_pids": sorted(found),
                "stop_ack_seconds": acknowledged,
                "idle_after_stop_seconds": elapsed,
                "remaining_workers": sorted(remaining),
                "state": stopped,
            }
            report["trials"].append(trial)
            assert elapsed < report["limit_seconds"], trial
            assert not remaining, trial
            assert stopped["task"]["phase"] == "stopped" and stopped["task"]["completed"] == 0
            assert not stopped["world"]["active"] and stopped["world"]["execution"] is None
            assert not any(e["text"].startswith("ROS →") for e in stopped["task"]["events"])
            print(
                json.dumps(
                    {
                        k: trial[k]
                        for k in (
                            "delay_after_worker_seconds",
                            "stop_ack_seconds",
                            "idle_after_stop_seconds",
                            "remaining_workers",
                        )
                    }
                ),
                flush=True,
            )
        request(8765, "api/reset", {})
        request(8765, "api/run", dict(TASK, planner="rai", execution="feedback"))
        restarted = wait_task(timeout=250)
        report["restarted"] = restarted
        assert restarted["task"]["phase"] == "success", restarted
        assert restarted["world"]["inside_box"] and not restarted["world"]["holding"], restarted
        assert restarted["world"]["execution"]["state"] == "completed", restarted
        assert restarted["world"]["execution"]["run_id"] == restarted["task"]["run_id"]
        assert restarted["task"]["run_id"] not in [
            r["state"]["task"]["run_id"] for r in report["trials"]
        ]
        assert not workers(), "successful worker was not reaped"
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
        report["source_sha256"] = {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in (
                "scripts/check_planning_cancellation.py",
                "src/embodied_agent/sim_app.py",
                "src/embodied_agent/adapters/rai_planner.py",
                "scripts/rai_plan.py",
                "config/rai.ollama.toml",
                "outputs/models/context-bc-v1/policy.npz",
            )
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(report["status"], flush=True)


if __name__ == "__main__":
    main()
