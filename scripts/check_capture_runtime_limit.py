"""Pause a live caller; verify an independent supervisor enforces its job limit."""

import argparse
import functools
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from embodied_agent.capture_supervisor import SupervisedCaptureJob
from scripts.check_evidence_checkpoint import digest
from scripts.check_owner_watchdog import states


def blocked(directory):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (Path(directory) / "worker-ready").write_text("true")
    while True:
        time.sleep(0.01)


def owner(directory):
    job = SupervisedCaptureJob(functools.partial(blocked, str(directory)), runtime_limit_s=1)
    result = job.wait_started()
    pids = [result["supervisor_pid"], result["events"][0]["worker_pid"]]
    (directory / "pids.json").write_text(json.dumps(pids))
    deadline = time.monotonic() + 0.5
    while not (directory / "worker-ready").exists() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert (directory / "worker-ready").exists()
    os.kill(os.getpid(), signal.SIGSTOP)
    # Only the external test controller resumes us after checking the children.
    result = job.close()
    (directory / "owner-result.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        owner(args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/check_capture_runtime_limit.py",
        "scripts/check_owner_watchdog.py",
        "src/embodied_agent/capture_supervisor.py",
    ]
    sources = {p: digest(p) for p in paths}
    cases = []
    for repeat in range(2):
        directory = args.output / str(repeat)
        directory.mkdir()
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scripts.check_capture_runtime_limit",
                "--output",
                str(directory),
                "--worker",
            ],
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 5
            while (
                not states([process.pid]).get(process.pid, "").startswith("T")
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            assert states([process.pid]).get(process.pid, "").startswith("T")
            pids = json.loads((directory / "pids.json").read_text())
            deadline = time.monotonic() + 3
            observed = states(pids)
            while (
                any(not state.startswith("Z") for state in observed.values())
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)
                observed = states(pids)
            owner_state = states([process.pid]).get(process.pid, "")
            os.kill(process.pid, signal.SIGCONT)
            code = process.wait(timeout=3)
            result = json.loads((directory / "owner-result.json").read_text())
            final = result["events"][-1]
            final_states = states(pids)
            passed = (
                pids[1] not in observed
                and all(state.startswith("Z") for state in observed.values())
                and not final_states
                and owner_state.startswith("T")
                and code == 0
                and final["runtime_limit_exceeded"]
                and not final["owner_connection_closed"]
                and final["status"] == "stopped"
            )
            cases.append(
                {
                    "passed": passed,
                    "owner_state_before_resume": owner_state,
                    "children_before_resume": observed,
                    "children_after_resume": final_states,
                    "owner_exitcode": code,
                    "result": result,
                }
            )
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)
    assert sources == {p: digest(p) for p in paths}
    (args.output / "report.json").write_text(
        json.dumps({"sources": sources, "cases": cases}, indent=2) + "\n"
    )
    print([(c["passed"], c["result"]["events"][-1]["elapsed_s"]) for c in cases])
    if not all(c["passed"] for c in cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
