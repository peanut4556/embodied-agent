"""Owner exits while the worker is SIGSTOP-paused; independent supervisor survives."""

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


def paused_job(directory):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (Path(directory) / "worker-ready").write_text(str(os.getpid()))
    os.kill(os.getpid(), signal.SIGSTOP)
    while True:
        time.sleep(0.01)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        job = SupervisedCaptureJob(functools.partial(paused_job, str(args.output)))
        started = job.wait_started()
        pids = [started["supervisor_pid"], started["events"][0]["worker_pid"]]
        (args.output / "pids.json").write_text(json.dumps(pids))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if (args.output / "worker-ready").exists() and states([pids[1]]).get(
                pids[1], ""
            ).startswith("T"):
                (args.output / "stopped-confirmed").write_text("true")
                os._exit(93)
            time.sleep(0.01)
        os._exit(94)
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/check_capture_supervisor.py",
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
                "scripts.check_capture_supervisor",
                "--output",
                str(directory),
                "--worker",
            ],
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=8)
            pids = json.loads((directory / "pids.json").read_text())
            deadline = time.monotonic() + 3
            observed = states(pids)
            while observed and time.monotonic() < deadline:
                time.sleep(0.02)
                observed = states(pids)
            passed = code == 93 and not observed and (directory / "stopped-confirmed").exists()
            cases.append(
                {"owner_exitcode": code, "pids": pids, "final_states": observed, "passed": passed}
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
    print(cases)
    if not all(c["passed"] for c in cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
