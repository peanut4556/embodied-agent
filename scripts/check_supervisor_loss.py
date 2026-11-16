"""Kill only the supervisor; observe reader fallback and conservative cleanup state."""

import argparse
import functools
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from embodied_agent.process_pair_capture import ProcessPairCapture
from scripts.check_evidence_checkpoint import digest
from scripts.check_owner_watchdog import factory, states


def case(directory, stage):
    backend = ProcessPairCapture(
        functools.partial(factory, str(directory), stage), capacity=1, supervised=True
    )
    try:
        if stage == "reader":
            assert backend.wait_ready()
            from types import SimpleNamespace

            backend.submit(
                SimpleNamespace(request_id=1, session_id="test", host_time=time.monotonic())
            )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            info = backend.result()["supervision"][0]
            if info["events"]:
                pid = info["events"][0]["worker_pid"]
                if (directory / f"entered-{pid}").exists():
                    break
            time.sleep(0.01)
        else:
            raise RuntimeError("reader did not enter blocked state")
        supervisor = backend.slots[0]["pid"]
        os.kill(supervisor, signal.SIGKILL)
        deadline = time.monotonic() + 3
        observed = states([pid])
        while observed and time.monotonic() < deadline:
            time.sleep(0.02)
            observed = states([pid])
        backend.cancel()
        cleanup = backend.reap(0.25)
        passed = (
            not observed
            and not cleanup["workers_stopped"]
            and cleanup["supervision"][0]["cleanup_status"] == "supervisor_lost"
        )
        return {
            "stage": stage,
            "worker_pid": pid,
            "supervisor_pid": supervisor,
            "final_worker_states": observed,
            "cleanup": cleanup,
            "passed": passed,
        }
    finally:
        backend.cancel()
        backend.reap(0.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=("reader", "factory"))
    args = parser.parse_args()
    if args.worker:
        result = case(args.output, args.worker)
        (args.output / "case.json").write_text(json.dumps(result, indent=2) + "\n")
        return
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/check_supervisor_loss.py",
        "scripts/check_owner_watchdog.py",
        "src/embodied_agent/capture_supervisor.py",
        "src/embodied_agent/process_pair_capture.py",
    ]
    sources = {p: digest(p) for p in paths}
    cases = []
    for repeat in range(2):
        for stage in ("reader", "factory"):
            directory = args.output / f"{stage}-{repeat}"
            directory.mkdir()
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "scripts.check_supervisor_loss",
                    "--output",
                    str(directory),
                    "--worker",
                    stage,
                ],
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=10)
                result = json.loads((directory / "case.json").read_text())
                result["driver_exitcode"] = code
                cases.append(result)
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
    print([(c["stage"], c["passed"]) for c in cases])
    if not all(c["passed"] and c["driver_exitcode"] == 0 for c in cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
