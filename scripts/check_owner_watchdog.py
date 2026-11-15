"""Abrupt owner death while synthetic readers block and ignore SIGTERM (POSIX)."""

import argparse
import functools
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from embodied_agent.process_pair_capture import ProcessPairCapture
from scripts.check_evidence_checkpoint import digest


def block(directory):
    Path(directory, f"entered-{os.getpid()}").write_text("blocked")
    while True:
        time.sleep(0.01)


class Reader:
    def __init__(self, directory):
        self.directory = directory

    def __call__(self, request):
        block(self.directory)


def factory(directory, stage):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if stage == "factory":
        block(directory)
    return Reader(directory)


def owner(directory, stage):
    backend = ProcessPairCapture(
        functools.partial(factory, str(directory), stage), owner_watchdog=True
    )
    pids = [s["pid"] for s in backend.slots]
    (directory / "pids.json").write_text(json.dumps(pids))
    if stage == "reader":
        assert backend.wait_ready()
        for number in range(2):
            assert backend.submit(
                SimpleNamespace(request_id=number, session_id="test", host_time=time.monotonic())
            )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if all((directory / f"entered-{pid}").exists() for pid in pids):
            os._exit(93)  # Deliberately skip backend.cancel(), reap(), and finally.
        time.sleep(0.01)
    os._exit(94)


def states(pids):
    result = subprocess.run(
        ["ps", "-p", ",".join(map(str, pids)), "-o", "pid=,stat="],
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    return {int(line.split()[0]): line.split()[1] for line in result.stdout.splitlines()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=("reader", "factory"))
    args = parser.parse_args()
    if args.worker:
        owner(args.output, args.worker)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    paths = ["scripts/check_owner_watchdog.py", "src/embodied_agent/process_pair_capture.py"]
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
                    "scripts.check_owner_watchdog",
                    "--output",
                    str(directory),
                    "--worker",
                    stage,
                ],
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=8)
                pids = json.loads((directory / "pids.json").read_text())
                start = time.monotonic()
                observed = states(pids)
                while (
                    any(not status.startswith("Z") for status in observed.values())
                    and time.monotonic() - start < 3
                ):
                    time.sleep(0.02)
                    observed = states(pids)
                exited = not any(not status.startswith("Z") for status in observed.values())
                cases.append(
                    {
                        "stage": stage,
                        "owner_exitcode": code,
                        "pids": pids,
                        "observed_states": observed,
                        "workers_exited": exited,
                        "observation_elapsed_s": time.monotonic() - start,
                        "passed": code == 93 and exited,
                    }
                )
            finally:
                # Only this test's isolated process group, for failure cleanup.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=3)
    assert sources == {p: digest(p) for p in paths}
    report = {"sources": sources, "cases": cases, "passed": sum(c["passed"] for c in cases)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(c["passed"] for c in cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
