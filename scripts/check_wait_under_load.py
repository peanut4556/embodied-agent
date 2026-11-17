"""Fixed-denominator waiting probes with bounded background CPU workers."""

import argparse
import json
import multiprocessing
import os
import queue
import time
from pathlib import Path

from scripts.check_evidence_checkpoint import digest
from scripts.check_fixed_wait_probe import run


def burn(stop, reports, lifetime):
    started = time.monotonic()
    cpu = time.process_time()
    reports.put({"event": "ready", "pid": os.getpid()})
    rounds = 0
    while not stop.is_set() and time.monotonic() - started < lifetime:
        cycle = time.monotonic()
        while time.monotonic() - cycle < 0.015 and not stop.is_set():
            value = 1
            for number in range(2000):
                value = (value * 33 + number) % 1000003
            rounds += 1
        stop.wait(max(0, 0.02 - (time.monotonic() - cycle)))
    reports.put(
        {
            "event": "stopped",
            "pid": os.getpid(),
            "cpu_s": time.process_time() - cpu,
            "wall_s": time.monotonic() - started,
            "rounds": rounds,
            "reason": "requested" if stop.is_set() else "lifetime",
        }
    )


class BoundedLoad:
    def __init__(self, count=2, lifetime=20):
        if type(count) is not int or not 1 <= count <= 4:
            raise ValueError("load worker count must be 1..4")
        if (
            isinstance(lifetime, bool)
            or not isinstance(lifetime, (int, float))
            or not 0.1 <= lifetime <= 30
        ):
            raise ValueError("load lifetime must be .1..30 seconds")
        ctx = multiprocessing.get_context("spawn")
        self.stop, self.reports = ctx.Event(), ctx.Queue()
        self.processes = []
        self.count, self.lifetime = count, lifetime
        self.events = []
        self.exitcodes = []

    def __enter__(self):
        try:
            for _ in range(self.count):
                process = multiprocessing.get_context("spawn").Process(
                    target=burn, args=(self.stop, self.reports, self.lifetime)
                )
                process.start()
                self.processes.append(process)
            for _ in self.processes:
                event = self.reports.get(timeout=5)
                self.events.append(event)
                if event["event"] != "ready":
                    raise RuntimeError("load expired before all workers were ready")
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        self.stop.set()
        for process in self.processes:
            process.join(1)
            if process.is_alive():
                process.kill()
                process.join(1)
            self.exitcodes.append(process.exitcode)
            process.close()
        while True:
            try:
                self.events.append(self.reports.get(timeout=0.05))
            except queue.Empty:
                break
        self.reports.close()
        self.reports.join_thread()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        "scripts/check_wait_under_load.py",
        "scripts/check_fixed_wait_probe.py",
        "scripts/check_process_pair_capture.py",
        "src/embodied_agent/process_pair_capture.py",
        "src/embodied_agent/capture_supervisor.py",
        "src/embodied_agent/scheduled_pair_capture.py",
    ]
    sources = {p: digest(p) for p in paths}
    baseline = [run(margin) for margin in (0, 0.0015, 0.0015, 0)]
    with BoundedLoad() as load:
        loaded = [run(margin) for margin in (0.0015, 0, 0, 0.0015)]
    load_completed = (
        all(e["reason"] == "requested" for e in load.events if e["event"] == "stopped")
        and sum(e["event"] == "stopped" for e in load.events) == 2
    )
    assert sources == {p: digest(p) for p in paths}
    cases = [
        {"condition": label, **case}
        for label, group in [("baseline", baseline), ("two_cpu_workers", loaded)]
        for case in group
    ]
    report = out / "report.json"
    report.write_text(
        json.dumps(
            {
                "sources": sources,
                "cases": cases,
                "load_events": load.events,
                "load_exitcodes": load.exitcodes,
            },
            indent=2,
        )
        + "\n"
    )
    summary = {
        "sources": sources,
        "report_path": str(report),
        "report_sha256": digest(report),
        "load_events": load.events,
        "load_exitcodes": load.exitcodes,
        "load_completed": load_completed,
        "cases": [
            {
                "condition": c["condition"],
                "spin_window_s": c["spin_window_s"],
                **c["summary"],
                "workers_stopped": c["capture"]["workers_stopped"],
            }
            for c in cases
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if (
        not load_completed
        or any(code != 0 for code in load.exitcodes)
        or not all(
            c["summary"]["measurement_complete"] and c["capture"]["workers_stopped"] for c in cases
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
