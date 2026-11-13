"""Compare just-in-time dispatch with advance reservations, using synthetic readers."""

import argparse
import functools
import hashlib
import json
import time
from pathlib import Path

from embodied_agent.phase_sweep_challenge import SWEEP_INTERVALS_MS
from embodied_agent.process_pair_capture import ProcessPairCapture
from embodied_agent.scheduled_pair_capture import ScheduledPairRequest, ScheduledReaderFactory
from embodied_agent.two_stage_capture import StageCaptureRequest
from scripts.check_capture_timing import DelayedReader


def run(profile, mode, repetition):
    factory = functools.partial(
        DelayedReader,
        0.005 if mode == "reader_delay" else 0,
        0.005 if mode == "return_delay" else 0,
    )
    if profile == "reserved":
        factory = ScheduledReaderFactory(factory)
    backend = ProcessPairCapture(factory, trace_timing=True)
    try:
        assert backend.wait_ready()
        start = time.monotonic() + 0.05
        targets = [start]
        for interval in SWEEP_INTERVALS_MS[:11]:
            targets.append(targets[-1] + interval / 1000)
        lead = 0.02 if profile == "reserved" else 0
        rows, index = {}, 0
        deadline = targets[-1] + 0.15
        while time.monotonic() < deadline:
            for request_id, pair, error in backend.drain():
                row = rows[request_id]
                received = time.monotonic()
                row["error"] = error
                row["received_at"] = received
                if pair is not None:
                    exposures = [pair.primary.captured_at, pair.secondary.captured_at]
                    row["exposures"] = exposures
                    row["window_met"] = (
                        not error
                        and pair.session_id == "synthetic-reservation"
                        and all(row["target"] <= t <= row["target"] + 0.002 for t in exposures)
                        and all(t <= received for t in exposures)
                    )
                    row["age_at_drain_s"] = received - min(exposures)
            now = time.monotonic()
            if index < len(targets) and now >= targets[index] - lead:
                target = targets[index]
                request_id = index + 1
                row = {"request_id": request_id, "target": target, "window_met": False}
                rows[request_id] = row
                if profile == "reserved" and now > target:
                    row["error"] = "reservation_submission_missed"
                else:
                    req = (
                        ScheduledPairRequest(
                            request_id, "synthetic-reservation", now, target, target + 0.002
                        )
                        if profile == "reserved"
                        else StageCaptureRequest(
                            request_id, "probe", "synthetic-reservation", now - start, now
                        )
                    )
                    if mode == "dispatch_delay":
                        time.sleep(0.005)
                    row["submitted"] = backend.submit(req)
                    if not row["submitted"]:
                        row["error"] = "capacity_unavailable"
                index += 1
            if index == len(targets) and all(
                "received_at" in r or "error" in r for r in rows.values()
            ):
                break
            time.sleep(0.0005)
        # Unissued slots and missing results count as failures, never disappear.
        for i, target in enumerate(targets, 1):
            rows.setdefault(i, {"request_id": i, "target": target, "window_met": False})
            if "received_at" not in rows[i] and "error" not in rows[i]:
                rows[i]["error"] = "probe_deadline"
        backend.cancel()
        cleanup = backend.reap(0.25)
        assert cleanup["workers_stopped"]
        return {
            "profile": profile,
            "mode": mode,
            "repetition": repetition,
            "lead_s": lead,
            "rows": list(rows.values()),
            "capture": backend.result(),
            "task_confirmation_performed": False,
        }
    finally:
        backend.cancel()
        backend.reap(0.25)


def sources():
    paths = (
        "scripts/evaluate_scheduled_capture.py",
        "scripts/check_capture_timing.py",
        "src/embodied_agent/scheduled_pair_capture.py",
        "src/embodied_agent/process_pair_capture.py",
        "src/embodied_agent/phase_sweep_challenge.py",
        "src/embodied_agent/threaded_pair_capture.py",
        "src/embodied_agent/two_stage_capture.py",
        "src/embodied_agent/observation_capture.py",
    )
    return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = sources()
    cases = []
    for repetition in range(3):
        for mode in ("baseline", "dispatch_delay", "reader_delay", "return_delay"):
            profiles = (
                ("immediate", "reserved") if repetition % 2 == 0 else ("reserved", "immediate")
            )
            for profile in profiles:
                cases.append(run(profile, mode, repetition))
    assert sources() == before
    report = args.output / "report.json"
    report.write_text(json.dumps({"sources": before, "cases": cases}, indent=2) + "\n")
    summary = {
        "kind": "advance_reservation_synthetic_exposure_probe",
        "task_confirmation_performed": False,
        "sources": before,
        "report_path": str(report),
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "groups": [],
    }
    for mode in ("baseline", "dispatch_delay", "reader_delay", "return_delay"):
        for profile in ("immediate", "reserved"):
            selected = [c for c in cases if c["mode"] == mode and c["profile"] == profile]
            rows = [r for c in selected for r in c["rows"]]
            summary["groups"].append(
                {
                    "mode": mode,
                    "profile": profile,
                    "slots": len(rows),
                    "windows_met": sum(r["window_met"] for r in rows),
                    "errors": sum(bool(r.get("error")) for r in rows),
                    "per_run_windows_met": [
                        sum(r["window_met"] for r in c["rows"]) for c in selected
                    ],
                    "all_workers_stopped": all(c["capture"]["workers_stopped"] for c in selected),
                }
            )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["groups"], indent=2))


if __name__ == "__main__":
    main()
