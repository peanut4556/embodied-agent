"""Replay frozen rendered pairs through delivery faults; no new physical trials."""

import argparse
import hashlib
import json
from pathlib import Path

from embodied_agent.confirmation_challenge import ConfirmationChallenge
from embodied_agent.delivered_challenge import DeliveredChallenge
from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition

FAULTS = (
    "on_time",
    "latency_20ms",
    "final_late_20ms",
    "drop_final",
    "drop_burst_frame",
    "reorder_burst",
    "skew_2ms",
    "session_change",
    "clock_drift_60ms",
)


def replay(case, budget, fault, *, delivery_budget_s=0.0, latency_s=0.02):
    if fault not in FAULTS:
        raise ValueError("unknown delivery fault")
    raw = ConfirmationChallenge(0, position_error_m=budget)
    guarded = DeliveredChallenge(
        0,
        ExposureClockMap(0, 0),
        "camera",
        position_error_m=budget,
        delivery_budget_s=delivery_budget_s,
    )
    events = []
    for sample in case["samples"]:
        if "challenge" not in sample["recipients"]:
            continue
        t = sample["elapsed_s"]
        if fault == "drop_final" and abs(t - 3) < 1e-9:
            continue
        if fault == "drop_burst_frame" and abs(t - 2.05) < 1e-9:
            continue
        arrival = t + (
            latency_s
            if fault == "latency_20ms" or (fault == "final_late_20ms" and abs(t - 3) < 1e-9)
            else 0
        )
        if fault == "reorder_burst" and abs(t - 2.05) < 1e-9:
            arrival = 2.11
        events.append((arrival, t, sample["profiles"]["coverage"]))
    events.sort(key=lambda event: event[0])
    log = []
    for arrival, exposure, packet in events:
        # A watchdog at the configured delivery deadline wins over later delivery.
        if arrival > guarded.deadline + 1e-9:
            guarded.tick(guarded.deadline)
        session = (
            "replacement" if fault == "session_change" and exposure >= 2.05 - 1e-9 else "camera"
        )
        host = arrival + (0.06 if fault == "clock_drift_60ms" and exposure >= 2.05 - 1e-9 else 0)
        secondary_time = exposure + (
            0.002 if fault == "skew_2ms" and exposure >= 2.05 - 1e-9 else 0
        )
        # Delay delivery so the skew fault tests synchronization rather than future time.
        if secondary_time > host:
            host = arrival = secondary_time
        a = (
            None
            if packet["primary"] is None
            else StampedPosition(tuple(packet["primary"]), exposure)
        )
        b = (
            None
            if packet["secondary"] is None
            else StampedPosition(tuple(packet["secondary"]), secondary_time)
        )
        guarded.deliver(arrival, host, a, b, session_id=session)
        # Exposure-only control deliberately lacks delivery/session/skew validation.
        if raw.status == "pending":
            try:
                raw.step(exposure, packet["primary"], packet["secondary"])
            except ValueError:
                raw.step(3.01, None, None)
        log.append(
            {
                "delivery_time": arrival,
                "exposure_time": exposure,
                "guarded_status": guarded.status,
                "raw_status": raw.status,
            }
        )
    guarded.tick(guarded.deadline)
    return {
        "fault": fault,
        "raw_exposure_only": raw.result(),
        "guarded": guarded.result(),
        "events": log,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance_path = Path("docs/evaluations/confirmation-challenge-v1.json")
    provenance = json.loads(provenance_path.read_text())
    source = Path("outputs/evaluations/confirmation-challenge-v1/report.json")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != provenance["raw_report_sha256"]:
        raise ValueError("frozen render report hash mismatch")
    original = json.loads(source.read_text())
    if original["status"] != "complete":
        raise ValueError("completed render report required")
    paths = [Path(__file__), provenance_path] + [
        Path("src/embodied_agent") / name
        for name in (
            "delivered_challenge.py",
            "confirmation_challenge.py",
            "exposure_confirmation.py",
            "observation_capture.py",
            "settled_reference.py",
            "temporal_reference.py",
            "cross_view.py",
            "release_confirmation.py",
            "reference_confirmation.py",
        )
    ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "scope": "Delivery fault replay of previously rendered pairs; not new dynamics or real camera timing",
        "input_report": str(source),
        "input_sha256": source_hash,
        "source_sha256": hashes,
        "cases": [],
    }

    def save():
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    save()
    try:
        for case in original["cases"]:
            for fault in FAULTS:
                result = replay(case, original["budget_m"], fault)
                report["cases"].append({"mode": case["mode"], "seed": case["seed"], **result})
        report["summary"] = {
            f: {
                k: sum(
                    c[fkey]["status"] == "confirmed_in_tray"
                    for c in report["cases"]
                    if c["fault"] == f
                )
                for k, fkey in (
                    ("exposure_only_confirmations", "raw_exposure_only"),
                    ("guarded_confirmations", "guarded"),
                )
            }
            for f in FAULTS
        }
        if hashes != {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}:
            raise RuntimeError("source changed during replay")
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        save()
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
