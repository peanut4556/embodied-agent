"""Explicit 3.0/3.1 s delivery contracts on identical frozen exposure evidence."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.evaluate_delivered_challenge import replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path("outputs/evaluations/confirmation-challenge-v1/report.json")
    provenance = Path("docs/evaluations/confirmation-challenge-v1.json")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != json.loads(provenance.read_text())["raw_report_sha256"]:
        raise ValueError("frozen input hash mismatch")
    original = json.loads(source.read_text())
    if original["status"] != "complete":
        raise ValueError("completed input required")
    paths = [Path(__file__), Path("scripts/evaluate_delivered_challenge.py"), provenance] + [
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
    conditions = [(f"latency_{ms}ms", "latency_20ms", ms / 1000) for ms in (0, 20, 80, 100, 120)]
    conditions += [
        (f, f, 0.02)
        for f in (
            "drop_final",
            "drop_burst_frame",
            "reorder_burst",
            "skew_2ms",
            "session_change",
            "clock_drift_60ms",
        )
    ]
    report = {
        "schema_version": 1,
        "status": "running",
        "scope": "Recorded RGB-D replay with explicit delivery reserve; no new physical experiments",
        "input_sha256": digest,
        "source_sha256": hashes,
        "cases": [],
    }
    args.output.mkdir(parents=True, exist_ok=False)

    def save():
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    save()
    try:
        for case in original["cases"]:
            for label, fault, latency in conditions:
                for reserve in (0, 0.1):
                    result = replay(
                        case,
                        original["budget_m"],
                        fault,
                        delivery_budget_s=reserve,
                        latency_s=latency,
                    )
                    report["cases"].append(
                        {
                            "mode": case["mode"],
                            "seed": case["seed"],
                            "condition": label,
                            "reserve_s": reserve,
                            "result": result["guarded"],
                        }
                    )
        report["summary"] = {
            label: {
                str(reserve): sum(
                    c["condition"] == label
                    and c["reserve_s"] == reserve
                    and c["result"]["status"] == "confirmed_in_tray"
                    for c in report["cases"]
                )
                for reserve in (0, 0.1)
            }
            for label, _, _ in conditions
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
