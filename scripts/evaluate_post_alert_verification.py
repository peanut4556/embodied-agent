"""Compare bounded visual rechecks with clearing an alert on one inside frame."""

import argparse
from collections import Counter
from itertools import product

from embodied_agent.exposure_confirmation import ExposureClockMap, ExposureConfirmation
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.post_alert_verification import PostAlertVerification
from embodied_agent.recovery_report import write_recovery_report


def run_case(onset, delay, response):
    mapping = ExposureClockMap(16, 0)
    initial = ExposureConfirmation(0, mapping)
    for tick in range(9):
        exposure = tick / 8
        initial.step(exposure, 16 + exposure, StampedPosition((0.64, 0, 0.04), 16 + exposure))
    supervisor = PostAlertVerification(initial.result(), 1, "camera", mapping)
    first_inside = None
    for tick in range(1, 49):
        exposure = 1 + tick / 8
        task = exposure + delay
        point = (0.64, 0, 0.04)
        after_alert = exposure >= onset + 0.375
        if onset <= exposure < onset + 0.375:
            point = (0.7, 0, 0.04)
        elif after_alert:
            if response == "outside" or (
                response == "transient_return" and exposure > onset + 0.375
            ):
                point = (0.7, 0, 0.04)
            elif response == "missing":
                point = None
            elif response == "jitter":
                point = (0.63 if tick % 2 else 0.65, 0, 0.04)
        frame = None if point is None else StampedPosition(point, 16 + exposure)
        # Deliberately weak comparator; it is not the previous production policy.
        if (
            first_inside is None
            and supervisor.status == "rechecking"
            and point is not None
            and abs(point[0] - 0.64) < 0.05
        ):
            first_inside = task
        supervisor.step(
            task,
            16 + task + (0.125 if after_alert and response == "clock_drift" else 0),
            frame,
            session_id="replacement" if after_alert and response == "session_change" else "camera",
            stop=after_alert and response == "stop",
        )
        if supervisor.status not in {"monitoring", "rechecking"}:
            break
    expected = (
        "reconfirmed"
        if response == "stable_return"
        else "cancelled"
        if response == "stop"
        else "invalidated"
        if response in {"session_change", "clock_drift"}
        else "needs_review"
    )
    result = supervisor.result()
    return {
        "onset": onset,
        "delivery_delay": delay,
        "response": response,
        "expected_status": expected,
        "actual_status": result["status"],
        "match": expected == result["status"],
        "single_frame_clear_time": first_inside,
        "single_frame_premature_clear": response in {"transient_return", "jitter"}
        and first_inside is not None,
        "result": result,
    }


def evaluate():
    rows = [
        run_case(*args)
        for args in product(
            (1.25, 1.5, 2.25),
            (0, 0.0625, 0.125),
            (
                "stable_return",
                "transient_return",
                "outside",
                "missing",
                "jitter",
                "session_change",
                "clock_drift",
                "stop",
            ),
        )
    ]
    return {
        "schema_version": 1,
        "test_kind": "synthetic_post_alert_verification",
        "cases": len(rows),
        "mismatches": sum(not row["match"] for row in rows),
        "status_counts": dict(Counter(row["actual_status"] for row in rows)),
        "single_frame_premature_clears": sum(row["single_frame_premature_clear"] for row in rows),
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate()
    write_recovery_report(args.output, report)
    print({key: value for key, value in report.items() if key != "results"})
    return int(report["mismatches"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
