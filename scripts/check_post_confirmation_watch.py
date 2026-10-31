"""Synthetic confirmed release followed by bounded independent visual monitoring."""

import argparse

from embodied_agent.exposure_confirmation import ExposureClockMap, ExposureConfirmation
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.post_confirmation_watch import PostConfirmationWatch
from embodied_agent.recovery_report import write_recovery_report


def scenarios():
    mapping = ExposureClockMap(16, 0)
    initial = ExposureConfirmation(0, mapping)
    for index in range(5):
        task = index / 4
        initial.step(task + 1 / 64, 16 + task + 1 / 64, StampedPosition((0.64, 0, 0.04), 16 + task))
    reports = {}
    for name in (
        "inside",
        "transient",
        "outside",
        "missing",
        "stale",
        "session_change",
        "stop",
        "return_inside",
    ):
        watch = PostConfirmationWatch(initial.result(), initial.last_delivery, "camera", mapping)
        frames = []
        for index in range(5):
            exposure = 1.25 + index / 4
            task = exposure + 1 / 64
            point = (0.64, 0, 0.04)
            if (
                name == "outside"
                or (name == "transient" and index == 1)
                or (name == "return_inside" and index < 2)
            ):
                point = (0.7, 0, 0.04)
            stamp = 16 + exposure if name != "stale" else 17
            observation = None if name == "missing" else StampedPosition(point, stamp)
            session = "replacement" if name == "session_change" and index >= 1 else "camera"
            watch.step(
                task, 16 + task, observation, session_id=session, stop=name == "stop" and index == 1
            )
            frames.append(
                {
                    "task_time": task,
                    "point": point if observation else None,
                    "captured_at": stamp if observation else None,
                    "result": watch.result(),
                }
            )
            if watch.status != "monitoring":
                break
        before_end = watch.result()
        watch.step(watch.deadline, 16 + watch.deadline, None, session_id="camera")
        reports[name] = {"frames": frames, "before_end": before_end, "final": watch.result()}
    return {
        "schema_version": 1,
        "test_kind": "synthetic_stamped_positions",
        "original_confirmation": initial.result(),
        "cases": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = scenarios()
    write_recovery_report(args.output, report)
    print(f"Wrote {len(report['cases'])} post-confirmation scenarios")


if __name__ == "__main__":
    main()
