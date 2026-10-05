"""Check paired takeover provenance and compare complete expert outcomes."""

import json
from pathlib import Path

from embodied_agent.correction_data import payload_digest
from embodied_agent.memory_policy import digest


def main():
    paths = [
        Path("docs/evaluations/contact-recovery-v1.json"),
        Path("docs/evaluations/contact-stable-v1.json"),
    ]
    reports = [json.loads(p.read_text()) for p in paths]
    old, new = reports
    for report in reports:
        if (
            report["status"] != "complete"
            or payload_digest(report["dataset"]) != report["dataset_sha256"]
        ):
            raise ValueError("incomplete or changed source dataset")
    for key in ("cases", "split", "learner_weights_sha256", "reference_trace_sha256"):
        if old["config"][key] != new["config"][key]:
            raise ValueError("unpaired recovery experiments")
    path = new["config"]["reference_trace"]
    if digest(path) != new["config"]["reference_trace_sha256"]:
        raise ValueError("reference trace changed")
    trace = json.loads(Path(path).read_text())["trace"]
    rows = []
    for before, after in zip(old["episodes"], new["episodes"], strict=True):
        if (
            before["tick"] != after["tick"]
            or not before["takeover_matches"]
            or not after["takeover_matches"]
        ):
            raise ValueError("takeovers differ")
        count = 0
        for row in trace[: after["tick"] + 1]:
            count = count + 1 if row["holding"] else 0
        decision = after["contact_decision"]
        if (
            decision["mode"] != "stable-contact-v1"
            or decision["holding_ticks"] != count
            or decision["required_ticks"] != 3
            or decision["preserve_grasp"] != (count >= 3)
        ):
            raise ValueError("stability decision differs from actual causal contact history")
        rows.append(
            {
                "tick": after["tick"],
                "consecutive_holding_ticks": count,
                "previous_success": before["success"],
                "stable_success": after["success"],
                "supervised_frames": after["supervised_frames"],
                "reason": after["reason"],
            }
        )
    result = {
        "status": "complete",
        "source_reports": {str(p): digest(p) for p in paths},
        "episodes": rows,
        "previous_successes": sum(e["success"] for e in old["episodes"]),
        "stable_successes": sum(e["success"] for e in new["episodes"]),
        "previous_supervised_frames": old["supervised_frames"],
        "stable_supervised_frames": new["supervised_frames"],
        "training_performed": False,
        "test_executed": False,
    }
    Path("docs/evaluations/contact-expert-comparison-v1.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
