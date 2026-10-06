"""Paired axis versus oriented-size recovery comparison with unchanged fixtures."""

import json
from pathlib import Path

import numpy as np

from embodied_agent.correction_data import payload_digest
from embodied_agent.memory_policy import digest


def main():
    paths = [Path(f"docs/evaluations/contact-{name}-v1.json") for name in ("stable", "oriented")]
    reports = [json.loads(p.read_text()) for p in paths]
    canonical, manifests = [], []
    for report, mode in zip(reports, ("axis", "oriented"), strict=True):
        if (
            report["status"] != "complete"
            or payload_digest(report["dataset"]) != report["dataset_sha256"]
        ):
            raise ValueError("incomplete or changed dataset")
        config = dict(report["config"])
        if config.pop("perception_size_mode", "axis") != mode:
            raise ValueError("unexpected perception mode")
        config.pop("description")
        canonical.append(config)
        manifests.append(json.loads((Path(report["dataset"]) / "recording.json").read_text()))
    if canonical[0] != canonical[1]:
        raise ValueError("unpaired collection configuration")
    rows = []
    for before, after in zip(*[m["episodes"] for m in manifests], strict=True):
        for field in ("qpos", "qvel", "ctrl"):
            if not np.array_equal(before["takeover"][field], after["takeover"][field]):
                raise ValueError("takeover states differ")
        if before["contact_decision"] != after["contact_decision"]:
            raise ValueError("contact decisions differ")
        rows.append(
            {
                "tick": after["takeover"]["tick"],
                "axis_success": before["success"],
                "oriented_success": after["success"],
                "reason": after["reason"],
                "supervised_frames": after["supervised_frames"],
                "detection": after["detection"],
            }
        )
    result = {
        "status": "complete",
        "source_reports": {str(p): digest(p) for p in paths},
        "episodes": rows,
        "axis_successes": sum(r["axis_success"] for r in rows),
        "oriented_successes": sum(r["oriented_success"] for r in rows),
        "supervised_frames": reports[1]["supervised_frames"],
        "training_performed": False,
        "test_executed": False,
    }
    Path("docs/evaluations/contact-oriented-comparison-v1.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
