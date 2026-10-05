"""Recheck saved RGB-D inputs with diagnostic-only oriented top-face measurements."""

import json
from pathlib import Path

import numpy as np

from embodied_agent.memory_policy import digest
from embodied_agent.perception import locate_red_cube


def main():
    root = Path("outputs/evaluations/contact-vision-v1")
    report = json.loads((root / "audit.json").read_text())
    if report["status"] != "complete":
        raise ValueError("incomplete physical replay")
    for case in report["cases"]:
        path = root / f"episode-{case['episode']}-tick-{case['decision_tick']}-rgbd.npz"
        if digest(path) != case["rgbd_sha256"]:
            raise ValueError("RGB-D evidence changed")
        with np.load(path, allow_pickle=False) as data:
            diagnostics = {}
            try:
                detection = locate_red_cube(
                    data["rgb"],
                    data["depth"],
                    data["camera_position"],
                    data["camera_rotation"],
                    float(data["fovy"]),
                    diagnostics,
                )
                error = None
            except ValueError as exc:
                detection, error = None, str(exc)
        if detection != case["detection"] or error != case["error"]:
            raise ValueError("diagnostics changed perception result")
        case["diagnostics"] = diagnostics
    report["recheck_perception_sha256"] = digest("src/embodied_agent/perception.py")
    report["recheck_script_sha256"] = digest(__file__)
    report["limitations"] = [
        "three inspected training trajectories only",
        "oriented box is diagnostic and does not change acceptance",
        "truth orientation used only to explain failure, never supplied to detector",
        "visible segmentation is not a full occlusion fraction",
        "no training, threshold relaxation or controller change",
    ]
    Path("docs/evaluations/contact-vision-v1.json").write_text(json.dumps(report, indent=2) + "\n")
    for c in report["cases"]:
        print(c["episode"], c["diagnostics"])


if __name__ == "__main__":
    main()
