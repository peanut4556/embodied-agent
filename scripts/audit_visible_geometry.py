"""Compare visible RGB-D geometry with held-out simulator scoring quantities."""

import json
from pathlib import Path

import numpy as np

from embodied_agent.memory_policy import digest
from embodied_agent.visible_geometry import estimate


def main():
    report = {
        "status": "complete",
        "group": "train",
        "training_performed": False,
        "test_executed": False,
        "cases": [],
        "estimator_sha256": digest("src/embodied_agent/visible_geometry.py"),
    }
    source = json.loads(Path("docs/evaluations/contact-vision-v1.json").read_text())
    for case in source["cases"]:
        path = Path(
            f"outputs/evaluations/contact-vision-v1/episode-{case['episode']}-tick-{case['decision_tick']}-rgbd.npz"
        )
        if digest(path) != case["rgbd_sha256"]:
            raise ValueError("source image changed")
        with np.load(path, allow_pickle=False) as d:
            estimated = estimate(
                d["rgb"], d["depth"], d["camera_position"], d["camera_rotation"], d["fovy"]
            )
        report["cases"].append(
            {
                "source": str(path),
                "sha256": digest(path),
                "estimate": estimated,
                "scoring_only": {
                    "nearest_face_tilt_degrees": case["truth_diagnostic_only"][
                        "nearest_face_tilt_degrees"
                    ],
                    "dominant_plane_tilt_difference_degrees": estimated["dominant_visible_plane"][
                        "tilt_degrees"
                    ]
                    - case["truth_diagnostic_only"]["nearest_face_tilt_degrees"],
                },
            }
        )
    wait = json.loads(Path("docs/evaluations/bounded-wait-v1.json").read_text())
    path = Path("outputs/evaluations/bounded-wait-v1/wait-100.npz")
    if digest(path) != wait["observations"][-1]["rgbd_sha256"]:
        raise ValueError("wait observation changed")
    with np.load(path, allow_pickle=False) as d:
        estimated = estimate(
            d["rgb"], d["depth"], d["camera_position"], d["camera_rotation"], d["fovy"]
        )
    support = json.loads(Path("docs/evaluations/support-reach-v1.json").read_text())
    if support["source_sha256"] != wait["source_sha256"] or not np.allclose(
        support["support"][-1]["xyz"],
        wait["observations"][-1]["truth_diagnostic_only"]["xyz"],
        atol=1e-10,
        rtol=0,
    ):
        raise ValueError("scoring state does not match saved observation")
    truth = support["geometry"]
    report["four_second_contact_case"] = {
        "estimate": estimated,
        "scoring_only": {
            "cube_aabb_max": truth["cube_aabb_max"],
            "visible_max_x_error_m": estimated["visible_aabb_max"][0] - truth["cube_aabb_max"][0],
            "tray_left_surface_x": 0.54,
            "true_signed_x_gap_m": 0.54 - truth["cube_aabb_max"][0],
        },
        "source_sha256": digest(path),
    }
    report["source_reports"] = {
        name: digest("docs/evaluations/" + name)
        for name in ("contact-vision-v1.json", "bounded-wait-v1.json", "support-reach-v1.json")
    }
    report["script_sha256"] = digest(__file__)
    report["limitations"] = [
        "four snapshots from one training scenario family, not an independent test set",
        "dominant visible plane is not a unique full cube pose",
        "red/teal color and right-side height/Y-window obstacle priors are fixture-specific",
        "positive visible gap cannot establish absence of contact; occluded surfaces unresolved",
        "no recovery action, perception gate change or training",
    ]
    Path("docs/evaluations/visible-geometry-v1.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
