"""Fixed error-budget sweep on saved observations; truth used only for scoring."""

import json
from pathlib import Path

import numpy as np

from embodied_agent.conservative_geometry import envelope
from embodied_agent.memory_policy import digest
from embodied_agent.visible_geometry import estimate, point_cloud


def analyze(path):
    with np.load(path, allow_pickle=False) as d:
        rgb = d["rgb"]
        args = (rgb, d["depth"], d["camera_position"], d["camera_rotation"], d["fovy"])
        visible = estimate(*args)
        cloud, valid = point_cloud(*args)
        r, g, b = rgb.astype(float).transpose(2, 0, 1)
        points = cloud[valid & (r > 70) & (r > 1.6 * g) & (r > 1.4 * b)]
    return {
        "source": str(path),
        "source_sha256": digest(path),
        "visible_x_gap_m": visible["visible_x_gap_m"],
        "bounds": [
            envelope(points, 0.05, error, visible["visible_wall_x"], error)
            for error in (0.0, 0.001, 0.003, 0.005)
        ],
    }


def main():
    source = json.loads(Path("docs/evaluations/visible-geometry-v1.json").read_text())
    rows = []
    for case in source["cases"]:
        path = Path(case["source"])
        if digest(path) != case["sha256"]:
            raise ValueError("source observation changed")
        rows.append(analyze(path))
    path = Path("outputs/evaluations/bounded-wait-v1/wait-100.npz")
    if digest(path) != source["four_second_contact_case"]["source_sha256"]:
        raise ValueError("contact observation changed")
    contact = analyze(path)
    truth = json.loads(Path("docs/evaluations/support-reach-v1.json").read_text())["geometry"]
    for bound in contact["bounds"]:
        bound["scoring_only_true_aabb_contained"] = bool(
            np.all(np.array(bound["aabb_min"]) <= truth["cube_aabb_min"])
            and np.all(np.array(bound["aabb_max"]) >= truth["cube_aabb_max"])
        )
    report = {
        "status": "complete",
        "group": "train",
        "training_performed": False,
        "test_executed": False,
        "estimator_sha256": digest("src/embodied_agent/conservative_geometry.py"),
        "script_sha256": digest(__file__),
        "source_reports": {
            name: digest("docs/evaluations/" + name)
            for name in ("visible-geometry-v1.json", "support-reach-v1.json")
        },
        "cases": rows,
        "known_contact_case": contact,
        "error_budget_status": "0/1/3/5 mm per-axis sensitivity assumptions, not measured or calibrated error guarantees",
        "limitations": [
            "fixed 50mm cube prior; no general shape completion",
            "all selected red points must correspond to the object within the assumed error",
            "nearest-wall assumption is unverified by a single view",
            "no contact action or runtime perception change",
        ],
    }
    Path("docs/evaluations/conservative-geometry-v1.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    for row in rows + [contact]:
        print(
            row["source"],
            [
                (
                    x["point_error_m"],
                    x["conditional_x_gap_lower_m"],
                    x.get("scoring_only_true_aabb_contained"),
                )
                for x in row["bounds"]
            ],
        )


if __name__ == "__main__":
    main()
