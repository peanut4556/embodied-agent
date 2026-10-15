"""Generate two explicitly synthetic pairing fixtures; contains truth only for generation."""

import copy
import json
from pathlib import Path

import numpy as np

from embodied_agent.reference_fit import predict


def main():
    root = Path("examples/external-joint")
    base = json.loads(Path("docs/evaluations/redundant-reference-v1.json").read_text())
    planes = [(np.array(p["normal"]), p["offset_m"]) for p in base["planes"]]
    truth = np.array(base["true_parameters"])
    policy = json.loads(Path("examples/external-observations/policy.json").read_text())
    external = json.loads(Path("examples/external-observations/observations.json").read_text())
    template = external["observations"][0]
    rays = np.array(
        [[x, y, -1.0] for x in np.linspace(-0.25, 0.25, 7) for y in np.linspace(-0.2, 0.2, 7)]
    )
    rng = np.random.default_rng(240000)
    clean = predict(rays, planes, truth)
    repeated = (
        clean + rng.normal(0, 0.00005, (128, 4, len(rays))) + rng.normal(0, 0.00005, (128, 4, 1))
    )
    fits = []
    observations = []
    for index, shift in enumerate((0.0, 0.008)):
        oid = f"joint_fixture_{index + 1}"
        camera = f"2026-10-14T10:00:00.{20 + 20 * index:03d}+08:00"
        observation = copy.deepcopy(template)
        observation.update(
            observation_id=oid, camera_time=camera, units="m", position=truth[:3].tolist()
        )
        observations.append(observation)
        moved = [(n, float(d + n @ np.array([shift, 0, 0]))) for n, d in planes]
        depths = (
            predict(rays, moved, truth)
            + rng.normal(0, 0.00005, (4, len(rays)))
            + rng.normal(0, 0.00005, (4, 1))
        )
        fits.append(
            {
                "fit_id": f"visual_fixture_{index + 1}",
                "observation_id": oid,
                "camera_time": camera,
                "depths_m": depths.tolist(),
            }
        )
    metadata = {
        key: policy[key]
        for key in (
            "data_kind",
            "camera_id",
            "frame_id",
            "clock_id",
            "reference_origin_id",
            "position_semantics",
            "nominal_camera_origin_m",
        )
    }
    visual = {
        "schema_version": 1,
        "metadata": metadata,
        "rays": rays.tolist(),
        "planes": base["planes"],
        "calibration": {
            "start_time": "2026-10-14T09:59:20+08:00",
            "end_time": "2026-10-14T09:59:50+08:00",
            "depths_m": repeated.tolist(),
        },
        "fits": fits,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "visual.json").write_text(
        json.dumps(visual, separators=(",", ":"), allow_nan=False) + "\n"
    )
    (root / "observations.json").write_text(
        json.dumps({"schema_version": 1, "observations": observations}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
