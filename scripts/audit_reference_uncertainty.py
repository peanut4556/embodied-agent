"""Replay all fixed redundant-plane fits with a bounded-noise diagnostic threshold."""

import json
from pathlib import Path

import numpy as np

from embodied_agent.memory_policy import digest
from embodied_agent.reference_fit import predict
from embodied_agent.reference_uncertainty import check_uncertain_reference


def main():
    baseline_path = Path("docs/evaluations/redundant-reference-v1.json")
    baseline = json.loads(baseline_path.read_text())
    fit_source = json.loads(Path("docs/evaluations/reference-fit-v1.json").read_text())
    if (
        digest("docs/evaluations/reference-fit-v1.json") != baseline["baseline_sha256"]
        or digest("src/embodied_agent/reference_fit.py") != fit_source["implementation_sha256"]
        or digest("src/embodied_agent/multiplane_reference.py") != fit_source["geometry_sha256"]
    ):
        raise ValueError("baseline fit implementation changed")
    planes = [(np.array(p["normal"]), p["offset_m"]) for p in baseline["planes"]]
    cases = []
    for c in baseline["cases"]:
        ex, ey = baseline["parameters"]["patch_half_widths"][c["patch"]]
        rays = np.array(
            [[x, y, -1.0] for x in np.linspace(-ex, ex, 21) for y in np.linspace(-ey, ey, 21)]
        )
        actual = [
            (n, d + delta) for (n, d), delta in zip(planes, c["plane_offset_errors_m"], strict=True)
        ]
        b = c["noise_bound_m"]
        observed = predict(rays, actual, baseline["true_parameters"]) + np.random.default_rng(
            c["seed"]
        ).uniform(-b, b, (4, len(rays)))
        fit = c["heldout"]["fit"]
        if fit["status"] == "converged":
            residual = observed[3] - predict(rays, planes[3:], fit["parameters"])[0]
            np.testing.assert_allclose(
                np.quantile(np.abs(residual), 0.95),
                c["heldout"]["heldout_p95_m"],
                atol=1e-12,
                rtol=0,
            )
        result = check_uncertain_reference(rays, planes, observed, fit, b)
        cases.append(
            {k: c[k] for k in ("patch", "noise_bound_m", "seed", "plane_error")}
            | {"original_status": c["heldout"]["status"], "propagated": result}
        )
    summary = {}
    for label in dict.fromkeys(c["plane_error"] for c in cases):
        rows = [c for c in cases if c["plane_error"] == label]
        summary[label] = {
            "cases": len(rows),
            "original": {
                s: sum(c["original_status"] == s for c in rows)
                for s in ("alarm", "no_alarm", "unknown")
            },
            "propagated": {
                s: sum(c["propagated"]["status"] == s for c in rows)
                for s in ("alarm", "no_alarm", "unknown")
            },
        }
    report = {
        "status": "complete",
        "script_sha256": digest(__file__),
        "implementation_sha256": digest("src/embodied_agent/reference_uncertainty.py"),
        "baseline_sha256": digest(baseline_path),
        "summary": summary,
        "cases": cases,
        "parameters": {
            "threshold_floor_m": 0.003,
            "percentile": 0.95,
            "relative_rank_threshold": 1e-7,
        },
        "training_performed": False,
        "test_executed": False,
        "recovery_executed": False,
        "calibration_certified": False,
        "limitations": [
            "synthetic noise bound supplied as an explicit assumption, not estimated from heldout data",
            "first-order propagation only; no nonlinear remainder or plane-location uncertainty bound",
            "worst-case pointwise bounded errors may be correlated; not a probabilistic confidence interval",
            "P95 rule permits sparse threshold violations; no all-pixel guarantee",
            "common reference translation remains unidentifiable; no correction applied",
        ],
    }
    Path("docs/evaluations/reference-uncertainty-v1.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
