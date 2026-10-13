"""Replay all prior narrow stalls and explicit controls; instrument without relabeling."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from embodied_agent.memory_policy import digest
from embodied_agent.reference_fit import fit_reference, predict
from embodied_agent.reference_fit_diagnostics import trace_reference_fit
from scripts.audit_exposure_batches import distribution, draw


def select_cases(source, control_count):
    by_condition = {
        (c["patch"], c["noise_model"]): {
            case["seed"]: case for batch in c["batches"] for case in batch["cases"]
        }
        for c in source["conditions"]
    }
    selected = []
    for noise in ("independent", "global_shared", "exposure_shared"):
        narrow = by_condition[("narrow", noise)]
        for seed, case in narrow.items():
            if case["fit_status"] == "stalled":
                selected.extend(
                    [
                        {
                            "group": "narrow_stalled",
                            "patch": "narrow",
                            "noise_model": noise,
                            "seed": seed,
                            "source_status": "stalled",
                        },
                        {
                            "group": "wide_paired",
                            "patch": "wide",
                            "noise_model": noise,
                            "seed": seed,
                            "source_status": by_condition[("wide", noise)][seed]["fit_status"],
                        },
                    ]
                )
        controls = [seed for seed, case in narrow.items() if case["fit_status"] == "converged"][
            :control_count
        ]
        selected.extend(
            {
                "group": "narrow_converged_control",
                "patch": "narrow",
                "noise_model": noise,
                "seed": seed,
                "source_status": "converged",
            }
            for seed in controls
        )
    return selected


def summarize(cases):
    result = []
    for group in ("narrow_stalled", "wide_paired", "narrow_converged_control"):
        for noise in ("independent", "global_shared", "exposure_shared"):
            rows = [c for c in cases if c["group"] == group and c["noise_model"] == noise]
            trials = [t for c in rows for t in c["diagnostic"]["trace"][-1]["trials"]]
            terminal = [c["diagnostic"]["terminal"] for c in rows]
            result.append(
                {
                    "group": group,
                    "noise_model": noise,
                    "cases": len(rows),
                    "statuses": dict(Counter(c["diagnostic"]["fit"]["status"] for c in rows)),
                    "terminal_trial_outcomes": dict(Counter(t["outcome"] for t in trials)),
                    "terminal_metrics": {
                        key: distribution([t[key] for t in terminal if t[key] is not None])
                        for key in (
                            "condition_number",
                            "scaled_step_norm",
                            "max_residual_column_cosine",
                            "scaled_gradient_infinity_norm_m2",
                            "linear_reduction_over_eps_sse",
                        )
                    },
                    "max_parameter_bound_fraction": distribution(
                        [c["diagnostic"]["max_parameter_bound_fraction"] for c in rows]
                    ),
                    "cases_with_stable_decrease_rejected": sum(
                        any(
                            t["outcome"] == "objective_increase"
                            and t.get("stable_sse_change_m2", 0) < 0
                            for t in c["diagnostic"]["trace"][-1]["trials"]
                        )
                        for c in rows
                    ),
                    "cases_linear_gain_below_one_eps_sse": sum(
                        t["linear_reduction_over_eps_sse"] is not None
                        and t["linear_reduction_over_eps_sse"] < 1
                        for t in terminal
                    ),
                    "fd_step_norms": {
                        str(h): distribution(
                            [
                                d["scaled_step_norm"]
                                for c in rows
                                for d in c["diagnostic"]["terminal_fd_sensitivity"]
                                if d["step"] == h
                            ]
                        )
                        for h in (1e-7, 1e-6, 1e-5)
                    },
                }
            )
    return result


def main():
    config_path = Path("config/fit-stall-diagnostics.json")
    config = json.loads(config_path.read_text())
    source_path = Path(config["source_report"])
    source = json.loads(source_path.read_text())
    for path, expected in source["sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"source implementation changed: {path}")
    baseline_path = Path("docs/evaluations/redundant-reference-v1.json")
    baseline = json.loads(baseline_path.read_text())
    planes = [(np.array(p["normal"]), p["offset_m"]) for p in baseline["planes"]]
    truth = np.array(baseline["true_parameters"])
    geometries = {}
    for patch, (ex, ey) in (("wide", (0.25, 0.20)), ("narrow", (0.025, 0.02))):
        rays = np.array(
            [[x, y, -1.0] for x in np.linspace(-ex, ex, 21) for y in np.linspace(-ey, ey, 21)]
        )
        geometries[patch] = (rays, predict(rays, planes, truth))
    cases = []
    for index, selected in enumerate(select_cases(source, config["converged_controls_per_noise"])):
        rays, clean = geometries[selected["patch"]]
        e = 0.001 if selected["noise_model"] == "exposure_shared" else 0.0
        g = 0.001 if selected["noise_model"] == "global_shared" else 0.0
        observed = draw(clean, selected["seed"], e, g)[:3]
        original = fit_reference(rays, planes[:3], observed)
        diagnostic = trace_reference_fit(rays, planes[:3], observed)
        if original != diagnostic["fit"] or original["status"] != selected["source_status"]:
            raise AssertionError(f"replay mismatch: {selected}")
        cases.append(
            {
                **selected,
                "observations_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
                "original_matches_trace": True,
                "diagnostic": diagnostic,
            }
        )
        if (index + 1) % 256 == 0:
            print(f"{index + 1} replays matched", flush=True)
    paths = [
        Path(__file__),
        config_path,
        source_path,
        baseline_path,
        Path("scripts/audit_exposure_batches.py"),
        *[
            Path(f"src/embodied_agent/{n}.py")
            for n in ("reference_fit_diagnostics", "reference_fit", "multiplane_reference")
        ],
    ]
    report = {
        "status": "complete",
        "experiment": "offline_narrow_fit_stall_diagnostics",
        "config": config,
        "cases": cases,
        "summary": summarize(cases),
        "sha256": {str(p): digest(p) for p in paths},
        "production_solver_modified": False,
        "unknowns_reclassified": False,
        "training_performed": False,
        "correction_applied": False,
        "limitations": [
            "selected replay of previous cases, not independent validation",
            "first 64 converged controls per noise are descriptive, not a random sample",
            "epsilon*SSE is a numerical scale, not a rigorous error bound",
            "stable objective differences still use rounded float64 geometry predictions",
            "finite difference probes are diagnostic only; no new convergence tolerance chosen",
            "full local rank does not establish global identifiability or accurate calibration",
        ],
    }
    Path("docs/evaluations/fit-stalls-v1.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    for row in report["summary"]:
        print(row["group"], row["noise_model"], row["cases"], row["terminal_metrics"], flush=True)


if __name__ == "__main__":
    main()
