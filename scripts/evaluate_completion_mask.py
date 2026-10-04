"""Frozen three-arm future-supervision comparison; normal-input development evaluation."""

import copy
import json
from pathlib import Path

from embodied_agent.memory_evaluation import save_preview, verify_physics, verify_run
from embodied_agent.memory_policy import MemoryPolicy, digest
from embodied_agent.reactive_evaluation import run_case


def main():
    output = Path("outputs/evaluations/completion-mask-v1")
    if output.exists():
        raise FileExistsError(output)
    arms = {"bc": (0.0, False), "unmasked": (0.1, False), "masked": (0.1, True)}
    roots = {a: Path(f"outputs/models/completion-{a}") for a in arms}
    verified = {a: verify_run(roots[a]) for a in arms}
    canonical = None
    for arm, (weight, mask) in arms.items():
        config = copy.deepcopy(verified[arm][1])
        if (
            config["candidate_epochs"] != [300]
            or config["future_targets"]["weight"] != weight
            or config["future_targets"]["completion_mask"] != mask
        ):
            raise ValueError("frozen training conditions differ")
        del config["future_targets"]["weight"], config["future_targets"]["completion_mask"]
        if canonical is not None and config != canonical:
            raise ValueError("unpaired experiment settings")
        canonical = config
        for key in (
            "source_sha256",
            "pretrained_weights_sha256",
            "train_supervised_frames",
            "validation_supervised_frames",
        ):
            if verified[arm][0][key] != verified["bc"][0][key]:
                raise ValueError("data or parent differs")
    output.mkdir(parents=True)
    report = {
        "status": "running",
        "test_executed": False,
        "script_sha256": digest(__file__),
        "training": {a: verified[a][0] for a in arms},
        "results": [],
    }
    try:
        for arm in arms:
            policy = MemoryPolicy(roots[arm] / "epoch-300")
            verify_physics(policy)
            if (
                digest(roots[arm] / "epoch-300/future-head.pt")
                != policy.metadata["future_head_sha256"]
            ):
                raise ValueError("auxiliary artifact changed")
            cases = [("validation", c) for c in canonical["validation_cases"]] + [
                ("training_regression", {"name": "training-completed-302", "x": 0.302})
            ]
            for group, case in cases:
                result, frames = run_case(policy, case, "memory", canonical["max_seconds"])
                report["results"].append(
                    {
                        "arm": arm,
                        "group": group,
                        "weights_sha256": policy.metadata["weights_sha256"],
                        **result,
                    }
                )
                save_preview(output, f"{arm}-{case['name']}", frames, 25)
                print(json.dumps(report["results"][-1]), flush=True)
        report["scores"] = {}
        for arm in arms:
            report["scores"][arm] = {}
            for group in ("validation", "training_regression"):
                rows = [r for r in report["results"] if r["arm"] == arm and r["group"] == group]
                report["scores"][arm][group] = {
                    "tasks": len(rows),
                    "verified_pick_place": sum(r["quality"]["verified_pick_place"] for r in rows),
                    "endpoint_success": sum(r["success"] for r in rows),
                    "stable_grasp": sum(r["quality"]["stable_grasp_observed"] for r in rows),
                }
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
