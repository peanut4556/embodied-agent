"""Compare final BC and local-state-distilled policies on unchanged development cases."""

import copy
import json
from pathlib import Path

from embodied_agent.memory_evaluation import save_preview, verify_physics, verify_run
from embodied_agent.memory_policy import MemoryPolicy, digest
from embodied_agent.reactive_evaluation import run_case


def main():
    output = Path("outputs/evaluations/state-distill-v1")
    if output.exists():
        raise FileExistsError(output)
    arms = ("control", "distilled")
    roots = {a: Path(f"outputs/models/state-distill-{a}") for a in arms}
    runs = {a: verify_run(roots[a]) for a in arms}
    configs = [copy.deepcopy(runs[a][1]) for a in arms]
    for c, w in zip(configs, (0.0, 0.1), strict=True):
        if c["candidate_epochs"] != [300] or c["local_targets"]["weight"] != w:
            raise ValueError("fixed budget or distillation weight changed")
        del c["local_targets"]["weight"]
    if configs[0] != configs[1]:
        raise ValueError("paired configs differ beyond auxiliary loss weight")
    for key in (
        "source_sha256",
        "pretrained_weights_sha256",
        "train_supervised_frames",
        "validation_supervised_frames",
    ):
        if runs["control"][0][key] != runs["distilled"][0][key]:
            raise ValueError("paired lineage or data differ")
    output.mkdir(parents=True)
    report = {
        "status": "running",
        "script_sha256": digest(__file__),
        "test_executed": False,
        "training": {a: runs[a][0] for a in arms},
        "results": [],
    }
    try:
        for arm in arms:
            policy = MemoryPolicy(roots[arm] / "epoch-300")
            verify_physics(policy)
            for case in configs[0]["validation_cases"]:
                result, frames = run_case(policy, case, "memory", configs[0]["max_seconds"])
                result = {"arm": arm, "weights_sha256": policy.metadata["weights_sha256"], **result}
                report["results"].append(result)
                print(json.dumps(result), flush=True)
                save_preview(output, f"{arm}-{case['name']}", frames, 25)
        report["scores"] = {}
        for arm in arms:
            rows = [r for r in report["results"] if r["arm"] == arm]
            report["scores"][arm] = {
                "tasks": len(rows),
                "verified_pick_place": sum(r["quality"]["verified_pick_place"] for r in rows),
                "endpoint_success": sum(r["success"] for r in rows),
                "stable_grasp": sum(r["quality"]["stable_grasp_observed"] for r in rows),
                "sustained_lift": sum(r["quality"]["sustained_lift"] for r in rows),
            }
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
