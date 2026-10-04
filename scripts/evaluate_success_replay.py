"""Frozen successful rehearsal comparison, with parent regression reference."""

import copy
import json
from pathlib import Path

import numpy as np
import torch

from embodied_agent.memory_evaluation import save_preview, verify_physics, verify_run
from embodied_agent.memory_policy import MemoryPolicy, digest
from embodied_agent.reactive_evaluation import run_case


def main():
    output = Path("outputs/evaluations/success-replay-v1")
    if output.exists():
        raise FileExistsError(output)
    roots = {a: Path(f"outputs/models/success-replay-{a}") for a in ("control", "replay")}
    verified = {a: verify_run(r) for a, r in roots.items()}
    settings = []
    for arm, weight in (("control", 0.0), ("replay", 1.0)):
        config = copy.deepcopy(verified[arm][1])
        if config["candidate_epochs"] != [300] or config["success_replay"]["weight"] != weight:
            raise ValueError("budget or replay weight differs")
        del config["success_replay"]["weight"]
        settings.append(config)
    if settings[0] != settings[1]:
        raise ValueError("unpaired experiment configuration")
    for key in (
        "source_sha256",
        "pretrained_weights_sha256",
        "train_supervised_frames",
        "validation_supervised_frames",
    ):
        if verified["control"][0][key] != verified["replay"][0][key]:
            raise ValueError("training provenance differs")
    config = settings[0]
    path = config["success_replay"]["path"]
    if digest(path) != config["success_replay"]["sha256"]:
        raise ValueError("replay changed")
    pack = json.loads(Path(path).read_text())
    roots = {
        "parent": Path("outputs/models/state-distill-distilled/epoch-300"),
        **{a: r / "epoch-300" for a, r in roots.items()},
    }
    output.mkdir(parents=True)
    report = {
        "status": "running",
        "test_executed": False,
        "training": {a: v[0] for a, v in verified.items()},
        "script_sha256": digest(__file__),
        "results": [],
        "replay_action_mse": {},
    }
    try:
        for arm, root in roots.items():
            policy = MemoryPolicy(root)
            verify_physics(policy)
            expected_parent = (
                policy.metadata["weights_sha256"]
                if arm == "parent"
                else policy.metadata["pretrained_weights_sha256"]
            )
            if expected_parent != pack["weights_sha256"]:
                raise ValueError("parent provenance differs")
            x = torch.from_numpy(
                (
                    (np.asarray(pack["observations"], dtype=np.float32) - policy.mean)
                    / policy.scale
                ).astype(np.float32)
            )[None]
            with torch.no_grad():
                report["replay_action_mse"][arm] = float(
                    (policy.model(x)[0] - torch.tensor(pack["normalized_targets"])[None])
                    .square()
                    .mean()
                )
            cases = [("validation", c) for c in config["validation_cases"]] + [
                ("training_regression", {"name": "training-completed-302", "x": 0.302})
            ]
            for group, case in cases:
                result, frames = run_case(policy, case, "memory", config["max_seconds"])
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
                (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        report["scores"] = {
            a: {
                g: {
                    "tasks": len(rows),
                    "verified_pick_place": sum(r["quality"]["verified_pick_place"] for r in rows),
                    "stable_grasp": sum(r["quality"]["stable_grasp_observed"] for r in rows),
                }
                for g in ("validation", "training_regression")
                for rows in [[r for r in report["results"] if r["arm"] == a and r["group"] == g]]
            }
            for a in roots
        }
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
