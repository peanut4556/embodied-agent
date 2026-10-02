"""Frozen paired normal and single-pulse development evaluation, final epoch only."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from embodied_agent.memory_evaluation import save_preview, verify_physics, verify_run
from embodied_agent.memory_policy import MemoryPolicy, digest
from embodied_agent.reactive_evaluation import run_case
from scripts.audit_joint_sensitivity import PulseSession


def evaluate(control_root, augmented_root, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    runs = {"control": Path(control_root), "augmented": Path(augmented_root)}
    verified = {arm: verify_run(root) for arm, root in runs.items()}
    first = verified["control"][1]
    second = verified["augmented"][1]
    if {k: v for k, v in first.items() if k != "joint_augmentation"} != {
        k: v for k, v in second.items() if k != "joint_augmentation"
    }:
        raise ValueError("paired experiment differs beyond augmentation")
    for key in (
        "source_sha256",
        "pretrained_weights_sha256",
        "train_supervised_frames",
        "validation_supervised_frames",
    ):
        if verified["control"][0][key] != verified["augmented"][0][key]:
            raise ValueError("paired training inputs or parent differ")
    if (
        first["candidate_epochs"] != [300]
        or first["joint_augmentation"] is not None
        or second["joint_augmentation"]
        != {"probability": 0.1, "arm_radians": 0.01, "finger_meters": 0.001}
    ):
        raise ValueError("frozen budget or augmentation differs")
    output.mkdir(parents=True)
    report = {
        "status": "running",
        "test_executed": False,
        "console_policy_replaced": False,
        "script_sha256": digest(__file__),
        "training": {arm: verified[arm][0] for arm in runs},
        "conditions": {"normal": 0.0, "negative_wrist_pulse": -0.01, "positive_wrist_pulse": 0.01},
        "pulse_tick": 12,
        "joint": 2,
        "results": [],
    }
    try:
        for arm, root in runs.items():
            policy = MemoryPolicy(root / "epoch-300")
            verify_physics(policy)
            for condition, amount in report["conditions"].items():
                wrapped = SimpleNamespace(
                    metadata=policy.metadata,
                    bounds=policy.bounds,
                    session=lambda p=policy, a=amount: PulseSession(p, 2, a, tick=12),
                )
                for case in first["validation_cases"]:
                    result, frames = run_case(wrapped, case, "memory", first["max_seconds"])
                    report["results"].append(
                        {
                            "arm": arm,
                            "condition": condition,
                            "weights_sha256": policy.metadata["weights_sha256"],
                            **result,
                        }
                    )
                    save_preview(output, f"{arm}-{condition}-{case['name']}", frames, 25)
                    print(json.dumps(report["results"][-1]), flush=True)
                    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        report["scores"] = {}
        for arm in runs:
            report["scores"][arm] = {}
            for condition in report["conditions"]:
                rows = [
                    r for r in report["results"] if r["arm"] == arm and r["condition"] == condition
                ]
                report["scores"][arm][condition] = {
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

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", default="outputs/models/joint-training-control")
    parser.add_argument("--augmented", default="outputs/models/joint-training-augmented")
    parser.add_argument("--output", default="outputs/evaluations/joint-training-v1")
    args = parser.parse_args()
    evaluate(args.control, args.augmented, args.output)


if __name__ == "__main__":
    main()
