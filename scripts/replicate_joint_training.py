"""Fixed-seed replication; do not select the best random seed."""

import argparse
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", "outputs/hf-cache")

from embodied_agent.correction_finetune import train
from embodied_agent.memory_policy import MemoryPolicy, digest
from scripts.evaluate_joint_training import evaluate

PLAN = Path("config/joint-seeds-experiment.json")
ROOT = Path("outputs/models/joint-seeds-v1")
EVAL = Path("outputs/evaluations/joint-seeds-v1")


def same_parameters(first, second):
    import torch

    a, b = first.model.state_dict(), second.model.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def prepare(plan):
    if ROOT.exists():
        raise FileExistsError(ROOT)
    ROOT.mkdir(parents=True)
    (ROOT / "plan.json").write_bytes(PLAN.read_bytes())
    for seed in plan["new_seeds"]:
        for arm in ("control", "augmented"):
            config = json.loads(Path(f"config/joint-training-{arm}.json").read_text())
            config["seed"] = seed
            (ROOT / f"{seed}-{arm}.json").write_text(json.dumps(config, indent=2) + "\n")


def run_training(plan):
    prepare(plan)
    for seed in plan["new_seeds"]:
        for arm in ("control", "augmented"):
            train(
                "outputs/datasets/reactive-development-v2",
                ["outputs/datasets/corrections-early-v1"],
                "outputs/models/memory-correction-v1/epoch-600",
                ROOT / f"{seed}-{arm}.json",
                ROOT / f"{seed}-{arm}",
            )


def run_evaluation(plan):
    if EVAL.exists():
        raise FileExistsError(EVAL)
    if digest(ROOT / "plan.json") != digest(PLAN):
        raise ValueError("replication plan changed")
    EVAL.mkdir(parents=True)
    prior_path = Path("outputs/evaluations/joint-training-v1/evaluation.json")
    prior = json.loads(prior_path.read_text())
    if prior["status"] != "complete" or len(prior["results"]) != 18:
        raise ValueError("complete original paired experiment required")
    report = {
        "status": "running",
        "plan": plan,
        "plan_sha256": digest(PLAN),
        "previous_evaluation_sha256": digest(prior_path),
        "test_executed": False,
        "primary_metric": plan["primary_metric"],
        "seeds": {str(plan["previous_seed"]): prior},
        "control_parameter_identity": {},
    }
    try:
        original = MemoryPolicy("outputs/models/joint-training-control/epoch-300")
        for seed in plan["new_seeds"]:
            for arm in ("control", "augmented"):
                current = json.loads((ROOT / f"{seed}-{arm}" / "experiment.json").read_text())
                base = json.loads(Path(f"config/joint-training-{arm}.json").read_text())
                if current["seed"] != seed or {k: v for k, v in current.items() if k != "seed"} != {
                    k: v for k, v in base.items() if k != "seed"
                }:
                    raise ValueError("replication differs beyond random seed")
            control = MemoryPolicy(ROOT / f"{seed}-control" / "epoch-300")
            report["control_parameter_identity"][str(seed)] = same_parameters(original, control)
            report["seeds"][str(seed)] = evaluate(
                ROOT / f"{seed}-control", ROOT / f"{seed}-augmented", EVAL / f"seed-{seed}"
            )
            (EVAL / "replication.json").write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (EVAL / "replication.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if plan["previous_seed"] != 73 or plan["new_seeds"] != [101, 211]:
        raise ValueError("frozen seed budget differs")
    (run_training if args.mode == "train" else run_evaluation)(plan)
