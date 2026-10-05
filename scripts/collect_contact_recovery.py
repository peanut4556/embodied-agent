"""Fixed train-only recovery branches around audited early closure/contact loss."""

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "outputs/hf-cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np

from embodied_agent.correction_data import payload_digest, record, training_sequences
from embodied_agent.memory_policy import MemoryPolicy, digest
from embodied_agent.reactive import observation_features

PLAN = Path("config/correction-contact-scenarios.json")
MODEL = Path("outputs/models/success-replay-replay/epoch-300")
OUTPUT = Path("outputs/datasets/corrections-contact-v1")


def check_plan(config, weights):
    if config["learner_weights_sha256"] != weights:
        raise ValueError("contact recovery learner differs from audited model")
    if config["split"] != {"train": list(range(6)), "validation": [], "test": []}:
        raise ValueError("contact recovery must remain train-only")
    cases = config["cases"]
    if len(cases) != 6 or any(c["x"] != 0.302 for c in cases):
        raise ValueError("only the audited training position is allowed")
    if [c["takeover_seconds"] for c in cases] != [t / 25 for t in (75, 77, 78, 79, 82, 84)]:
        raise ValueError("fixed contact takeover schedule differs")


def verify_branch(takeover, trace):
    tick = takeover["tick"]
    if trace[tick]["tick"] != tick:
        raise ValueError("reference trace tick mismatch")
    for key, other in (
        ("qpos", "simulation_qpos"),
        ("qvel", "simulation_qvel"),
        ("ctrl", "previous_command"),
    ):
        if not np.allclose(takeover[key], trace[tick][other], atol=1e-8, rtol=0):
            raise ValueError("takeover differs from autonomous failure state")


def main(plan=PLAN, output=OUTPUT, report_path="docs/evaluations/contact-recovery-v1.json"):
    plan, output = Path(plan), Path(output)
    config = json.loads(plan.read_text())
    policy = MemoryPolicy(MODEL)
    check_plan(config, policy.metadata["weights_sha256"])
    if digest(config["reference_trace"]) != config["reference_trace_sha256"]:
        raise ValueError("audited trace changed")
    reference = json.loads(Path(config["reference_trace"]).read_text())
    validation = record(output, MODEL, plan)
    manifest = json.loads((output / "recording.json").read_text())
    rows = []
    for episode in manifest["episodes"]:
        verify_branch(episode["takeover"], reference["trace"])
        tick = episode["takeover"]["tick"]
        observed = reference["trace"][tick]
        rows.append(
            {
                "episode": episode["index"],
                "tick": tick,
                "seconds": tick / 25,
                "takeover_matches": True,
                "finger_contacts_before_takeover": observed["finger_contacts"],
                "holding_before_takeover": observed["holding"],
                "success": episode["success"],
                "reason": episode["reason"],
                "supervised_frames": episode["supervised_frames"],
                "quality": episode["quality"],
                "contact_decision": episode.get("contact_decision"),
            }
        )
    # Read through the public training gate; verify causal prefix and expert-only masks.
    loaded = []
    for episode in training_sequences(output, "train"):
        tick = manifest["episodes"][episode["episode"]]["takeover"]["tick"]
        inputs = episode["inputs"]
        images, joints, holding = (inputs[k] for k in manifest["policy_input_keys"])
        features = np.stack(
            [
                observation_features(np.rint(im.transpose(1, 2, 0) * 255).astype(np.uint8), j, h)
                for im, j, h in zip(
                    images[: tick + 1], joints[: tick + 1], holding[: tick + 1], strict=True
                )
            ]
        )
        if not np.allclose(features, reference["observations"][: tick + 1], atol=1e-5, rtol=0):
            raise ValueError("learner observation prefix differs")
        if episode["loss_mask"][:tick].any() or not episode["loss_mask"][tick]:
            raise ValueError("expert supervision boundary differs")
        loaded.append(episode["episode"])
    if loaded != [e["index"] for e in manifest["episodes"] if e["success"]]:
        raise ValueError("training reader did not exclude failed episodes")
    report = {
        "status": "complete",
        "group": "train",
        "config": config,
        "config_sha256": digest(plan),
        "script_sha256": digest(__file__),
        "dataset": str(output),
        "dataset_sha256": payload_digest(output),
        "validation": validation,
        "episodes": rows,
        "trainable_episodes": loaded,
        "supervised_frames": sum(r["supervised_frames"] for r in rows),
        "training_performed": False,
        "test_executed": False,
        "limitations": [
            "single previously inspected training position",
            "scheduled handovers, not learned recovery triggers",
            "success measures expert recoverability, not learner success",
            "no validation samples in this addition; training requires separate compatible validation data",
        ],
    }
    (output / "contact-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
