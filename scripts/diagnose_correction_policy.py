"""Development-only paired trajectory and observation-target disagreement audit."""

import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", "outputs/hf-cache")

import numpy as np

from embodied_agent.correction_data import payload_digest
from embodied_agent.correction_finetune import correction_sequences
from embodied_agent.memory_policy import MemoryPolicy, digest, sequences
from embodied_agent.reactive_evaluation import run_case
from embodied_agent.temporal_data import source_digest


def first_sustained(values, length=5):
    count = 0
    for i, value in enumerate(values):
        count = count + 1 if value else 0
        if count == length:
            return i - length + 1
    return None


def main():
    output = Path("outputs/evaluations/correction-diagnosis-v1")
    if output.exists():
        raise FileExistsError(output)
    model = Path("outputs/models/memory-correction-v1/epoch-600")
    base = Path("outputs/datasets/reactive-development-v2")
    correction = Path("outputs/datasets/corrections-expanded-v1")
    policy = MemoryPolicy(model)
    config = json.loads((model / "experiment.json").read_text())
    reference = sequences(base, config["split"], "validation")
    report = {
        "weights_sha256": policy.metadata["weights_sha256"],
        "source_sha256": source_digest(base),
        "correction_payload_sha256": payload_digest(correction),
        "script_sha256": digest(__file__),
        "test_executed": False,
        "status": "running",
        "thresholds": {"arm_radians": 0.15, "finger_meters": 0.01, "consecutive_frames": 5},
        "results": [],
    }
    output.mkdir(parents=True)
    for case, ref in zip(config["validation_cases"], reference, strict=True):
        trace = []
        result, _ = run_case(policy, case, "memory", config["max_seconds"], trace=trace)
        n = min(len(trace), len(ref["y"]))
        commands = np.array([r["command"] for r in trace[:n]])
        joints = np.array([r["joints"] for r in trace[:n]])
        # Reference features contain visual(6), joints(5), holding(1).
        errors = np.abs(commands - ref["y"][:n])
        state_errors = np.abs(joints - ref["x"][:n, 6:11])
        large = lambda e: (e[:, :3].max(1) > 0.15) | (e[:, 3:].max(1) > 0.01)
        session = policy.session()
        offline = np.stack([session.predict(x[None])[0] for x in ref["x"]])
        offline_errors = np.abs(offline - ref["y"])
        row = {
            "case": case["name"],
            "reference_episode": ref["episode"],
            "first_sustained_command_divergence_tick": first_sustained(large(errors)),
            "first_sustained_state_divergence_tick": first_sustained(large(state_errors)),
            "teacher_forced_first_divergence_tick": first_sustained(large(offline_errors)),
            "compared_frames": n,
            "result": result,
        }
        report["results"].append(row)
        (output / f"{case['name']}-trace.json").write_text(json.dumps(trace))
        print(json.dumps(row), flush=True)
    base_train = sequences(base, config["split"], "train")
    correction_train = correction_sequences(correction, "train")
    bx = np.concatenate([r["x"] for r in base_train])
    by = np.concatenate([r["y"] for r in base_train])
    pairs = []
    for episode in correction_train:
        valid = np.flatnonzero(episode["loss_mask"])
        for t in valid:
            distances = np.sqrt(np.mean(((bx - episode["x"][t]) / policy.scale) ** 2, axis=1))
            j = int(distances.argmin())
            action_distance = float(
                np.max(
                    np.abs(by[j] - episode["y"][t]) / (policy.bounds[:, 1] - policy.bounds[:, 0])
                )
            )
            if distances[j] < 0.1 and action_distance > 0.1:
                pairs.append(
                    {
                        "episode": episode["episode"],
                        "frame": int(t),
                        "base_flat_frame": j,
                        "observation_rms_z": float(distances[j]),
                        "max_normalized_action_gap": action_distance,
                    }
                )
    report.update(
        status="complete",
        similar_observation_disagreement_count=len(pairs),
        examples=pairs[:20],
        interpretation="Same-time reference divergence is diagnostic, not a proof of causality. Nearest observations omit recurrent history; disagreement does not prove contradictory recurrent targets.",
    )
    (output / "diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
