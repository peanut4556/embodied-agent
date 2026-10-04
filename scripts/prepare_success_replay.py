"""Reconfirm parent success and audit expert action differences at shared takeovers."""

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "outputs/hf-cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np
import torch

from embodied_agent.correction_data import payload_digest
from embodied_agent.correction_finetune import batch, correction_sequences
from embodied_agent.memory_evaluation import save_preview, verify_physics
from embodied_agent.memory_policy import MemoryPolicy, digest
from embodied_agent.reactive_evaluation import run_case

PARENT = Path("outputs/models/state-distill-distilled/epoch-300")
SOURCE = Path("outputs/datasets/corrections-onpolicy-v1")


class RecordingPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.metadata, self.bounds = policy.metadata, policy.bounds
        self.observations = []

    def session(self):
        session = self.policy.session()
        owner = self

        class Recorder:
            metadata, bounds = owner.metadata, owner.bounds

            def predict(self, observations):
                owner.observations.extend(np.asarray(observations).tolist())
                return session.predict(observations)

        return Recorder()


def main():
    target = Path("outputs/datasets/success-replay-v1.json")
    output = Path("outputs/evaluations/success-replay-audit-v1")
    if target.exists() or output.exists():
        raise FileExistsError("success replay output already exists")
    policy = MemoryPolicy(PARENT)
    verify_physics(policy)
    manifest = json.loads((SOURCE / "recording.json").read_text())
    source_hash = payload_digest(SOURCE)
    if manifest["learner_weights_sha256"] != policy.metadata["weights_sha256"]:
        raise ValueError("collector differs from parent")
    position = 0.302  # Previously successful training case, fixed before this run.
    if position not in [
        manifest["episodes"][i]["scenario"]["x"] for i in manifest["split"]["train"]
    ]:
        raise ValueError("success case is not training-only")
    recorder, trace = RecordingPolicy(policy), []
    result, frames = run_case(
        recorder, {"name": "parent-success-302", "x": position}, "memory", 24, trace
    )
    if not result["quality"]["verified_pick_place"]:
        raise ValueError("parent success did not reproduce")
    obs = np.asarray(recorder.observations, dtype=np.float32)
    with torch.no_grad():
        targets = policy.model(
            torch.from_numpy(((obs - policy.mean) / policy.scale).astype(np.float32))[None]
        )[0][0].numpy()
    rows = []
    for episode in correction_sequences(SOURCE, "train"):
        record = manifest["episodes"][episode["episode"]]
        if record["scenario"]["x"] != position:
            continue
        tick = record["takeover"]["tick"]
        if not np.allclose(
            trace[tick]["simulation_qpos"], record["takeover"]["qpos"], atol=1e-8, rtol=0
        ):
            raise ValueError("takeover states differ")
        if not np.allclose(obs[: tick + 1], episode["x"][: tick + 1], atol=1e-5, rtol=0):
            raise ValueError("takeover observation histories differ")
        x, y, mask = batch([episode], policy)
        with torch.no_grad():
            predicted = policy.model(x)[0]
        delta = (y[0, tick] - predicted[0, tick]).numpy() * np.diff(policy.bounds).ravel()
        rows.append(
            {
                "episode": episode["episode"],
                "takeover_tick": tick,
                "same_history_verified": True,
                "expert_minus_parent_action": delta.tolist(),
                "arm_rmse_rad": float(np.sqrt(np.mean(delta[:3] ** 2))),
                "finger_rmse_m": float(np.sqrt(np.mean(delta[3:] ** 2))),
                "expert_history_normalized_mse": float(
                    ((predicted - y).square() * mask).sum() / (mask.sum() * 5)
                ),
            }
        )
    if payload_digest(SOURCE) != source_hash:
        raise ValueError("corrections changed")
    pack = {
        "group": "train",
        "position": position,
        "fps": policy.metadata["fps"],
        "weights_sha256": policy.metadata["weights_sha256"],
        "normalization_sha256": policy.metadata["normalization_sha256"],
        "model_sha256": policy.metadata["model_sha256"],
        "source_sha256": source_hash,
        "result": result,
        "observations": obs.tolist(),
        "normalized_targets": targets.tolist(),
        "target_semantics": "unclipped frozen parent action-head outputs on its own successful recurrent history",
    }
    output.mkdir(parents=True)
    target.write_text(json.dumps(pack, indent=2) + "\n")
    save_preview(output, "parent-success-302", frames, 25)
    report = {
        "status": "complete",
        "pack_sha256": digest(target),
        "frames": len(obs),
        "result": result,
        "same_history_comparisons": rows,
        "test_executed": False,
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
