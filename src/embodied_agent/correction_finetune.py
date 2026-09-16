"""Warm-start GRU fine-tuning; expert-only targets and demonstration rehearsal."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .correction_data import payload_digest, training_sequences
from .memory_policy import MemoryPolicy, check_config, digest, masked_loss, padded_batch, sequences
from .reactive import observation_features
from .temporal_data import INPUT_KEYS, check_source, read_json, source_digest


def check_groups(base, split, corrections, config, policy):
    """Reject leakage across datasets as well as inside a single recording."""
    groups = {}
    for manifest, partition, key in [(base, split, "scenario")] + [
        (m, m["split"], "scenario") for m in corrections
    ]:
        if (
            manifest["fps"] != policy.metadata["fps"]
            or manifest["model_sha256"] != policy.metadata["model_sha256"]
        ):
            raise ValueError("dataset physics or clock differs from pretrained policy")
        if partition["test"]:
            raise ValueError("training sources must be development-only")
        for group in ("train", "validation"):
            for i in partition[group]:
                x = round(manifest["episodes"][i][key]["x"], 9)
                if x in groups and groups[x] != group:
                    raise ValueError("cross-dataset train/validation overlap")
                groups[x] = group
    excluded = set(groups) | {round(x, 9) for x in config["previous_test_positions"]}
    if any(round(c["x"], 9) in excluded for c in config["test"]):
        raise ValueError("reserved test overlaps inspected data")


def correction_sequences(root, group):
    episodes = []
    for episode in training_sequences(root, group):
        inputs = episode["inputs"]
        x = [
            observation_features(
                np.rint(image.transpose(1, 2, 0) * 255).astype(np.uint8),
                state,
                holding,
            )
            for image, state, holding in zip(*(inputs[k] for k in INPUT_KEYS), strict=True)
        ]
        episodes.append(
            {
                "episode": episode["episode"],
                "x": np.stack(x),
                "y": episode["targets"],
                "loss_mask": episode["loss_mask"],
            }
        )
    return episodes


def batch(episodes, policy):
    """Retain roll-in observations in the GRU, but never regress their actions."""
    import torch

    x, y, mask = padded_batch(episodes, policy.mean, policy.scale, policy.bounds)
    for i, episode in enumerate(episodes):
        valid = np.asarray(episode.get("loss_mask", np.ones(len(episode["x"]), dtype=bool)))
        if valid.dtype != bool or valid.shape != (len(episode["x"]),) or not valid.any():
            raise ValueError("nonempty boolean supervision mask required")
        targets = episode["y"][valid]
        if not np.isfinite(episode["x"]).all() or not np.isfinite(targets).all():
            raise ValueError("nonfinite training data")
        if np.any(targets < policy.bounds[:, 0] - 1e-6) or np.any(
            targets > policy.bounds[:, 1] + 1e-6
        ):
            raise ValueError("expert targets exceed actuator bounds")
        mask[i, : len(valid), 0] = torch.from_numpy(valid.astype(np.float32))
    return x, y, mask


def train(base_root, correction_roots, pretrained, experiment, output):
    import torch

    base_root = Path(base_root).resolve()
    correction_roots = [Path(r).resolve() for r in correction_roots]
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite run: {output}")
    config = read_json(experiment)
    policy = MemoryPolicy(pretrained)
    base = check_source(base_root, config["split"])
    check_config(config, base)
    if (
        config.get("selection_metric") != "verified_pick_place"
        or config["hidden"] != policy.metadata["hidden"]
    ):
        raise ValueError("fine-tuning requires qualified selection and matching architecture")
    manifests = [read_json(Path(r) / "recording.json") for r in correction_roots]
    check_groups(base, config["split"], manifests, config, policy)
    if any(m["learner_weights_sha256"] != policy.metadata["weights_sha256"] for m in manifests):
        raise ValueError("corrections must come from the frozen pretrained learner")

    def hashes():
        return {
            "base": source_digest(base_root),
            "corrections": [payload_digest(r) for r in correction_roots],
        }

    sources = hashes()
    source_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    groups = {}
    for group in ("train", "validation"):
        original = sequences(base_root, config["split"], group)
        corrections = [e for root in correction_roots for e in correction_sequences(root, group)]
        if not corrections:
            raise ValueError("successful expert corrections required in both groups")
        groups[group] = original + corrections
    x, y, mask = batch(groups["train"], policy)
    vx, vy, vm = batch(groups["validation"], policy)
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    model = policy.model  # Actual pretrained weights, never a fresh network.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    output.mkdir(parents=True)
    raw = Path(experiment).read_bytes()
    (output / "experiment.json").write_bytes(raw)
    report = {
        "status": "training",
        "source_sha256": source_hash,
        "sources": sources,
        "experiment_sha256": digest(output / "experiment.json"),
        "pretrained_weights_sha256": policy.metadata["weights_sha256"],
        "normalization": "frozen pretrained training-only statistics",
        "test_used_for_training": False,
        "candidates": [],
        "curve": [],
        "train_episodes": len(groups["train"]),
        "validation_episodes": len(groups["validation"]),
        "train_supervised_frames": int(mask.sum()),
        "validation_supervised_frames": int(vm.sum()),
    }
    with torch.no_grad():
        report["initial_validation_mse"] = float(masked_loss(model(vx)[0], vy, vm))
    try:
        for epoch in range(1, max(config["candidate_epochs"]) + 1):
            model.train()
            optimizer.zero_grad()
            loss = masked_loss(model(x)[0], y, mask)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            if epoch == 1 or epoch % 50 == 0 or epoch in config["candidate_epochs"]:
                model.eval()
                with torch.no_grad():
                    val = float(masked_loss(model(vx)[0], vy, vm))
                row = {
                    "epoch": epoch,
                    "train_mse_before_update": float(loss.detach()),
                    "validation_mse": val,
                }
                report["curve"].append(row)
                print(json.dumps(row), flush=True)
            if epoch in config["candidate_epochs"]:
                target = output / f"epoch-{epoch}"
                target.mkdir()
                torch.save(model.state_dict(), target / "weights.pt")
                # Byte-identical normalization makes pretrained input semantics explicit.
                (target / "normalization.npz").write_bytes(
                    (Path(pretrained) / "normalization.npz").read_bytes()
                )
                (target / "experiment.json").write_bytes(raw)
                metadata = {
                    **policy.metadata,
                    "epoch": epoch,
                    "validation_normalized_mse": val,
                    "source_sha256": source_hash,
                    "experiment_sha256": report["experiment_sha256"],
                    "weights_sha256": digest(target / "weights.pt"),
                    "pretrained_weights_sha256": report["pretrained_weights_sha256"],
                    "finetune_source_sha256": digest(__file__),
                    "supervision": "expert-only mask plus base rehearsal",
                }
                (target / "training.json").write_text(json.dumps(metadata, indent=2))
                report["candidates"].append(
                    {
                        "directory": target.name,
                        "epoch": epoch,
                        "validation_mse": val,
                        "weights_sha256": metadata["weights_sha256"],
                    }
                )
        if hashes() != sources:
            raise ValueError("training sources changed")
        report["status"] = "trained"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
