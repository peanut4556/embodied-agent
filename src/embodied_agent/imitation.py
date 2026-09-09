"""A visual-context behavior-cloning baseline for the fixed tabletop task.

Fits polynomial regression from the initial image feature to an action trajectory.
This is an open-loop learned baseline, not a reactive policy or an ACT/VLA model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .demonstrations import IMAGE_KEY, JOINTS, TASK, write_json
from .physics import PhysicsWorld


def visual_context(rgb):
    """Extract a normalized horizontal centroid; no depth or object pose is used."""
    rgb = np.asarray(rgb)
    if rgb.shape != (240, 320, 3) or rgb.dtype != np.uint8:
        raise ValueError("expected a 240x320 uint8 RGB frame")
    red, green, blue = rgb.astype(float).transpose(2, 0, 1)
    y, x = np.nonzero((red > 70) & (red > 1.6 * green) & (red > 1.4 * blue))
    if not 80 <= len(x) <= 260:
        raise ValueError("initial image requires one fully visible red cube")
    width, height = np.ptp(x) + 1, np.ptp(y) + 1
    if not (10 <= width <= 18 and 10 <= height <= 18) or len(x) / (width * height) < 0.7:
        raise ValueError("initial image target is ambiguous or occluded")
    if abs((y.mean() + 0.5) / 240 - 0.5) > 0.02:
        raise ValueError("initial target is outside the demonstrated motion plane")
    return float((x.mean() + 0.5) / 320)


def check_split(manifest, split):
    if set(split) != {"train", "validation", "test"} or any(not ids for ids in split.values()):
        raise ValueError("non-empty train, validation and test splits required")
    ids = [i for values in split.values() for i in values]
    if any(type(i) is not int for i in ids) or sorted(ids) != list(
        range(len(manifest["episodes"]))
    ):
        raise ValueError("split must cover every episode exactly once")
    scenes = {}
    for group, values in split.items():
        for index in values:
            x = round(manifest["episodes"][index]["initial_cube_x"], 9)
            if x in scenes and scenes[x] != group:
                raise ValueError("same initial scene appears across different splits")
            scenes[x] = group
    if len(split["train"]) < 5:
        raise ValueError("at least five training episodes required")


def fit_context_model(context, actions, degree, mean, scale):
    design = np.polynomial.polynomial.polyvander((context - mean) / scale, degree)
    coefficients = np.linalg.solve(
        design.T @ design + 1e-6 * np.eye(degree + 1),
        design.T @ actions.reshape(len(context), -1),
    )
    return coefficients.reshape(degree + 1, *actions.shape[1:])


def predict_context(coefficients, context, mean, scale):
    design = np.polynomial.polynomial.polyvander(
        (np.asarray(context) - mean) / scale, len(coefficients) - 1
    )
    return np.einsum("...d,dtj->...tj", design, coefficients)


def dataset_manifest(root):
    manifest = json.loads((root / "recording.json").read_text())
    validation = json.loads((root / "validation.json").read_text())
    if manifest["status"] != "validated" or not validation["success"]:
        raise ValueError("training requires a successfully validated dataset")
    if manifest["joint_names"] != JOINTS or manifest["task"] != TASK:
        raise ValueError("dataset robot or task mismatch")
    return manifest


def train(root: Path, output: Path, split_path: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite model: {output}")
    manifest = dataset_manifest(root)
    split = json.loads(Path(split_path).read_text())
    check_split(manifest, split)
    dataset = LeRobotDataset(manifest["repo_id"], root=root, video_backend="pyav")
    offsets = np.cumsum([0] + [ep["frames"] for ep in manifest["episodes"]])
    horizon = manifest["episodes"][split["train"][0]]["frames"]

    def load_group(group):
        contexts, actions = [], []
        for index in split[group]:
            if manifest["episodes"][index]["frames"] != horizon:
                raise ValueError("baseline requires equal-duration demonstrations")
            first = dataset[int(offsets[index])]
            rgb = np.rint(first[IMAGE_KEY].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            contexts.append(visual_context(rgb))
            actions.append(
                np.stack(
                    [
                        dataset[i]["action"].numpy()
                        for i in range(offsets[index], offsets[index + 1])
                    ]
                )
            )
        return np.array(contexts), np.stack(actions)

    # Test observations/actions are never loaded for fitting or model selection.
    train_x, train_y = load_group("train")
    val_x, val_y = load_group("validation")
    mean, scale = float(train_x.mean()), float(train_x.std())
    if scale < 1e-5:
        raise ValueError("training scenes have no visual diversity")
    world = PhysicsWorld()
    bounds = world.model.actuator_ctrlrange.copy()
    world.close()
    action_scale = bounds[:, 1] - bounds[:, 0]
    scores, candidates = [], []
    for degree in (1, 2, 3):
        coefficients = fit_context_model(train_x, train_y, degree, mean, scale)
        prediction = predict_context(coefficients, val_x, mean, scale)
        score = float(np.mean(((prediction - val_y) / action_scale) ** 2))
        candidates.append(coefficients)
        scores.append({"degree": degree, "validation_normalized_mse": score})
    best = min(range(len(scores)), key=lambda i: scores[i]["validation_normalized_mse"])
    baseline = train_y.mean(axis=0)
    report = {
        "algorithm": "visual-context polynomial behavior cloning",
        "control": "open-loop action trajectory conditioned on first RGB frame",
        "split": split,
        "selection_metric": "validation_normalized_mse",
        "candidates": scores,
        "selected_degree": scores[best]["degree"],
        "baseline_validation_normalized_mse": float(
            np.mean(((baseline - val_y) / action_scale) ** 2)
        ),
        "test_used_for_training": False,
        "fps": manifest["fps"],
        "horizon": horizon,
        "joint_names": JOINTS,
        "joint_units": manifest["joint_units"],
        "model_sha256": manifest["model_sha256"],
        "dataset_manifest_sha256": hashlib.sha256(
            (root / "recording.json").read_bytes()
        ).hexdigest(),
        "train_context_min": float(train_x.min()),
        "train_context_max": float(train_x.max()),
        "camera": manifest["camera"],
    }
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "policy.npz",
        coefficients=candidates[best],
        mean=mean,
        scale=scale,
        bounds=bounds,
        baseline=baseline,
    )
    write_json(output / "training.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


class ContextPolicy:
    def __init__(self, root):
        root = Path(root)
        self.metadata = json.loads((root / "training.json").read_text())
        with np.load(root / "policy.npz", allow_pickle=False) as weights:
            self.coefficients = weights["coefficients"].copy()
            self.mean, self.scale = float(weights["mean"]), float(weights["scale"])
            self.bounds, self.baseline = weights["bounds"].copy(), weights["baseline"].copy()
        if (
            self.coefficients.shape
            != (self.metadata["selected_degree"] + 1, self.metadata["horizon"], 5)
            or self.metadata["fps"] not in {25, 50}
            or self.metadata["joint_names"] != JOINTS
            or self.baseline.shape != (self.metadata["horizon"], 5)
            or self.bounds.shape != (5, 2)
            or self.scale <= 0
            or not np.isfinite(self.coefficients).all()
            or not np.isfinite(self.baseline).all()
            or not np.isfinite(self.bounds).all()
            or not np.isfinite([self.mean, self.scale]).all()
        ):
            raise ValueError("invalid checkpoint")
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("invalid actuator bounds")

    def plan(self, initial_rgb, baseline=False):
        context = visual_context(initial_rgb)
        if not self.metadata["train_context_min"] <= context <= self.metadata["train_context_max"]:
            raise ValueError("visual target is outside the training range")
        action = (
            self.baseline.copy()
            if baseline
            else predict_context(self.coefficients, context, self.mean, self.scale).reshape(-1, 5)
        )
        if not np.isfinite(action).all():
            raise ValueError("model produced non-finite actions")
        return np.clip(action, self.bounds[:, 0], self.bounds[:, 1])


def evaluate(root: Path, model: Path, output: Path):
    """Frozen-policy physical evaluation on the reserved test positions only."""
    from PIL import Image, ImageDraw

    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    manifest = dataset_manifest(root)
    policy = ContextPolicy(model)
    metadata = policy.metadata
    if (
        hashlib.sha256((root / "recording.json").read_bytes()).hexdigest()
        != metadata["dataset_manifest_sha256"]
    ):
        raise ValueError("evaluation dataset differs from training provenance")
    source = Path(__file__).parent / "assets/tabletop.xml"
    if hashlib.sha256(source.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("physics model differs from checkpoint")
    check_split(manifest, metadata["split"])
    output.mkdir(parents=True)
    results = []
    for index in metadata["split"]["test"]:
        x = manifest["episodes"][index]["initial_cube_x"]
        for baseline in (False, True):
            world = PhysicsWorld()
            renderer = None
            frames = []
            try:
                world.data.qpos[5] = x  # Test scene setup, never a policy input.
                mujoco.mj_forward(world.model, world.data)
                world.tick(0.2)
                renderer = mujoco.Renderer(world.model, height=240, width=320)
                renderer.update_scene(world.data, camera="perception")
                actions = policy.plan(renderer.render(), baseline=baseline)
                for frame_index, action in enumerate(actions):
                    world.data.ctrl[:] = action
                    for _ in range(round(1 / metadata["fps"] / world.model.opt.timestep)):
                        mujoco.mj_step(world.model, world.data)
                    if frame_index % 5 == 0:
                        mujoco.mj_forward(world.model, world.data)
                    if not baseline and frame_index % 5 == 0:
                        renderer.update_scene(world.data, camera="perception")
                        image = Image.fromarray(renderer.render())
                        ImageDraw.Draw(image).text(
                            (5, 5),
                            f"Learned policy | test x={x:.3f}",
                            fill="white",
                            stroke_width=1,
                            stroke_fill="black",
                        )
                        frames.append(image)
                success = world.inside_box() and not world.holding
                result = {
                    "episode": index,
                    "initial_x": x,
                    "controller": "mean_action_baseline" if baseline else "learned_policy",
                    "success": success,
                    "final_block_xyz": world.snapshot()["block_xyz"],
                }
                results.append(result)
                print(json.dumps(result), flush=True)
                if frames:
                    frames[0].save(
                        output / f"test-{index}.gif",
                        save_all=True,
                        append_images=frames[1:],
                        duration=round(5000 / metadata["fps"]),
                        loop=0,
                    )
            finally:
                if renderer is not None:
                    renderer.close()
                world.close()
    learned = [r["success"] for r in results if r["controller"] == "learned_policy"]
    baseline = [r["success"] for r in results if r["controller"] == "mean_action_baseline"]
    report = {
        "results": results,
        "test_episodes": len(learned),
        "learned_success_rate": float(np.mean(learned)),
        "baseline_success_rate": float(np.mean(baseline)),
        "limitations": "fixed camera, upright cube, open-loop; no disturbance recovery",
    }
    write_json(output / "evaluation.json", report)
    return report
