"""Two-second development-only action-chain ablation; never changes production limits."""

import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", "outputs/hf-cache")

import mujoco
import numpy as np

from embodied_agent.memory_evaluation import verify_physics
from embodied_agent.memory_policy import MemoryPolicy, digest, sequences
from embodied_agent.physics import PhysicsWorld
from embodied_agent.reactive import ReactiveExecutor
from embodied_agent.temporal_data import source_digest
from scripts.diagnose_correction_policy import first_sustained


class ObservedSession:
    def __init__(self, policy):
        self.session = policy.session()
        self.metadata, self.bounds = policy.metadata, policy.bounds
        self.prediction = None

    def predict(self, observations):
        result = self.session.predict(observations)
        self.prediction = result[0].copy()
        return result


def shadow_step(model, data, command, steps):
    """Integrate a complete copied simulator state without touching the live world."""
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    shadow = mujoco.MjData(model)
    mujoco.mj_setState(model, shadow, state, spec)
    shadow.ctrl[:] = command
    for _ in range(steps):
        mujoco.mj_step(model, shadow)
    return shadow.qpos[:5].copy()


def run(policy, case, reference, mode, frames=50):
    if mode not in ("reference", "limited", "unlimited"):
        raise ValueError("unknown diagnostic mode")
    world = PhysicsWorld()
    renderer = None
    try:
        world.data.qpos[5] = case["x"]
        mujoco.mj_forward(world.model, world.data)
        world.tick(0.2)
        if not np.allclose(world.data.qpos[:5], reference["x"][0, 6:11], atol=1e-5, rtol=0):
            raise ValueError("reference initial state differs")
        observed = ObservedSession(policy)
        executor = ReactiveExecutor(observed, world.data.qpos[:5], max_seconds=3)
        if mode == "unlimited":
            executor.limit[:] = np.inf  # Counterfactual in this isolated world only.
        renderer = mujoco.Renderer(world.model, height=240, width=320)
        rows = []
        steps = round(1 / policy.metadata["fps"] / world.model.opt.timestep)
        for tick in range(frames):
            mujoco.mj_forward(world.model, world.data)
            before = world.data.qpos[:5].copy()
            if mode == "reference":
                prediction = command = reference["y"][tick].copy()
            else:
                renderer.update_scene(world.data, camera="perception")
                command = executor.step(renderer.render(), before, world.holding)
                if executor.state == "stopped":
                    raise ValueError(executor.reason)
                prediction = observed.prediction.copy()
            predicted_next = shadow_step(world.model, world.data, prediction, steps)
            world.data.ctrl[:] = command
            for _ in range(steps):
                mujoco.mj_step(world.model, world.data)
            after = world.data.qpos[:5].copy()
            rows.append(
                {
                    "tick": tick,
                    "prediction": prediction.tolist(),
                    "command": command.tolist(),
                    "joints_before": before.tolist(),
                    "joints_after": after.tolist(),
                    "shadow_unlimited_next_joints": predicted_next.tolist(),
                    "reference_command": reference["y"][tick].tolist(),
                    "reference_next_joints": reference["x"][tick + 1, 6:11].tolist(),
                }
            )
        return rows
    finally:
        if renderer is not None:
            renderer.close()
        world.close()


def summarize(rows, fps):
    values = {key: np.array([r[key] for r in rows]) for key in rows[0] if key != "tick"}
    delta = np.abs(values["prediction"] - values["command"])
    error = np.abs(values["joints_after"] - values["reference_next_joints"])
    first = first_sustained((error[:, :3].max(1) > 0.15) | (error[:, 3:].max(1) > 0.01))
    shadow = np.abs(values["joints_after"] - values["shadow_unlimited_next_joints"])
    result = {
        "rate_limited_frames": int(np.any(delta > 1e-8, axis=1).sum()),
        "first_sustained_next_state_error_tick": first,
        "max_same_state_limit_effect_arm_rad": float(shadow[:, :3].max()),
        "max_same_state_limit_effect_finger_m": float(shadow[:, 3:].max()),
    }
    for label, n in [("first_second", fps), ("two_seconds", len(rows))]:
        result[label] = {
            "arm_state_rmse_rad": float(np.sqrt(np.mean(error[:n, :3] ** 2))),
            "finger_state_rmse_m": float(np.sqrt(np.mean(error[:n, 3:] ** 2))),
            "max_limit_action_change_arm_rad": float(delta[:n, :3].max()),
            "max_limit_action_change_finger_m": float(delta[:n, 3:].max()),
        }
    return result


def main():
    root = Path("outputs/evaluations/action-chain-v1")
    if root.exists():
        raise FileExistsError(root)
    config = json.loads(Path("config/correction-early-experiment.json").read_text())
    base = Path("outputs/datasets/reactive-development-v2")
    refs = sequences(base, config["split"], "validation")
    models = [
        "outputs/models/memory-correction-v1/epoch-600",
        "outputs/models/memory-correction-v2/epoch-600",
    ]
    root.mkdir(parents=True)
    report = {
        "status": "running",
        "source_sha256": source_digest(base),
        "script_sha256": digest(__file__),
        "horizon_seconds": 2,
        "test_executed": False,
        "production_limits_changed": False,
        "results": [],
    }
    try:
        for model in models:
            policy = MemoryPolicy(model)
            verify_physics(policy)
            for case, ref in zip(config["validation_cases"], refs, strict=True):
                for mode in ("reference", "limited", "unlimited"):
                    rows = run(policy, case, ref, mode)
                    name = Path(model).parent.name + "-" + case["name"] + "-" + mode + ".json"
                    (root / name).write_text(json.dumps(rows) + "\n")
                    result = {
                        "model": model,
                        "weights_sha256": policy.metadata["weights_sha256"],
                        "case": case,
                        "mode": mode,
                        "trace": name,
                        "trace_sha256": digest(root / name),
                        **summarize(rows, policy.metadata["fps"]),
                    }
                    report["results"].append(result)
                    print(json.dumps(result), flush=True)
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
