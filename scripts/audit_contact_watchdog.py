"""Known training-scene regression; no fitting, reserved tests or policy changes."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq

from embodied_agent.contact_watchdog import GuardedExecutor
from embodied_agent.grasp_quality import GraspQuality
from embodied_agent.memory_evaluation import verify_physics
from embodied_agent.memory_policy import MemoryPolicy, digest
from embodied_agent.physics import PhysicsWorld
from embodied_agent.reactive import ReactiveExecutor

ROOT = Path("outputs/datasets/corrections-contact-oriented-v1")


def restore(world, initial):
    for key in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
        getattr(world.data, key)[:] = initial[key]
    world.data.time = initial["time"]
    mujoco.mj_forward(world.model, world.data)


def advance(world, command, fps):
    world.data.ctrl[:] = command
    for _ in range(round(1 / fps / world.model.opt.timestep)):
        mujoco.mj_step(world.model, world.data)


def observe(quality, world, command):
    quality.observe(
        world.holding, world.data.body("red_block").xpos.copy(), command, world.inside_box()
    )


def historical(manifest):
    path = ROOT / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path, columns=["replay.action", "observation.state"]).to_pylist()
    results, offset = [], 0
    for episode in manifest["episodes"]:
        world = PhysicsWorld()
        try:
            restore(world, episode["initial_state"])
            quality = GraspQuality(manifest["fps"])
            error = 0.0
            for row in rows[offset : offset + episode["frames"]]:
                mujoco.mj_forward(world.model, world.data)
                error = max(
                    error, float(np.max(np.abs(world.data.qpos[:5] - row["observation.state"])))
                )
                observe(quality, world, row["replay.action"])
                advance(world, row["replay.action"], manifest["fps"])
            result = quality.result()
            assert error < 1e-6, error
            assert result["verified_pick_place"] == episode["success"]
            results.append(
                {
                    "takeover_tick": episode["takeover"]["tick"],
                    "max_joint_replay_error": error,
                    **result,
                }
            )
            offset += episode["frames"]
        finally:
            world.close()
    return {"parquet_sha256": digest(path), "cases": results}


def run(policy_path, initial, guarded, output, name, stop_tick=None):
    policy = MemoryPolicy(policy_path)
    verify_physics(policy)
    fps = policy.metadata["fps"]
    world = PhysicsWorld(perception="rgbd", perception_size_mode="oriented")
    renderer = None
    try:
        restore(world, initial)
        learner = ReactiveExecutor(policy.session(), world.data.qpos[:5], max_seconds=30)
        guard = GuardedExecutor(world, learner) if guarded else None
        renderer = mujoco.Renderer(world.model, height=240, width=320)
        quality = GraspQuality(fps)
        actions, states, terminal = [], [], None
        for tick in range(30 * fps):
            mujoco.mj_forward(world.model, world.data)
            renderer.update_scene(world.data, camera="perception")
            rgb = renderer.render()
            stop = stop_tick is not None and tick >= stop_tick
            command = (
                guard.step(rgb, stop=stop)
                if guard
                else learner.step(rgb, world.data.qpos[:5], world.holding, stop=stop)
            )
            observe(quality, world, command)
            actions.append(command.copy())
            states.append(world.data.qpos.copy())
            advance(world, command, fps)
            done = (
                bool(guard.reason) or bool(guard.expert and guard.expert.done)
                if guard
                else learner.state == "stopped"
            )
            if done or quality.result()["verified_pick_place"]:
                terminal = tick if terminal is None else terminal
                if tick - terminal >= fps:
                    break
        trace = output / f"{name}.npz"
        np.savez_compressed(trace, actions=actions, qpos=states)
        result = {
            "name": name,
            "policy": str(policy_path),
            "weights_sha256": digest(policy_path / "weights.pt"),
            "guarded": guarded,
            "takeover_tick": guard.takeover_tick if guard else None,
            "watchdog_events": guard.events if guard else [],
            "reason": guard.reason if guard else learner.reason,
            "frames": len(actions),
            "budget_exhausted": len(actions) == 30 * fps,
            "trace_sha256": digest(trace),
            "learner_ticks": learner.ticks,
            **quality.result(),
        }
        print(json.dumps(result), flush=True)
        return result
    finally:
        if renderer:
            renderer.close()
        world.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/evaluations/contact-watchdog-v1")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT / "recording.json").read_text())
    report = {
        "scope": "single known training position x=0.302; no generalization estimate",
        "manifest_sha256": digest(ROOT / "recording.json"),
        "historical": historical(manifest),
        "runs": [],
    }
    initial = manifest["episodes"][5]["initial_state"]
    for name, path in [("replay", "success-replay-replay"), ("parent", "state-distill-distilled")]:
        for guarded in (False, True):
            label = f"{name}-{'guarded' if guarded else 'baseline'}"
            report["runs"].append(
                run(Path("outputs/models") / path / "epoch-300", initial, guarded, output, label)
            )
    for tick in (20, 90):
        report["runs"].append(
            run(
                Path("outputs/models/success-replay-replay/epoch-300"),
                initial,
                True,
                output,
                f"stop-{tick}",
                stop_tick=tick,
            )
        )
    runs = {row["name"]: row for row in report["runs"]}
    traces = {name: np.load(output / f"{name}.npz") for name in runs}
    try:
        checks = {
            "historical_five_of_six": sum(
                r["verified_pick_place"] for r in report["historical"]["cases"]
            )
            == 5,
            "failing_baseline_reproduced": not runs["replay-baseline"]["verified_pick_place"],
            "causal_recovery_success": (
                runs["replay-guarded"]["verified_pick_place"]
                and runs["replay-guarded"]["takeover_tick"] == 79
                and runs["replay-guarded"]["learner_ticks"] == 79
            ),
            "identical_prefix": np.array_equal(
                traces["replay-baseline"]["actions"][:79], traces["replay-guarded"]["actions"][:79]
            ),
            "healthy_policy_unchanged": (
                runs["parent-guarded"]["verified_pick_place"]
                and runs["parent-guarded"]["takeover_tick"] is None
                and all(
                    np.array_equal(traces["parent-baseline"][key], traces["parent-guarded"][key])
                    for key in ("actions", "qpos")
                )
            ),
        }
        actions = traces["replay-guarded"]["actions"]
        checks["recovery_rate_bounds"] = bool(
            np.all(np.abs(np.diff(actions, axis=0)) <= np.array([2, 2, 2, 1, 1]) / 25 + 1e-12)
        )
        for tick in (20, 90):
            name = f"stop-{tick}"
            commands = traces[name]["actions"][tick:]
            checks[name] = (
                runs[name]["reason"] == "user stop"
                and runs[name]["learner_ticks"] == min(tick, 79)
                and np.all(commands == commands[0])
            )
        report["checks"] = {key: bool(value) for key, value in checks.items()}
    finally:
        for trace in traces.values():
            trace.close()
    report["status"] = "passed" if all(report["checks"].values()) else "failed"
    report["source_sha256"] = {
        str(path): digest(path)
        for path in [
            Path(__file__),
            Path("src/embodied_agent/contact_watchdog.py"),
            Path("src/embodied_agent/correction_data.py"),
            Path("src/embodied_agent/assets/tabletop.xml"),
        ]
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "passed":
        raise RuntimeError(f"regression failed: {report['checks']}")


if __name__ == "__main__":
    main()
