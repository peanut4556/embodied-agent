"""Frozen paired MuJoCo evaluation; all truth access stays in this fixture."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .feedback import FeedbackExecutor
from .imitation import ContextPolicy
from .physics import PhysicsWorld
from .reactive import ReactiveExecutor, ReactivePolicy
from .temporal_data import read_json


def run_case(policy, case, mode, max_seconds):
    if mode not in {"open_loop", "feedback", "reactive", "memory"}:
        raise ValueError("unknown evaluation controller")
    world = PhysicsWorld()
    renderer = None
    frames, stops = [], []
    disturbance = None
    try:
        world.data.qpos[5] = case["x"]  # Fixture only; never passed to the controller.
        mujoco.mj_forward(world.model, world.data)
        world.tick(0.2)
        renderer = mujoco.Renderer(world.model, height=240, width=320)
        renderer.update_scene(world.data, camera="perception")
        executor = (
            ReactiveExecutor(
                policy.session() if mode == "memory" else policy,
                world.data.qpos[:5],
                max_seconds,
            )
            if mode in {"reactive", "memory"}
            else FeedbackExecutor(
                policy, renderer.render(), world.data.qpos[:5], feedback=mode == "feedback"
            )
        )
        fps = policy.metadata["fps"]
        body = world.model.body("red_block").id
        start = round(case.get("force_at", -1) * fps)
        force_ticks = round(case.get("force_duration", 0) * fps)
        stable_goal, completed_at, reached_goal = 0, None, False
        loss_seen, regrasp_seen = False, False
        for tick in range(round(max_seconds * fps) + fps + 1):
            world.data.xfrc_applied[:] = 0
            mujoco.mj_forward(world.model, world.data)
            holding = world.holding
            if tick == start:
                disturbance = {
                    "tick": tick,
                    "holding_before_force": holding,
                    "force_z": case["force_z"],
                    "duration": case["force_duration"],
                }
            if start <= tick < start + force_ticks:
                world.data.xfrc_applied[body, 2] = case["force_z"]
            if disturbance and disturbance["holding_before_force"] and tick > start:
                loss_seen |= not holding
                regrasp_seen |= loss_seen and holding
            renderer.update_scene(world.data, camera="perception")
            rgb = renderer.render()
            user_stop = "stop_at" in case and tick >= round(case["stop_at"] * fps)
            if completed_at is None:
                command = executor.step(rgb, world.data.qpos[:5].copy(), holding, stop=user_stop)
                if tick >= round(max_seconds * fps) and executor.state != "stopped":
                    command = executor.stop(world.data.qpos[:5], "execution timeout")
                world.data.ctrl[:] = command
            if executor.state == "stopped":
                stops.append(world.data.ctrl.copy())
            for _ in range(round(1 / fps / world.model.opt.timestep)):
                mujoco.mj_step(world.model, world.data)
            mujoco.mj_forward(world.model, world.data)
            goal = world.inside_box() and not world.holding
            stable_goal = stable_goal + 1 if goal else 0
            if tick % 5 == 0:
                frame = Image.fromarray(rgb)
                ImageDraw.Draw(frame).text(
                    (5, 5),
                    f"{mode} | {tick / fps:.1f}s | {executor.state}",
                    fill="white",
                    stroke_width=1,
                    stroke_fill="black",
                )
                frames.append(frame)
            # Evaluation-only goal adjudication; this is not learned completion.
            if completed_at is None and (
                stable_goal >= fps or executor.state in {"stopped", "completed"}
            ):
                completed_at = tick
                reached_goal = stable_goal >= fps
            if completed_at is not None and tick - completed_at >= fps:
                break
        success = bool(world.inside_box() and not world.holding)
        stop_ok = bool(stops) and bool(np.allclose(stops, stops[0]))
        result = {
            "case": case["name"],
            "controller": mode,
            "success": success,
            "state": executor.state,
            "reason": executor.reason,
            "evaluator_detected_stable_goal": reached_goal,
            "duration_seconds": (tick + 1) / fps,
            "requested_stop": "stop_at" in case,
            "stop_command_held": stop_ok,
            "user_stop_passed": executor.reason == "user stop" and stop_ok,
            "disturbance": disturbance,
            "contact_loss_after_force": loss_seen,
            "regrasp_after_loss": regrasp_seen,
            "recovery_success": bool(
                disturbance
                and disturbance["holding_before_force"]
                and loss_seen
                and regrasp_seen
                and success
            ),
            "final_block_xyz": world.snapshot()["block_xyz"],
        }
        return result, frames
    finally:
        world.data.xfrc_applied[:] = 0
        if renderer is not None:
            renderer.close()
        world.close()


def evaluate(model_path, baseline_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    policy, baseline = ReactivePolicy(model_path), ContextPolicy(baseline_path)
    config = read_json(Path(model_path) / "experiment.json")
    scene = Path(__file__).parent / "assets/tabletop.xml"
    digest = hashlib.sha256(scene.read_bytes()).hexdigest()
    if any(p.metadata["model_sha256"] != digest for p in (policy, baseline)):
        raise ValueError("evaluation physics differs from checkpoint")
    world = PhysicsWorld()
    try:
        if any(
            not np.array_equal(p.bounds, world.model.actuator_ctrlrange) for p in (policy, baseline)
        ):
            raise ValueError("actuator limits differ from checkpoint")
    finally:
        world.close()
    if policy.metadata["fps"] != baseline.metadata["fps"]:
        raise ValueError("controller clocks differ")
    output.mkdir(parents=True)
    (output / "status.json").write_text(json.dumps({"status": "running"}))
    results = []
    try:
        for case in config["test"]:
            for mode in ("open_loop", "feedback", "reactive"):
                result, frames = run_case(
                    policy if mode == "reactive" else baseline, case, mode, config["max_seconds"]
                )
                results.append(result)
                print(json.dumps(result), flush=True)
                frames[0].save(
                    output / f"{case['name']}-{mode}.gif",
                    save_all=True,
                    append_images=frames[1:],
                    duration=round(5000 / policy.metadata["fps"]),
                    loop=0,
                )
                (output / "partial-results.json").write_text(json.dumps(results, indent=2))
    except BaseException as exc:
        (output / "status.json").write_text(json.dumps({"status": "failed", "error": str(exc)}))
        raise
    scores = {}
    for mode in ("open_loop", "feedback", "reactive"):
        tasks = [r for r in results if r["controller"] == mode and not r["requested_stop"]]
        pulls = [r for r in tasks if r["disturbance"] is not None]
        valid_pulls = [
            r
            for r in pulls
            if r["disturbance"]["holding_before_force"] and r["contact_loss_after_force"]
        ]
        scores[mode] = {
            "task_success": sum(r["success"] for r in tasks),
            "tasks": len(tasks),
            "disturbance_tasks": len(pulls),
            "valid_slip_trials": len(valid_pulls),
            "recovery_success": sum(r["recovery_success"] for r in valid_pulls),
            "non_timeout_unrequested_stops": sum(
                r["state"] == "stopped" and r["reason"] != "execution timeout" for r in tasks
            ),
            "timeouts": sum(r["reason"] == "execution timeout" for r in tasks),
            "mean_task_seconds": float(np.mean([r["duration_seconds"] for r in tasks])),
            "user_stop_passed": all(
                r["user_stop_passed"]
                for r in results
                if r["controller"] == mode and r["requested_stop"]
            ),
        }
    report = {
        "scores": scores,
        "results": results,
        "model_sha256": policy.metadata["weights_sha256"],
        "experiment_sha256": policy.metadata["experiment_sha256"],
        "baseline_sha256": hashlib.sha256(
            (Path(baseline_path) / "policy.npz").read_bytes()
        ).hexdigest(),
        "limitations": "four frozen interpolation cases; evaluation-only truth goal detection; no learned completion; fixed visual conditions",
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2))
    (output / "status.json").write_text(json.dumps({"status": "complete"}))
    return report
