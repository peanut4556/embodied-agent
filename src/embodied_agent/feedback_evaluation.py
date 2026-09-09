"""Paired external-relocation experiments for open-loop and feedback execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .demonstrations import write_json
from .feedback import FeedbackExecutor
from .imitation import ContextPolicy
from .physics import PhysicsWorld


def run_case(policy, case, feedback):
    world = PhysicsWorld()
    renderer = None
    preview, disturbances = [], []
    try:
        world.data.qpos[5] = case["x"]
        mujoco.mj_forward(world.model, world.data)
        world.tick(0.2)
        renderer = mujoco.Renderer(world.model, height=240, width=320)
        renderer.update_scene(world.data, camera="perception")
        executor = FeedbackExecutor(
            policy, renderer.render(), world.data.qpos[:5], feedback=feedback
        )
        fps, finished_at = policy.metadata["fps"], None
        stopped_commands, stopped_joints = [], []
        for tick in range(30 * fps):
            if "relocate_at" in case and tick == round(case["relocate_at"] * fps):
                # Explicit evaluation-only relocation, not a physical push or controller action.
                world.data.qpos[5:8] = [case["target_x"], 0, 0.027]
                world.data.qpos[8:12] = [1, 0, 0, 0]
                world.data.qvel[5:11] = 0
                disturbances.append(
                    {"tick": tick, "kind": "external_relocation", "target_x": case["target_x"]}
                )
            mujoco.mj_forward(world.model, world.data)
            renderer.update_scene(world.data, camera="perception")
            rgb = renderer.render()
            stop_requested = "stop_at" in case and tick >= round(case["stop_at"] * fps)
            command = executor.step(rgb, world.data.qpos[:5], world.holding, stop=stop_requested)
            world.data.ctrl[:] = command
            if executor.state == "stopped":
                stopped_commands.append(command)
            for _ in range(round(1 / fps / world.model.opt.timestep)):
                mujoco.mj_step(world.model, world.data)
            if executor.state == "stopped":
                stopped_joints.append(world.data.qpos[:5].copy())
            if tick % 5 == 0:
                frame = Image.fromarray(rgb)
                label = "Feedback" if feedback else "Open-loop"
                ImageDraw.Draw(frame).text(
                    (5, 5),
                    f"{label} | {executor.state} | {tick / fps:.1f}s",
                    fill="white",
                    stroke_width=1,
                    stroke_fill="black",
                )
                preview.append(frame)
            if executor.state in {"stopped", "completed"}:
                if finished_at is None:
                    finished_at = tick
                if tick - finished_at >= fps:
                    break
        else:
            executor.stop(world.data.qpos[:5], "overall execution timeout")
        success = world.inside_box() and not world.holding
        stop_held = bool(stopped_commands) and bool(
            np.allclose(stopped_commands, stopped_commands[0])
        )
        stop_settled = len(stopped_joints) >= fps and bool(
            np.all(np.ptp(stopped_joints[-5:], axis=0) < [0.01, 0.01, 0.01, 0.001, 0.001])
        )
        result = {
            "case": case["name"],
            "controller": "feedback" if feedback else "open_loop",
            "expect": case["expect"],
            "success": success,
            "state": executor.state,
            "stop_reason": executor.reason,
            "stop_command_held": stop_held,
            "stop_joints_settled": stop_settled,
            "retries": executor.retries,
            "events": executor.events,
            "disturbances": disturbances,
            "duration_seconds": (tick + 1) / fps,
            "final_block_xyz": world.snapshot()["block_xyz"],
        }
        result["expectation_met"] = (
            success
            if case["expect"] == "success"
            else executor.state == "stopped" and stop_held and stop_settled
        )
        return result, preview
    finally:
        if renderer is not None:
            renderer.close()
        world.close()


def evaluate_feedback(model, scenarios, output, group="test"):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    config = json.loads(Path(scenarios).read_text())
    policy = ContextPolicy(model)
    scene = Path(__file__).parent / "assets/tabletop.xml"
    if hashlib.sha256(scene.read_bytes()).hexdigest() != policy.metadata["model_sha256"]:
        raise ValueError("physics model differs from the frozen policy")
    if group not in {"development", "test"}:
        raise ValueError("unknown scenario group")
    output.mkdir(parents=True)
    results = []
    for case in config[group]:
        pair = []
        for feedback in (False, True):
            result, frames = run_case(policy, case, feedback)
            results.append(result)
            pair.append(frames)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        combined = []
        for i in range(max(map(len, pair))):
            frame = Image.new("RGB", (640, 240))
            for side in range(2):
                frame.paste(pair[side][min(i, len(pair[side]) - 1)], (320 * side, 0))
            combined.append(frame)
        combined[0].save(
            output / f"{case['name']}.gif",
            save_all=True,
            append_images=combined[1:],
            duration=round(5000 / policy.metadata["fps"]),
            loop=0,
        )
        write_json(output / "partial-results.json", results)
    successful_tasks = [r for r in results if r["expect"] == "success"]
    guards = [r for r in results if r["expect"] == "stopped" and r["controller"] == "feedback"]
    report = {
        "group": group,
        "results": results,
        "task_success": {
            name: {
                "passed": sum(r["success"] for r in successful_tasks if r["controller"] == name),
                "total": sum(r["controller"] == name for r in successful_tasks),
            }
            for name in ("open_loop", "feedback")
        },
        "feedback_stop_checks": {
            "passed": sum(r["expectation_met"] for r in guards),
            "total": len(guards),
        },
        "policy_sha256": hashlib.sha256((Path(model) / "policy.npz").read_bytes()).hexdigest(),
        "scenarios_sha256": hashlib.sha256(Path(scenarios).read_bytes()).hexdigest(),
        "limitations": "hybrid feedback rules; evaluation relocations are not physical pushes",
    }
    write_json(output / "evaluation.json", report)
    return report
