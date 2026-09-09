"""Apply physical force pulses to the cube; evaluate the production learned adapter.

Object pose is set only for initial scene setup. During execution this fixture
uses MuJoCo external forces, never object teleportation or a fake holding signal.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from embodied_agent.physics import PhysicsWorld


def run_case(model, case, feedback):
    world = PhysicsWorld(policy_path=model)
    frames, forces, stopped_joints = [], [], []
    stopped_command = None
    try:
        world.data.qpos[5] = case["x"]  # Initial scene only, not an execution input.
        mujoco.mj_forward(world.model, world.data)
        world.tick(0.2)
        world.begin("learned_pick_place", {"object": "red_block", "destination": "box"})
        motion = world.policy_motion
        motion.controller.feedback = feedback
        body = world.model.body("red_block").id
        force_start = round(case.get("force_at", -1) * 25)
        force_end = force_start + round(case.get("force_duration", 0) * 25)
        stop_sent = False
        finished_at = None
        first_loss, confirmed_at = None, None
        for tick in range(825):
            world.data.xfrc_applied[body, :] = 0
            if tick == force_start:
                assert world.holding, "force pulse must begin during a real contact grasp"
            if force_start <= tick < force_end:
                world.data.xfrc_applied[body, 2] = case["force_z"]
                forces.append({"tick": tick, "force_z": case["force_z"]})
            if (
                case.get("stop_on_recovery")
                and feedback
                and not stop_sent
                and motion.controller.state == "returning"
            ):
                world.begin("stop", {})
                stop_sent = True
            world.tick(0.04)
            if (
                force_start >= 0
                and tick >= force_start
                and not world.holding
                and first_loss is None
            ):
                first_loss = tick
            if motion.controller.slips and confirmed_at is None:
                confirmed_at = tick
            if not world.active:
                if finished_at is None:
                    finished_at = tick
                if motion.controller.state == "stopped":
                    if stopped_command is None:
                        stopped_command = world.data.ctrl.copy()
                    np.testing.assert_array_equal(world.data.ctrl, stopped_command)
                    stopped_joints.append(world.data.qpos[:5].copy())
                if tick - finished_at >= 25:
                    break
            if tick % 5 == 0:
                # Separate renderer from the one closed by the production adapter at completion.
                frame = (
                    Image.open(io.BytesIO(world.render_jpeg())).convert("RGB").resize((480, 320))
                )
                ImageDraw.Draw(frame).text(
                    (8, 8),
                    f"{'Feedback' if feedback else 'Open loop'} | {motion.controller.state} | {tick / 25:.2f}s",
                    fill="white",
                    stroke_width=1,
                    stroke_fill="black",
                )
                frames.append(frame)
        else:
            raise AssertionError("production adapter failed to terminate")
        ex = motion.snapshot()
        settled = bool(stopped_joints) and bool(
            np.all(np.ptp(stopped_joints[-5:], axis=0) < [0.01, 0.01, 0.01, 0.001, 0.001])
        )
        inside = world.inside_box()
        complete = ex["state"] == "completed" and inside
        if case["expect"] == "completed":
            expected = complete and ex["slips"] == 0 and ex["retries"] == 0
        elif case["expect"] == "recovered":
            expected = complete and ex["slips"] >= 1 and ex["retries"] >= 1
        else:
            expected = ex["state"] == "stopped" and ex["slips"] >= 1 and settled
            if case.get("stop_on_recovery"):
                expected = expected and stop_sent and ex["reason"] == "user stop"
        return {
            "case": case["name"],
            "controller": "feedback" if feedback else "open_loop",
            "expected": case["expect"],
            "expectation_met": bool(expected),
            "task_success": complete,
            "inside_box": inside,
            "execution": ex,
            "force_samples": forces,
            "force_dt": 0.04,
            "first_contact_loss_tick": first_loss,
            "slip_confirmed_tick": confirmed_at,
            "confirmation_latency_seconds": None
            if confirmed_at is None or first_loss is None
            else (confirmed_at - first_loss) / 25,
            "stopped_command_held": stopped_command is not None,
            "stopped_joints_settled": settled,
            "final_xyz": world.snapshot()["block_xyz"],
        }, frames
    finally:
        world.data.xfrc_applied[:] = 0
        world.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("outputs/models/context-bc-v1"))
    parser.add_argument("--scenarios", type=Path, default=Path("config/slip-scenarios.json"))
    parser.add_argument("--group", choices=["development", "test"], default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.scenarios.read_text())
    results = []
    for case in config[args.group]:
        pair = []
        for feedback in (False, True):
            result, frames = run_case(args.model, case, feedback)
            results.append(result)
            pair.append(frames)
            print(json.dumps(result), flush=True)
            (args.output / "partial-results.json").write_text(json.dumps(results, indent=2) + "\n")
        combined = []
        for index in range(max(map(len, pair))):
            frame = Image.new("RGB", (960, 320))
            for side in range(2):
                frame.paste(pair[side][min(index, len(pair[side]) - 1)], (480 * side, 0))
            combined.append(frame)
        combined[0].save(
            args.output / f"{case['name']}.gif",
            save_all=True,
            append_images=combined[1:],
            duration=200,
            loop=0,
        )
    checked = [r for r in results if r["controller"] == "feedback"]
    report = {
        "group": args.group,
        "results": results,
        "feedback_checks": {
            "passed": sum(r["expectation_met"] for r in checked),
            "total": len(checked),
        },
        "policy_sha256": hashlib.sha256((args.model / "policy.npz").read_bytes()).hexdigest(),
        "scenarios_sha256": hashlib.sha256(args.scenarios.read_bytes()).hexdigest(),
        "disturbance": "world-Z force pulse at cube center of mass; no pose/velocity edits during execution",
    }
    (args.output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    if not all(r["expectation_met"] for r in checked):
        raise SystemExit("slip checks failed; see evaluation.json")


if __name__ == "__main__":
    main()
