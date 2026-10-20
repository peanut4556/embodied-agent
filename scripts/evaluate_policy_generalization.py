"""Frozen Cartesian position/appearance audit, paired with optional expert recovery."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from embodied_agent.contact_watchdog import GuardedExecutor
from embodied_agent.grasp_quality import GraspQuality
from embodied_agent.memory_evaluation import verify_physics
from embodied_agent.memory_policy import MemoryPolicy, digest
from embodied_agent.physics import PhysicsWorld
from embodied_agent.reactive import ReactiveExecutor, observation_features


def cases(config):
    positions = config["positions"]
    if (
        len(set(positions)) != len(positions)
        or any(not np.isfinite(x) or not 0.28 <= x <= 0.40 for x in positions)
        or set(positions) & set(config["excluded_positions"])
    ):
        raise ValueError("positions must be new, unique and within the workspace")
    if config["modes"] != ["policy", "guarded"] or config["max_seconds"] != 30:
        raise ValueError("paired modes and fixed budget required")
    visuals = config["visuals"]
    if len({v["name"] for v in visuals}) != len(visuals):
        raise ValueError("duplicate visual conditions")
    for visual in visuals:
        if (
            not np.isfinite([visual["light_scale"], visual["camera_dx"]]).all()
            or not 0.3 <= visual["light_scale"] <= 1.5
            or abs(visual["camera_dx"]) > 0.03
        ):
            raise ValueError("invalid visual perturbation")
    return [
        {
            "name": f"x{round(x * 1000):03d}-{v['name']}",
            "x": x,
            **v,
            "visual": v["name"],
            "case_id": f"x{round(x * 1000):03d}-{v['name']}",
        }
        for x in positions
        for v in visuals
    ]


def apply_visual(world, case):
    # Alter rendering only. RGB-D expert uses the actual calibrated camera pose.
    world.model.light_diffuse[:] *= case["light_scale"]
    world.model.light_specular[:] *= case["light_scale"]
    camera = world.model.camera("perception").id
    world.model.cam_pos[camera, 0] += case["camera_dx"]
    mujoco.mj_forward(world.model, world.data)


def run(policy, case, mode, output, seconds, acquisition_seconds=None):
    world = PhysicsWorld(perception="rgbd", perception_size_mode="oriented")
    renderer = None
    try:
        world.data.qpos[5] = case["x"]
        mujoco.mj_forward(world.model, world.data)
        world.tick(0.2)
        initial = {
            key: getattr(world.data, key).tolist()
            for key in ("qpos", "qvel", "ctrl", "qacc_warmstart")
        }
        initial["time"] = float(world.data.time)
        apply_visual(world, case)
        fps = policy.metadata["fps"]
        learner = ReactiveExecutor(policy.session(), world.data.qpos[:5], max_seconds=seconds)
        controller = (
            GuardedExecutor(world, learner, acquisition_seconds=acquisition_seconds)
            if mode != "policy"
            else None
        )
        renderer = mujoco.Renderer(world.model, height=240, width=320)
        quality = GraspQuality(fps)
        actions, states, features = [], [], []
        reason = ""
        name = f"{case['case_id']}-{mode}"
        for tick in range(seconds * fps):
            mujoco.mj_forward(world.model, world.data)
            renderer.update_scene(world.data, camera="perception")
            rgb = renderer.render().copy()
            if tick == 0:
                Image.fromarray(rgb).save(output / f"{name}-initial.png")
            joints, holding = world.data.qpos[:5].copy(), world.holding
            features.append(observation_features(rgb, joints, holding))
            command = controller.step(rgb) if controller else learner.step(rgb, joints, holding)
            quality.observe(
                holding, world.data.body("red_block").xpos.copy(), command, world.inside_box()
            )
            actions.append(command.copy())
            states.append(world.data.qpos.copy())
            world.data.ctrl[:] = command
            for _ in range(round(1 / fps / world.model.opt.timestep)):
                mujoco.mj_step(world.model, world.data)
            reason = controller.reason if controller else learner.reason
            if quality.result()["verified_pick_place"] or reason:
                break
        else:
            reason = "execution timeout"
            # Explicitly stop the controller at the same budget boundary.
            if controller:
                controller.step(rgb)
            else:
                learner.stop(world.data.qpos[:5], reason)
        renderer.update_scene(world.data, camera="perception")
        Image.fromarray(renderer.render()).save(output / f"{name}-final.png")
        trace = output / f"{name}.npz"
        np.savez_compressed(trace, actions=actions, qpos=states, features=features)
        result = quality.result()
        takeover = controller.takeover_tick if controller else None
        failure = (
            "none"
            if result["verified_pick_place"]
            else "expert_rejected"
            if reason.startswith("expert")
            else "no_stable_grasp"
            if not result["stable_grasp_observed"]
            else "no_sustained_lift"
            if not result["sustained_lift"]
            else "no_intentional_release"
            if not result["intentional_release"]
            else "no_stable_goal"
        )
        return {
            "case_id": case["case_id"],
            "x": case["x"],
            "visual": case["visual"],
            "mode": mode,
            "initial_state": initial,
            "takeover_tick": takeover,
            "watchdog_events": controller.events if controller else [],
            "learner_ticks": learner.ticks,
            "reason": reason,
            "failure": failure,
            "frames": len(actions),
            "trace": trace.name,
            "trace_sha256": digest(trace),
            "initial_image_sha256": digest(output / f"{name}-initial.png"),
            "quality": result,
        }
    finally:
        if renderer:
            renderer.close()
        world.close()


def summarize(rows):
    result = {}
    for mode in ("policy", "guarded"):
        group = [r for r in rows if r["mode"] == mode]
        result[mode] = {}
        for visual in ["all"] + sorted({r["visual"] for r in group}):
            subset = [r for r in group if visual == "all" or r["visual"] == visual]
            successes = sum(r["quality"]["verified_pick_place"] for r in subset)
            result[mode][visual] = {
                "successes": successes,
                "cases": len(subset),
                "success_rate": successes / len(subset) if subset else None,
                "takeovers": sum(r["takeover_tick"] is not None for r in subset),
                "successful_after_takeover": sum(
                    r["takeover_tick"] is not None and r["quality"]["verified_pick_place"]
                    for r in subset
                ),
                "failures": {
                    kind: sum(r["failure"] == kind for r in subset)
                    for kind in sorted({r["failure"] for r in subset})
                    if kind != "none"
                },
            }
    return result


def paired_checks(rows, output):
    checks = []
    for case in sorted({r["case_id"] for r in rows}):
        a, b = [
            next(r for r in rows if r["case_id"] == case and r["mode"] == mode)
            for mode in ("policy", "guarded")
        ]
        with np.load(output / a["trace"]) as ta, np.load(output / b["trace"]) as tb:
            prefix = b["takeover_tick"]
            if prefix is None:
                prefix = min(len(ta["actions"]), len(tb["actions"]))
            equal = all(
                np.array_equal(ta[key][:prefix], tb[key][:prefix])
                for key in ("actions", "qpos", "features")
            )
        checks.append(
            {
                "case_id": case,
                "equal_initial_state": a["initial_state"] == b["initial_state"],
                "equal_initial_image": a["initial_image_sha256"] == b["initial_image_sha256"],
                "equal_pre_takeover_prefix": equal,
                "baseline_success_lost": a["quality"]["verified_pick_place"]
                and not b["quality"]["verified_pick_place"],
            }
        )
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/policy-generalization.json")
    parser.add_argument("--output", default="outputs/evaluations/policy-generalization-v1")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    grid = cases(config)
    for path, expected in config["inventory_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"position inventory changed: {path}")
    policy = MemoryPolicy(config["policy"])
    if policy.metadata["weights_sha256"] != config["weights_sha256"]:
        raise ValueError("frozen checkpoint changed")
    verify_physics(policy)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "config": config,
        "config_sha256": digest(args.config),
        "results": [],
        "limitations": "24 correlated grid scenarios at 6 positions; descriptive rates only, no population confidence interval. Correct camera calibration is supplied to expert. No retraining.",
    }
    report["source_sha256"] = {
        str(p): digest(p)
        for p in [
            Path(__file__),
            Path("src/embodied_agent/contact_watchdog.py"),
            Path("src/embodied_agent/physics.py"),
            Path("src/embodied_agent/correction_data.py"),
            Path("src/embodied_agent/grasp_quality.py"),
            Path("src/embodied_agent/assets/tabletop.xml"),
        ]
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for case in grid:
            for mode in config["modes"]:
                row = run(policy, case, mode, output, config["max_seconds"])
                report["results"].append(row)
                save()
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in ("case_id", "mode", "takeover_tick", "failure", "frames")
                        }
                    ),
                    flush=True,
                )
        report["paired_checks"] = paired_checks(report["results"], output)
        if not all(
            r[k]
            for r in report["paired_checks"]
            for k in ("equal_initial_state", "equal_initial_image", "equal_pre_takeover_prefix")
        ):
            raise RuntimeError("paired execution provenance mismatch")
        report["scores"] = summarize(report["results"])
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
