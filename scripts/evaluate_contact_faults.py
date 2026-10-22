"""Monitor-only contact faults with independent, unmodified physical scoring."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from embodied_agent.contact_watchdog import GuardedExecutor
from embodied_agent.grasp_quality import GraspQuality
from embodied_agent.memory_policy import digest
from embodied_agent.physics import PhysicsWorld
from scripts.audit_contact_watchdog import advance, restore
from scripts.evaluate_recovery_positives import ReplayLearner


def observed_contact(truth, tick, fault):
    if fault["value"] is not None and fault["start"] <= tick < fault["end"]:
        return fault["value"]
    return bool(truth)


def validate(config):
    names = set()
    for fault in config["faults"]:
        if fault["name"] in names or (
            fault["value"] is not None and type(fault["value"]) is not bool
        ):
            raise ValueError("invalid fault name or value")
        if (
            any(type(fault[k]) is not int for k in ("start", "end"))
            or not 0 <= fault["start"] < fault["end"] <= 750
        ):
            raise ValueError("invalid fault window")
        names.add(fault["name"])
    if not names or len({s["name"] for s in config["sources"]}) != len(config["sources"]):
        raise ValueError("invalid sources or empty faults")


def run(initial, actions, source, fault, output, baseline=False):
    world = PhysicsWorld(perception="rgbd", perception_size_mode="oriented")
    try:
        restore(world, initial)
        learner = ReplayLearner(actions, 1)
        guard = None if baseline else GuardedExecutor(world, learner, 4, descent_grace=True)
        quality = GraspQuality(25)
        commands, states, truth_rows, observed, active = [], [], [], [], []
        velocities, controls = [], []
        for tick in range(750):
            mujoco.mj_forward(world.model, world.data)
            truth = bool(world.holding)
            signal = observed_contact(truth, tick, fault)
            monitoring = guard is not None and guard.expert is None and not guard.reason
            command = (
                guard.step(None, monitor_holding=signal)
                if guard
                else learner.step(None, world.data.qpos[:5], truth)
            )
            # Never feed the injected signal to quality, learner or recovery expert.
            quality.observe(
                truth, world.data.body("red_block").xpos.copy(), command, world.inside_box()
            )
            commands.append(command.copy())
            states.append(world.data.qpos.copy())
            velocities.append(world.data.qvel.copy())
            controls.append(world.data.ctrl.copy())
            truth_rows.append(truth)
            observed.append(signal)
            active.append(monitoring)
            advance(world, command, 25)
            if quality.result()["verified_pick_place"] or (guard and guard.reason):
                break
        else:
            if guard:
                guard.step(None)  # Latch the shared 30-second budget boundary.
            else:
                learner.stop(world.data.qpos[:5], "execution timeout")
        name = source + "-" + ("baseline" if baseline else fault["name"])
        path = output / (name + ".npz")
        np.savez_compressed(
            path,
            actions=commands,
            qpos=states,
            qvel=velocities,
            ctrl=controls,
            true_holding=truth_rows,
            monitor_holding=observed,
            monitor_active=active,
        )
        affected = sum(t != o and a for t, o, a in zip(truth_rows, observed, active, strict=True))
        return {
            "name": name,
            "source": source,
            "fault": None if baseline else fault,
            "baseline": baseline,
            "frames": len(commands),
            "takeover_tick": guard.takeover_tick if guard else None,
            "events": guard.events if guard else [],
            "reason": guard.reason if guard else learner.reason,
            "effective_changed_monitor_frames": affected,
            "quality": quality.result(),
            "trace": path.name,
            "trace_sha256": digest(path),
        }
    finally:
        world.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/contact-faults.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    validate(config)
    for path, expected in config["inputs_sha256"].items():
        if digest(path) != expected:
            raise ValueError("frozen input changed: " + path)
    initial = json.loads(Path(config["manifest"]).read_text())["episodes"][5]["initial_state"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        Path("scripts/evaluate_recovery_positives.py"),
        Path("scripts/audit_contact_watchdog.py"),
    ]
    sources += [
        Path("src/embodied_agent") / p
        for p in (
            "contact_watchdog.py",
            "physics.py",
            "grasp_quality.py",
            "correction_data.py",
            "assets/tabletop.xml",
        )
    ]
    report = {
        "status": "running",
        "scope": "Monitor-only injected contact values; one known initial scene, two recorded action traces. Not policy generalization or end-to-end sensor failure.",
        "config": config,
        "config_sha256": digest(args.config),
        "source_sha256": {str(p): digest(p) for p in sources},
        "results": [],
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for source in config["sources"]:
            with np.load(source["trace"]) as data:
                actions = data["actions"].copy()
            nominal = next(f for f in config["faults"] if f["value"] is None)
            for fault, baseline in [(nominal, True)] + [(f, False) for f in config["faults"]]:
                row = run(initial, actions, source["name"], fault, output, baseline)
                report["results"].append(row)
                save()
                print(
                    row["name"],
                    row["quality"]["verified_pick_place"],
                    row["takeover_tick"],
                    row["effective_changed_monitor_frames"],
                    flush=True,
                )
        report["checks"] = []
        for row in report["results"]:
            base = next(
                r for r in report["results"] if r["source"] == row["source"] and r["baseline"]
            )
            with (
                np.load(output / row["trace"]) as data,
                np.load(output / base["trace"]) as original,
            ):
                prefix = row["takeover_tick"]
                same = all(
                    np.array_equal(data[k][:prefix], original[k][:prefix])
                    for k in ("actions", "qpos", "true_holding")
                )
                independent = GraspQuality(25)
                # Replay physical score from stored qpos and commands, using fresh contact computation.
                world = PhysicsWorld()
                try:
                    for state, velocity, control, command, truth in zip(
                        data["qpos"],
                        data["qvel"],
                        data["ctrl"],
                        data["actions"],
                        data["true_holding"],
                        strict=True,
                    ):
                        world.data.qpos[:] = state
                        world.data.qvel[:] = velocity
                        world.data.ctrl[:] = control
                        mujoco.mj_forward(world.model, world.data)
                        if bool(world.holding) != bool(truth):
                            raise RuntimeError("contact truth replay mismatch")
                        independent.observe(
                            bool(world.holding),
                            world.data.body("red_block").xpos.copy(),
                            command,
                            world.inside_box(),
                        )
                finally:
                    world.close()
                finite = bool(np.isfinite(data["actions"]).all())
            check = {
                "name": row["name"],
                "same_pre_takeover_or_entire_trace": same,
                "physical_quality_recomputed": independent.result() == row["quality"],
                "hash_matches": digest(output / row["trace"]) == row["trace_sha256"],
                "finite_commands": finite,
            }
            report["checks"].append(check)
            if not all(v for k, v in check.items() if k != "name"):
                raise RuntimeError("audit check failed")
        if any(digest(p) != expected for p, expected in report["source_sha256"].items()):
            raise RuntimeError("source changed during audit")
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
