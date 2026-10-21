"""Physical positive controls: exact successful actions and time-stretched playback.

Playback is a diagnostic controller, not a newly successful learned policy.
"""

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


class ReplayLearner:
    fps = 25
    max_ticks = 750

    def __init__(self, actions, stretch):
        self.actions = actions
        self.stretch = stretch
        self.ticks = 0
        self.reason = ""
        self.command = actions[0].copy()

    def step(self, rgb, joints, holding):
        t = min(self.ticks / self.stretch, len(self.actions) - 1)
        index = int(t)
        fraction = t - index
        self.command = (
            self.actions[index] * (1 - fraction)
            + self.actions[min(index + 1, len(self.actions) - 1)] * fraction
        )
        self.ticks += 1
        return self.command.copy()

    def stop(self, joints, reason):
        self.reason = reason
        self.command[:3] = joints[:3]
        return self.command.copy()


def run(initial, actions, stretch, mode, output, stop_tick=None):
    if mode not in {"baseline", "fixed", "descent"} or not np.isfinite(stretch) or stretch <= 0:
        raise ValueError("invalid replay mode or stretch")
    if stop_tick is not None and (
        type(stop_tick) is not int or not 0 <= stop_tick < 725 or mode == "baseline"
    ):
        raise ValueError("stop checks require a guarded mode and a full post-stop window")
    world = PhysicsWorld(perception="rgbd", perception_size_mode="oriented")
    try:
        restore(world, initial)
        learner = ReplayLearner(actions, stretch)
        guard = (
            GuardedExecutor(world, learner, acquisition_seconds=4, descent_grace=mode == "descent")
            if mode != "baseline"
            else None
        )
        quality = GraspQuality(25)
        commands, states, tips, contacts = [], [], [], []
        for tick in range(750):
            mujoco.mj_forward(world.model, world.data)
            command = (
                guard.step(None, stop=stop_tick is not None and tick >= stop_tick)
                if guard
                else learner.step(None, world.data.qpos[:5], world.holding)
            )
            quality.observe(
                world.holding, world.data.body("red_block").xpos.copy(), command, world.inside_box()
            )
            commands.append(command.copy())
            states.append(world.data.qpos.copy())
            tips.append(world.data.site("tip").xpos[[0, 2]].copy())
            contacts.append(world.holding)
            advance(world, command, 25)
            if stop_tick is not None:
                if tick >= stop_tick + 25:
                    break
            elif quality.result()["verified_pick_place"] or (guard and guard.reason):
                break
        name = f"stretch-{stretch}-{mode}" + (f"-stop-{stop_tick}" if stop_tick is not None else "")
        path = output / f"{name}.npz"
        np.savez_compressed(path, actions=commands, qpos=states, tips=tips, holding=contacts)
        return {
            "name": name,
            "stretch": stretch,
            "mode": mode,
            "frames": len(commands),
            "stop_tick": stop_tick,
            "learner_ticks": learner.ticks,
            "takeover_tick": guard.takeover_tick if guard else None,
            "events": guard.events if guard else [],
            "deferred_ticks": getattr(guard.acquisition, "deferred_ticks", 0) if guard else 0,
            "reason": guard.reason if guard else learner.reason,
            "quality": quality.result(),
            "trace": path.name,
            "trace_sha256": digest(path),
        }
    finally:
        world.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = Path("outputs/datasets/corrections-contact-oriented-v1/recording.json")
    trace = Path("outputs/evaluations/contact-watchdog-v1-verified/parent-baseline.npz")
    initial = json.loads(manifest.read_text())["episodes"][5]["initial_state"]
    with np.load(trace) as data:
        actions = data["actions"].copy()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "Known training scene. Playback positive controls, not policy generalization.",
        "manifest_sha256": digest(manifest),
        "actions_sha256": digest(trace),
        "source_sha256": {
            str(p): digest(p)
            for p in [
                Path(__file__),
                Path("src/embodied_agent/contact_watchdog.py"),
                Path("scripts/audit_contact_watchdog.py"),
                Path("src/embodied_agent/physics.py"),
                Path("src/embodied_agent/grasp_quality.py"),
                Path("src/embodied_agent/correction_data.py"),
                Path("src/embodied_agent/assets/tabletop.xml"),
            ]
        },
        "stretches": [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5],
        "results": [],
        "status": "running",
    }
    try:
        for stretch in (1, 1.25, 1.5, 1.75, 2, 2.25, 2.5):
            for mode in ("baseline", "fixed", "descent"):
                row = run(initial, actions, stretch, mode, output)
                report["results"].append(row)
                print(
                    row["name"],
                    row["quality"]["verified_pick_place"],
                    row["takeover_tick"],
                    flush=True,
                )
        report["paired_checks"] = []
        for stretch in report["stretches"]:
            baseline = next(
                r for r in report["results"] if r["stretch"] == stretch and r["mode"] == "baseline"
            )
            for mode in ("fixed", "descent"):
                row = next(
                    r for r in report["results"] if r["stretch"] == stretch and r["mode"] == mode
                )
                with np.load(output / baseline["trace"]) as a, np.load(output / row["trace"]) as b:
                    prefix = row["takeover_tick"]
                    same = all(
                        np.array_equal(a[k][:prefix], b[k][:prefix])
                        for k in ("actions", "qpos", "tips", "holding")
                    )
                report["paired_checks"].append(
                    {"stretch": stretch, "mode": mode, "same_pre_takeover_or_entire_trace": same}
                )
                if not same:
                    raise RuntimeError("positive control pairing mismatch")
        if any(digest(p) != expected for p, expected in report["source_sha256"].items()):
            raise RuntimeError("source changed during experiment")
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
