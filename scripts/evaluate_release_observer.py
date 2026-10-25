"""Shadow RGB-D confirmation during complete known-scene action playback."""

import argparse
import json
from pathlib import Path

import numpy as np

from embodied_agent.contact_watchdog import GuardedExecutor
from embodied_agent.grasp_quality import GraspQuality
from embodied_agent.memory_policy import digest
from embodied_agent.physics import PhysicsWorld
from embodied_agent.release_observer import ReleaseObserver
from scripts.audit_contact_watchdog import advance, restore
from scripts.evaluate_contact_faults import ContactFaultSignal
from scripts.evaluate_recovery_positives import ReplayLearner


def run(initial, actions, lag, missing, enabled, path, view="original"):
    if view not in {"original", "offset"}:
        raise ValueError("unknown observation view")
    world = PhysicsWorld(perception="rgbd", perception_size_mode="oriented")
    try:
        # Diagnostic virtual camera only; geometry and control inputs are unchanged.
        if view == "offset":
            world.model.camera("perception").pos[:] = [0.64, 0.35, 1.2]
        restore(world, initial)
        guard = GuardedExecutor(world, ReplayLearner(actions, 1), 4, True, True)
        observer = ReleaseObserver()
        quality = GraspQuality(25)
        fault = {"start": 0, "end": 750, "value": None}
        if lag:
            fault.update(kind="release_lag", frames=lag, value=True)
        signal = ContactFaultSignal(fault)
        states, commands = [], []
        release_tick = None
        for tick in range(750):
            command = guard.step(
                None, monitor_holding=signal.observe(world.holding, tick, world.data.ctrl)
            )
            if release_tick is None and any(
                e["event"] == "intentional_release" for e in guard.events
            ):
                release_tick = tick
            if enabled:

                def locate(current=tick, released=release_tick):
                    if missing == "all" or (missing == "early" and current - released < 15):
                        raise ValueError("injected missing observation")
                    return world.locate_block().tolist()

                observer.step(float(world.data.time), guard.events, locate)
            quality.observe(
                world.holding, world.data.body("red_block").xpos.copy(), command, world.inside_box()
            )
            states.append(world.data.qpos.copy())
            commands.append(command.copy())
            # Fixed post-release horizon for both branches, independent of observer outcome.
            if guard.reason or (release_tick is not None and tick >= release_tick + 80):
                break
            advance(world, command, 25)
        np.savez_compressed(path, qpos=states, actions=commands)
        return {
            "release_tick": release_tick,
            "takeover_tick": guard.takeover_tick,
            "events": guard.events,
            "reason": guard.reason,
            "frames": len(states),
            "confirmation": observer.status,
            "observations": observer.rows,
            "quality": quality.result(),
            "trace": path.name,
            "trace_sha256": digest(path),
        }
    finally:
        world.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--view", choices=("original", "offset"), default="original")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config_path = Path("config/contact-delayed-release.json")
    config = json.loads(config_path.read_text())
    for path, expected in config["inputs_sha256"].items():
        assert digest(path) == expected, path
    initial = json.loads(Path(config["manifest"]).read_text())["episodes"][5]["initial_state"]
    with np.load(config["sources"][0]["trace"]) as data:
        actions = data["actions"].copy()
    paths = [
        Path(__file__),
        config_path,
        Path("scripts/audit_contact_watchdog.py"),
        Path("scripts/evaluate_contact_faults.py"),
        Path("scripts/evaluate_recovery_positives.py"),
    ]
    paths += [
        Path("src/embodied_agent") / name
        for name in (
            "release_observer.py",
            "release_confirmation.py",
            "contact_watchdog.py",
            "physics.py",
            "perception.py",
            "grasp_quality.py",
            "assets/tabletop.xml",
        )
    ]
    hashes = {str(p): digest(p) for p in paths}
    report = {
        "status": "running",
        "view": args.view,
        "camera_position": [0.64, 0.35, 1.2] if args.view == "offset" else [0.4, 0, 1.2],
        "source_sha256": hashes,
        "inputs_sha256": config["inputs_sha256"],
        "cases": [],
    }
    for name, lag, missing in [
        ("nominal", 0, "none"),
        ("lag9", 9, "none"),
        ("early_missing", 0, "early"),
        ("all_missing", 0, "all"),
    ]:
        pair = []
        for enabled in (False, True):
            pair.append(
                run(
                    initial,
                    actions,
                    lag,
                    missing,
                    enabled,
                    output / f"{name}-{enabled}.npz",
                    args.view,
                )
            )
        with np.load(output / pair[0]["trace"]) as off, np.load(output / pair[1]["trace"]) as on:
            same = all(np.array_equal(off[k], on[k]) for k in ("qpos", "actions"))
        assert same, "shadow observer changed physical execution"
        assert pair[0]["quality"] == pair[1]["quality"]
        expected = "unconfirmed" if missing == "all" else "confirmed_in_tray"
        report["cases"].append(
            {
                "name": name,
                "identical_execution": same,
                "expected": expected,
                "matches_expected": pair[1]["confirmation"] == expected,
                "off": pair[0],
                "on": pair[1],
            }
        )
        print(name, pair[1]["confirmation"], same, flush=True)
    assert hashes == {str(p): digest(p) for p in paths}
    report["status"] = "complete"
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert all(c["matches_expected"] for c in report["cases"]), (
        "confirmation outcomes differ from expectations"
    )


if __name__ == "__main__":
    main()
