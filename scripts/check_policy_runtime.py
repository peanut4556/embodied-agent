"""Rendered checks of the production physics adapter using a local frozen checkpoint.

Relocations below are evaluation-only scene changes, never controller commands.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from embodied_agent.physics import PhysicsWorld


def check(model):
    results = []
    for case in ("normal", "relocate", "outside_range", "stop"):
        world = PhysicsWorld(policy_path=model)
        try:
            world.begin("learned_pick_place", {"object": "red_block", "destination": "box"})
            for tick in range(800):
                if tick == 25 and case in {"relocate", "outside_range"}:
                    world.data.qpos[5:8] = [0.38 if case == "relocate" else 0.46, 0, 0.027]
                    world.data.qpos[8:12] = [1, 0, 0, 0]
                    world.data.qvel[5:11] = 0
                    mujoco.mj_forward(world.model, world.data)
                if tick == 25 and case == "stop":
                    world.begin("stop", {})
                    command = world.data.ctrl.copy()
                    world.tick(1.0)
                    np.testing.assert_array_equal(world.data.ctrl, command)
                    break
                if world.tick(0.04):
                    break
            else:
                raise AssertionError("adapter timeout")
            state = world.snapshot()
            ex = state["execution"]
            if case in {"normal", "relocate"}:
                assert state["inside_box"] and ex["state"] == "completed", state
                if case == "relocate":
                    assert ex["retries"] > 0, ex
            else:
                assert ex["state"] == "stopped" and not state["active"], state
                assert ex["reason"], ex
            results.append({"case": case, "execution": ex, "inside_box": state["inside_box"]})
            print(f"PASS production adapter: {case} ({ex['retries']} retries)", flush=True)
            world.begin("reset", {})
            assert world.snapshot()["execution"] is None
        finally:
            world.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = check(args.model)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
