"""Prescribed moving-pose RGB-D comparison, not free-body dynamics."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from embodied_agent.rigid_reference import RigidReferenceConfirmation
from embodied_agent.settled_reference import SettledReferenceConfirmation
from embodied_agent.temporal_reference import TemporalReferenceConfirmation
from scripts.evaluate_reference_depth_coverage import paired_capture
from scripts.evaluate_rendered_precision import inside, place, sources
from scripts.evaluate_rigid_reference import CHECKPOINT, REFERENCES, make_world

DRIFT_SPEEDS = {
    "drift_0_4mm_s": 0.0004,
    "drift_0_8mm_s": 0.0008,
    "drift_1_2mm_s": 0.0012,
    "drift_1_6mm_s": 0.0016,
}
MODES = ("stationary", "slow_drift", "stop_after_0_6s", "depart_before_confirmation", *DRIFT_SPEEDS)
SEEDS = (202611071, 202611072)


def trajectory(mode, t):
    if mode in DRIFT_SPEEDS:
        speed = DRIFT_SPEEDS[mode]
        return 0.641 + speed * t, speed
    if mode == "stationary":
        return 0.641, 0
    if mode == "slow_drift":
        return 0.641 + 0.002 * t, 0.002
    if mode == "stop_after_0_6s":
        return 0.641 + 0.006 * min(t, 0.6), 0.006 if t < 0.6 else 0
    if mode == "depart_before_confirmation":
        return 0.686 + 0.006 * t, 0.006
    raise ValueError("unknown trajectory")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    budget_path = Path("docs/evaluations/rendered-precision-v3.json")
    budget = json.loads(budget_path.read_text())["fitted_profiles"]["detail"]["position_error_m"]
    paths = sources() + [
        Path(__file__),
        budget_path,
        Path("scripts/evaluate_reference_depth_coverage.py"),
        Path("src/embodied_agent/temporal_reference.py"),
        Path("src/embodied_agent/settled_reference.py"),
    ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "scope": "prescribed kinematic pose rendering with 1 mm pixel depth noise",
        "budget_m": budget,
        "source_sha256": hashes,
        "seeds": SEEDS,
        "cases": [],
    }

    def save():
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    save()
    world = None
    try:
        world = make_world("nominal")
        world.camera_renderer = mujoco.Renderer(world.model, height=480, width=640)
        correction = RigidReferenceConfirmation(0, REFERENCES, CHECKPOINT)
        for seed in SEEDS:
            for mi, mode in enumerate(MODES):
                place(world, (trajectory(mode, 0)[0], 0.002))
                start = float(world.data.time)
                checks = {
                    "short_window": TemporalReferenceConfirmation(start, position_error_m=budget),
                    "settled": SettledReferenceConfirmation(start, position_error_m=budget),
                }
                rng = np.random.default_rng(seed + mi * 1000)
                first = {k: None for k in checks}
                samples = []
                for tick in range(16):
                    elapsed = tick * 0.2
                    x, speed = trajectory(mode, elapsed)
                    world.data.qpos[5] = x
                    world.data.qvel[:] = 0
                    world.data.time = start + elapsed
                    mujoco.mj_forward(world.model, world.data)
                    before = world.data.qpos.copy()
                    packets = paired_capture(world, correction, "depth_noise_1mm", rng)
                    if not np.array_equal(before, world.data.qpos):
                        raise RuntimeError("acquisition altered prescribed pose")
                    truth = world.data.body("red_block").xpos.tolist()
                    for key, check in checks.items():
                        p = packets["coverage"]
                        check.step(start + elapsed, p["primary"], p["secondary"])
                        if first[key] is None and check.status == "confirmed_in_tray":
                            first[key] = {
                                "elapsed_s": elapsed,
                                "true_speed_m_s": speed,
                                "inside": inside(truth),
                                "continued_drift": speed > 0.001,
                            }
                    samples.append(
                        {
                            "time": start + elapsed,
                            "truth_xyz": truth,
                            "true_speed_m_s": speed,
                            "profiles": packets,
                        }
                    )
                report["cases"].append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "first_confirmation": first,
                        "policies": {k: c.result() for k, c in checks.items()},
                        "samples": samples,
                    }
                )
                save()
                print(seed, mode, first, flush=True)
        report["summary"] = {
            mode: {
                k: {
                    "confirmations": sum(
                        c["first_confirmation"][k] is not None
                        for c in report["cases"]
                        if c["mode"] == mode
                    ),
                    "while_drifting": sum(
                        c["first_confirmation"][k] is not None
                        and c["first_confirmation"][k]["continued_drift"]
                        for c in report["cases"]
                        if c["mode"] == mode
                    ),
                    "outside_confirmations": sum(
                        c["first_confirmation"][k] is not None
                        and not c["first_confirmation"][k]["inside"]
                        for c in report["cases"]
                        if c["mode"] == mode
                    ),
                }
                for k in checks
            }
            for mode in MODES
        }
        if hashes != {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}:
            raise RuntimeError("source changed during evaluation")
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        save()
        if world is not None:
            world.close()
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
