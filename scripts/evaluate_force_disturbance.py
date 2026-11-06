"""Force-driven RGB-D confirmation audit with physics-step motion truth."""

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

MODES = ("stationary", "early_impulse", "late_impulse", "interframe_impulses")
SEEDS = (202611081, 202611082)


def force_at(mode, t):
    """World x force in N; half-open 20 ms pulses, independent of observations."""
    if mode == "stationary":
        return 0.0
    if mode in ("early_impulse", "late_impulse"):
        onset = 0.4 if mode == "early_impulse" else 1.6
        return 2.0 if onset - 1e-9 <= t < onset + 0.02 - 1e-9 else 0.0
    if mode == "interframe_impulses":
        return (
            1.2
            if any(0.05 + i * 0.2 - 1e-9 <= t < 0.07 + i * 0.2 - 1e-9 for i in range(15))
            else 0.0
        )
    raise ValueError("unknown force schedule")


def motion_summary(rows, end, duration=1.0):
    """Independent audit only: never supplied to a confirmation policy."""
    window = [r for r in rows if end - duration - 1e-9 <= r["elapsed_s"] <= end + 1e-9]
    points = np.array([r["xyz"] for r in window])
    return {
        "peak_speed_m_s": max(r["speed_m_s"] for r in window),
        "max_axis_excursion_m": float(np.ptp(points, axis=0).max()),
        "physics_samples": len(window),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-interval", type=float, choices=(0.2, 0.05), default=0.2)
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
        "source_sha256": hashes,
        "seeds": SEEDS,
        "frame_interval_s": args.frame_interval,
        "budget_m": budget,
        "scope": "MuJoCo force-driven dynamics, paired RGB-D, 1 mm pixel depth noise; truth audit only",
        "cases": [],
    }

    def save():
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    world = None
    save()
    try:
        world = make_world("nominal")
        world.camera_renderer = mujoco.Renderer(world.model, height=480, width=640)
        bid = world.model.body("red_block").id
        dof = int(world.model.jnt_dofadr[world.model.body_jntadr[bid]])
        correction = RigidReferenceConfirmation(0, REFERENCES, CHECKPOINT)
        dt = float(world.model.opt.timestep)
        steps_per_frame = round(args.frame_interval / dt)
        if abs(steps_per_frame * dt - args.frame_interval) > 1e-12:
            raise ValueError("frame interval must be an integer physics-step count")
        for seed in SEEDS:
            for mi, mode in enumerate(MODES):
                world.data.xfrc_applied[:] = 0
                place(world, (0.641, 0.002))
                start = float(world.data.time)
                checks = {
                    "short_window": TemporalReferenceConfirmation(start, position_error_m=budget),
                    "settled": SettledReferenceConfirmation(start, position_error_m=budget),
                }
                rng = np.random.default_rng(seed + mi * 1000)
                first, samples, truth = {k: None for k in checks}, [], []
                for step in range(round(3 / dt) + 1):
                    elapsed = step * dt
                    mujoco.mj_forward(world.model, world.data)
                    xyz = world.data.body("red_block").xpos.tolist()
                    speed = float(np.linalg.norm(world.data.qvel[dof : dof + 3]))
                    truth.append(
                        {
                            "elapsed_s": elapsed,
                            "xyz": xyz,
                            "speed_m_s": speed,
                            "force_x_n": force_at(mode, elapsed),
                        }
                    )
                    if step % steps_per_frame == 0:
                        before = (
                            world.data.qpos.copy(),
                            world.data.qvel.copy(),
                            float(world.data.time),
                        )
                        packets = paired_capture(world, correction, "depth_noise_1mm", rng)
                        if not (
                            np.array_equal(before[0], world.data.qpos)
                            and np.array_equal(before[1], world.data.qvel)
                            and before[2] == world.data.time
                        ):
                            raise RuntimeError("capture altered physical state")
                        for key, check in checks.items():
                            p = packets["coverage"]
                            check.step(start + elapsed, p["primary"], p["secondary"])
                            if first[key] is None and check.status == "confirmed_in_tray":
                                first[key] = {
                                    "elapsed_s": elapsed,
                                    "inside": inside(xyz),
                                    "instantaneous_speed_m_s": speed,
                                    "preceding_second": motion_summary(truth, elapsed),
                                }
                        samples.append({"elapsed_s": elapsed, "profiles": packets})
                    if step < round(3 / dt):
                        world.data.xfrc_applied[bid, 0] = force_at(mode, elapsed)
                        mujoco.mj_step(world.model, world.data)
                world.data.xfrc_applied[:] = 0
                report["cases"].append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "first_confirmation": first,
                        "policies": {k: c.result() for k, c in checks.items()},
                        "samples": samples,
                        "physics_truth": truth,
                    }
                )
                save()
                print(seed, mode, first, flush=True)
        if hashes != {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}:
            raise RuntimeError("source changed during evaluation")
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        if world is not None:
            world.data.xfrc_applied[:] = 0
            world.close()
        save()


if __name__ == "__main__":
    main()
