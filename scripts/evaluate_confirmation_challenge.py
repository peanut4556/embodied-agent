"""Candidate-triggered high-rate challenge against uniform capture policies."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from embodied_agent.confirmation_challenge import ConfirmationChallenge
from embodied_agent.observation_schedule import observation_times_ms
from embodied_agent.rigid_reference import RigidReferenceConfirmation
from embodied_agent.settled_reference import SettledReferenceConfirmation
from scripts.evaluate_reference_depth_coverage import paired_capture
from scripts.evaluate_rendered_precision import inside, place, sources
from scripts.evaluate_rigid_reference import CHECKPOINT, REFERENCES, make_world

SCHEDULES = {k: observation_times_ms(k) for k in ("uniform_5hz", "uniform_20hz")}
MODES = (
    "stationary",
    "early_impulse",
    "late_impulse",
    "period_200_phase_10",
    "period_178_phase_34",
    "period_226_phase_82",
    "period_274_phase_126",
)
SEEDS = (202611111, 202611112)


def force_at(mode, t):
    if mode == "stationary":
        return 0.0
    if mode in ("early_impulse", "late_impulse"):
        onset = 0.4 if mode == "early_impulse" else 1.6
        return 2.0 if onset - 1e-9 <= t < onset + 0.02 - 1e-9 else 0.0
    if mode not in MODES:
        raise ValueError("unknown force schedule")
    parts = mode.split("_")
    period, phase = int(parts[1]) / 1000, int(parts[3]) / 1000
    return (
        1.2
        if any(phase + i * period - 1e-9 <= t < phase + i * period + 0.02 - 1e-9 for i in range(20))
        else 0.0
    )


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
    args = parser.parse_args()
    budget_path = Path("docs/evaluations/rendered-precision-v3.json")
    budget = json.loads(budget_path.read_text())["fitted_profiles"]["detail"]["position_error_m"]
    paths = sources() + [
        Path(__file__),
        budget_path,
        Path("scripts/evaluate_reference_depth_coverage.py"),
        Path("src/embodied_agent/temporal_reference.py"),
        Path("src/embodied_agent/observation_schedule.py"),
        Path("src/embodied_agent/confirmation_challenge.py"),
        Path("src/embodied_agent/settled_reference.py"),
    ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "source_sha256": hashes,
        "seeds": SEEDS,
        "schedules_ms": SCHEDULES,
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
        schedules = {k: {round(t / 1000 / dt) for t in SCHEDULES[k]} for k in SCHEDULES}
        for times in SCHEDULES.values():
            if any(abs(round(t / 1000 / dt) * dt - t / 1000) > 1e-12 for t in times):
                raise ValueError("schedule must align with physics steps")
        for seed in SEEDS:
            for mi, mode in enumerate(MODES):
                world.data.xfrc_applied[:] = 0
                place(world, (0.641, 0.002))
                start = float(world.data.time)
                checks = {
                    k: SettledReferenceConfirmation(start, position_error_m=budget)
                    for k in SCHEDULES
                }
                checks["challenge"] = ConfirmationChallenge(start, position_error_m=budget)
                capture_counts = {k: 0 for k in checks}
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
                    due = [
                        k
                        for k, c in checks.items()
                        if c.status == "pending"
                        and (
                            (k == "challenge" and start + elapsed >= c.next_capture_time - 1e-9)
                            or (k != "challenge" and step in schedules[k])
                        )
                    ]
                    if due:
                        before = (
                            world.data.qpos.copy(),
                            world.data.qvel.copy(),
                            float(world.data.time),
                        )
                        rng = np.random.default_rng(np.random.SeedSequence([seed, mi, step]))
                        packets = paired_capture(world, correction, "depth_noise_1mm", rng)
                        if not (
                            np.array_equal(before[0], world.data.qpos)
                            and np.array_equal(before[1], world.data.qvel)
                            and before[2] == world.data.time
                        ):
                            raise RuntimeError("capture altered physical state")
                        for key, check in checks.items():
                            if key not in due:
                                continue
                            capture_counts[key] += 1
                            p = packets["coverage"]
                            check.step(start + elapsed, p["primary"], p["secondary"])
                            if first[key] is None and check.status == "confirmed_in_tray":
                                first[key] = {
                                    "elapsed_s": elapsed,
                                    "inside": inside(xyz),
                                    "instantaneous_speed_m_s": speed,
                                    "preceding_second": motion_summary(truth, elapsed),
                                }
                        samples.append(
                            {"elapsed_s": elapsed, "recipients": due, "profiles": packets}
                        )
                    if step < round(3 / dt):
                        world.data.xfrc_applied[bid, 0] = force_at(mode, elapsed)
                        mujoco.mj_step(world.model, world.data)
                world.data.xfrc_applied[:] = 0
                report["cases"].append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "first_confirmation": first,
                        "capture_counts": capture_counts,
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
