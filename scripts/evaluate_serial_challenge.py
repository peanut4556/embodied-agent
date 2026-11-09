"""Fresh force-driven RGB-D at causal single-flight request times.

Processing delays use a virtual event clock, not measured rendering throughput.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from embodied_agent.exposure_confirmation import ExposureClockMap
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.rigid_reference import RigidReferenceConfirmation
from embodied_agent.serial_challenge import SerialChallengeController
from scripts.evaluate_confirmation_challenge import force_at
from scripts.evaluate_reference_depth_coverage import paired_capture
from scripts.evaluate_rendered_precision import inside, place, sources
from scripts.evaluate_rigid_reference import CHECKPOINT, REFERENCES, make_world

MODES = ("stationary", "early_impulse", "period_200_phase_10")
DELAYS_MS = (20, 50, 80, 100, "stall_burst")
SEEDS = (202611121, 202611122)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    budget_path = Path("docs/evaluations/rendered-precision-v3.json")
    budget = json.loads(budget_path.read_text())["fitted_profiles"]["detail"]["position_error_m"]
    paths = (
        sources()
        + [
            Path(__file__),
            budget_path,
            Path("scripts/evaluate_confirmation_challenge.py"),
            Path("scripts/evaluate_reference_depth_coverage.py"),
        ]
        + [
            Path("src/embodied_agent") / name
            for name in (
                "serial_challenge.py",
                "delivered_challenge.py",
                "confirmation_challenge.py",
                "exposure_confirmation.py",
                "observation_capture.py",
                "settled_reference.py",
                "temporal_reference.py",
            )
        ]
    )
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "scope": "Fresh MuJoCo RGB-D, causal serial requests, virtual service latency; no measured camera throughput",
        "source_sha256": hashes,
        "budget_m": budget,
        "seeds": SEEDS,
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
        correction = RigidReferenceConfirmation(0, REFERENCES, CHECKPOINT)
        bid = world.model.body("red_block").id
        dt = float(world.model.opt.timestep)
        if abs(dt - 0.002) > 1e-12:
            raise ValueError("this frozen event grid requires 2 ms physics steps")
        for seed in SEEDS:
            for mi, mode in enumerate(MODES):
                for delay in DELAYS_MS:
                    world.data.xfrc_applied[:] = 0
                    place(world, (0.641, 0.002))
                    control = SerialChallengeController(
                        0,
                        ExposureClockMap(0, 0),
                        "camera",
                        position_error_m=budget,
                        delivery_budget_s=0.1,
                    )
                    job, frames, first = None, [], None
                    for step in range(1552):
                        t = step * dt
                        mujoco.mj_forward(world.model, world.data)
                        if job is not None and job[0] is not None and step >= job[0]:
                            _, req, packet = job
                            a = (
                                None
                                if packet["primary"] is None
                                else StampedPosition(tuple(packet["primary"]), req.host_time)
                            )
                            b = (
                                None
                                if packet["secondary"] is None
                                else StampedPosition(tuple(packet["secondary"]), req.host_time)
                            )
                            control.complete(req.request_id, t, t, a, b, session_id=req.session_id)
                            job = None
                        request = control.poll(t, t)
                        if request is not None:
                            if job is not None:
                                raise RuntimeError("overlapping driver calls")
                            before = (
                                world.data.qpos.copy(),
                                world.data.qvel.copy(),
                                float(world.data.time),
                            )
                            rng = np.random.default_rng(np.random.SeedSequence([seed, mi, step]))
                            packet = paired_capture(world, correction, "depth_noise_1mm", rng)[
                                "coverage"
                            ]
                            if not (
                                np.array_equal(before[0], world.data.qpos)
                                and np.array_equal(before[1], world.data.qvel)
                                and before[2] == world.data.time
                            ):
                                raise RuntimeError("capture altered physical state")
                            stalled = (
                                delay == "stall_burst"
                                and control.check.evidence.phase == "challenge"
                            )
                            service = 0.02 if delay == "stall_burst" else delay / 1000
                            deliver_step = None if stalled else step + round(service / dt)
                            job = (deliver_step, request, packet)
                            frames.append(
                                {
                                    "request_id": request.request_id,
                                    "exposure_s": t,
                                    "planned_delivery_s": None
                                    if deliver_step is None
                                    else deliver_step * dt,
                                    "truth_xyz": world.data.body("red_block").xpos.tolist(),
                                    "packet": packet,
                                }
                            )
                        if control.status != "pending":
                            if control.status == "confirmed_in_tray":
                                xyz = world.data.body("red_block").xpos.tolist()
                                first = {
                                    "delivery_s": t,
                                    "truth_xyz_at_delivery": xyz,
                                    "inside_at_delivery": inside(xyz),
                                }
                            break
                        world.data.xfrc_applied[bid, 0] = force_at(mode, t)
                        mujoco.mj_step(world.model, world.data)
                    world.data.xfrc_applied[:] = 0
                    if control.status == "pending":
                        raise RuntimeError("controller failed to terminate")
                    report["cases"].append(
                        {
                            "mode": mode,
                            "seed": seed,
                            "service_ms": delay,
                            "result": control.result(),
                            "first_confirmation": first,
                            "frames": frames,
                        }
                    )
                    save()
                    print(
                        seed, mode, delay, control.status, control.reason, len(frames), flush=True
                    )
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
