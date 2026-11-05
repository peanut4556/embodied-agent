"""Fresh seeded RGB-D noise evaluation of guarded temporal filtering."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from embodied_agent.cross_view import CrossViewConfirmation
from embodied_agent.rigid_reference import RigidReferenceConfirmation
from embodied_agent.temporal_reference import GuardedMedianPair, TemporalReferenceConfirmation
from scripts.evaluate_reference_depth_coverage import paired_capture
from scripts.evaluate_rendered_precision import inside, place, sources
from scripts.evaluate_rigid_reference import CHECKPOINT, REFERENCES, make_world

SEEDS = (202611061, 202611062, 202611063)
POSITIONS = ((0.641, 0.002), (0.688, 0.002), (0.692, 0.002))


def score(samples, budget):
    start = samples[0]["time"]
    checks = {
        "raw": CrossViewConfirmation(start, position_error_m=budget),
        "temporal": TemporalReferenceConfirmation(start, position_error_m=budget),
    }
    metric_filter = GuardedMedianPair(budget)
    matched_raw, filtered, eligible, rejected = 0, 0, 0, 0
    errors = {"raw": [0.0] * 3, "temporal": [0.0] * 3}
    first_truth = dict.fromkeys(checks)
    first_time = dict.fromkeys(checks)
    for row in samples:
        packet = row["profiles"]["coverage"]
        pair = (packet["primary"], packet["secondary"])
        for name, check in checks.items():
            check.step(row["time"], *pair)
            if check.status == "confirmed_in_tray" and first_truth[name] is None:
                first_truth[name] = inside(row["truth_xyz"])
                first_time[name] = row["time"] - start
        output = metric_filter.observe(row["time"], *pair)
        if output[0] is None:
            rejected += 1
            continue
        for raw, smooth in zip(pair, output):
            eligible += 1
            raw_error = [abs(a - b) for a, b in zip(raw, row["truth_xyz"])]
            smoothed_error = [abs(a - b) for a, b in zip(smooth, row["truth_xyz"])]
            matched_raw += int(any(e > b for e, b in zip(raw_error, budget)))
            filtered += int(any(e > b for e, b in zip(smoothed_error, budget)))
            for key, error in (("raw", raw_error), ("temporal", smoothed_error)):
                errors[key] = [max(a, b) for a, b in zip(errors[key], error)]
    return {
        "policies": {k: c.result() for k, c in checks.items()},
        "false_confirmations": {k: v is False for k, v in first_truth.items()},
        "first_confirmation_s": first_time,
        "matched_views": eligible,
        "no_filtered_output_frames": rejected,
        "matched_raw_exceeded": matched_raw,
        "filtered_exceeded": filtered,
        "matched_max_error_m": errors if eligible else None,
        "metric_filter": metric_filter.result(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta_path = Path("docs/evaluations/reference-depth-coverage-v1.json")
    meta = json.loads(meta_path.read_text())
    prior_path = Path(meta["full_report"]["path"])
    raw = prior_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta["full_report"]["sha256"]:
        raise ValueError("development evidence hash mismatch")
    prior = json.loads(raw)
    budget = prior["budget_m"]
    paths = sources() + [
        Path(__file__),
        meta_path,
        Path("scripts/evaluate_reference_depth_coverage.py"),
        Path("src/embodied_agent/temporal_reference.py"),
    ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "source_sha256": hashes,
        "development_sha256": hashlib.sha256(raw).hexdigest(),
        "seeds": SEEDS,
        "positions": POSITIONS,
        "budget_m": budget,
        "cases": [],
    }
    report["development"] = [
        {"initial_xy": c["initial_xy"], **score(c["samples"], budget)}
        for c in prior["cases"]
        if c["fault"] == "depth_noise_1mm"
    ]

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
            for index, xy in enumerate(POSITIONS):
                place(world, xy)
                start = float(world.data.time)
                rng = np.random.default_rng(seed + index * 1000)
                samples = []
                for tick in range(16):
                    before = world.data.qpos.copy()
                    packets = paired_capture(world, correction, "depth_noise_1mm", rng)
                    if not np.array_equal(before, world.data.qpos):
                        raise RuntimeError("capture altered physical state")
                    samples.append(
                        {
                            "time": start + tick * 0.2,
                            "truth_xyz": world.data.body("red_block").xpos.tolist(),
                            "profiles": packets,
                        }
                    )
                    if tick < 15:
                        for _ in range(round(0.2 / world.model.opt.timestep)):
                            mujoco.mj_step(world.model, world.data)
                result = score(samples, budget)
                report["cases"].append(
                    {"seed": seed, "initial_xy": xy, **result, "samples": samples}
                )
                save()
                print(
                    seed,
                    xy,
                    result["matched_raw_exceeded"],
                    result["filtered_exceeded"],
                    {k: v["status"] for k, v in result["policies"].items()},
                    flush=True,
                )
        report["summary"] = {
            "matched_views": sum(c["matched_views"] for c in report["cases"]),
            "matched_raw_exceeded": sum(c["matched_raw_exceeded"] for c in report["cases"]),
            "filtered_exceeded": sum(c["filtered_exceeded"] for c in report["cases"]),
            "policies": {
                key: {
                    "confirmations": sum(
                        c["policies"][key]["status"] == "confirmed_in_tray" for c in report["cases"]
                    ),
                    "false_confirmations": sum(
                        c["false_confirmations"][key] for c in report["cases"]
                    ),
                }
                for key in ("raw", "temporal")
            },
        }
        report["controls"] = {}
        for fault in ("periodic_motion", "shared_bias"):
            samples = []
            for tick in range(16):
                point = (
                    (0.63 if tick % 3 == 0 else 0.64, 0, 0.04)
                    if fault == "periodic_motion"
                    else (0.684, 0, 0.04)
                )
                truth = point if fault == "periodic_motion" else (0.692, 0, 0.04)
                samples.append(
                    {
                        "time": tick * 0.2,
                        "truth_xyz": truth,
                        "profiles": {"coverage": {"primary": point, "secondary": point}},
                    }
                )
            report["controls"][fault] = score(samples, budget)
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
