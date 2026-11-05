"""Qualify frozen profiles on prior evidence, then evaluate new rendered scenes."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from embodied_agent.profile_admission import QualifiedViewConfirmation, admit_profiles
from embodied_agent.rigid_reference import RigidReferenceConfirmation
from embodied_agent.uncertainty_view import UncertaintyViewConfirmation
from scripts.evaluate_rendered_precision import RESOLUTIONS, acquire, inside, place, sources
from scripts.evaluate_rigid_reference import CHECKPOINT, REFERENCES, make_world

SCENES = [
    (0.594, -0.007, 0),
    (0.598, 0.009, 0.15),
    (0.612, -0.028, -0.15),
    (0.650, 0.027, 0.15),
    (0.684, -0.009, 0),
    (0.688, 0.008, -0.15),
    (0.640, -0.037, 0.15),
    (0.643, 0.037, -0.15),
    (0.588, -0.006, 0),
    (0.692, 0.006, 0),
    (0.637, -0.043, 0),
    (0.646, 0.043, 0),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence_summary = json.loads(Path("docs/evaluations/rendered-precision-v3.json").read_text())
    evidence_path = Path(evidence_summary["full_report"]["path"])
    raw = evidence_path.read_bytes()
    evidence_sha = hashlib.sha256(raw).hexdigest()
    if evidence_sha != evidence_summary["full_report"]["sha256"]:
        raise ValueError("prior evidence file hash mismatch")
    previous = json.loads(raw)
    if previous["status"] != "complete":
        raise ValueError("completed prior evidence required")
    budgets = {key: row["position_error_m"] for key, row in previous["fitted_profiles"].items()}
    calibration_ids = [json.dumps(row["initial_xy"]) for row in previous["calibration"]]
    validation = [
        {"id": json.dumps(row["initial_xy"]), "samples": row["samples"]}
        for row in previous["cases"]
    ]
    admission = admit_profiles(budgets, calibration_ids, validation)
    paths = sources() + [
        Path("scripts/evaluate_profile_admission.py"),
        Path("src/embodied_agent/profile_admission.py"),
    ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "prior_evidence_sha256": evidence_sha,
        "source_sha256": hashes,
        "scenes": SCENES,
        "admission": admission.result(),
        "cases": [],
    }

    def save():
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    save()
    print("eligible profiles", dict(admission.profiles), flush=True)
    world, renderers = None, {}
    new_cases = []
    try:
        world = make_world("nominal")
        for key, (h, w) in RESOLUTIONS.items():
            renderers[key] = mujoco.Renderer(world.model, height=h, width=w)
        correction = RigidReferenceConfirmation(0, REFERENCES, CHECKPOINT)
        for index, (x, y, yaw) in enumerate(SCENES):
            place(world, (x, y))
            world.data.qpos[8:12] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            world.data.qvel[:] = 0
            mujoco.mj_forward(world.model, world.data)
            for _ in range(500):
                mujoco.mj_step(world.model, world.data)
            start = float(world.data.time)
            checks = {
                "unqualified": UncertaintyViewConfirmation(start, budgets, switch_on_missing=True),
                "qualified": QualifiedViewConfirmation(start, admission),
            }
            first_truth = dict.fromkeys(checks)
            requests = {key: [] for key in checks}
            samples = []
            for tick in range(16):
                timestamp = start + 0.2 * tick
                before = world.data.qpos.copy()
                packets = {
                    key: acquire(world, renderer, correction) for key, renderer in renderers.items()
                }
                if not np.array_equal(before, world.data.qpos):
                    raise RuntimeError("acquisition changed physical state")
                truth = world.data.body("red_block").xpos.tolist()
                for key, check in checks.items():

                    def locate(profile, key=key, packets=packets, requests=requests):
                        requests[key].append(profile)
                        return packets[profile]["primary"], packets[profile]["secondary"]

                    check.step(timestamp, locate)
                    if first_truth[key] is None and check.status == "confirmed_in_tray":
                        first_truth[key] = inside(truth)
                samples.append({"time": timestamp, "truth_xyz": truth, "profiles": packets})
                if tick < 15:
                    for _ in range(round(0.2 / world.model.opt.timestep)):
                        mujoco.mj_step(world.model, world.data)
            row = {
                "scene": (x, y, yaw),
                "policies": {key: check.result() for key, check in checks.items()},
                "false_confirmations": {key: value is False for key, value in first_truth.items()},
                "requested_profiles": requests,
                "samples": samples,
            }
            report["cases"].append(row)
            new_cases.append({"id": f"new_scene_{index}", "samples": samples})
            save()
            print((x, y, yaw), {key: check.status for key, check in checks.items()}, flush=True)
        report["summary"] = {
            key: {
                "confirmations": sum(
                    c["policies"][key]["status"] == "confirmed_in_tray" for c in report["cases"]
                ),
                "false_confirmations": sum(c["false_confirmations"][key] for c in report["cases"]),
                "rejected_profile_requests": sum(
                    p not in dict(admission.profiles)
                    for c in report["cases"]
                    for p in c["requested_profiles"][key]
                ),
            }
            for key in checks
        }
        # This update is for future runs only, never for the just-scored policies.
        report["new_scene_audit"] = admit_profiles(budgets, calibration_ids, new_cases).result()
        report["future_admission"] = admit_profiles(
            budgets, calibration_ids, validation + new_cases
        ).result()
        if hashes != {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}:
            raise RuntimeError("source changed during evaluation")
        if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != evidence_sha:
            raise RuntimeError("prior evidence changed during evaluation")
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        save()
        if world is not None:
            world.camera_renderer = None
            for renderer in renderers.values():
                renderer.close()
            world.close()
    print(
        json.dumps(
            {key: report[key] for key in ("summary", "new_scene_audit", "future_admission")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
