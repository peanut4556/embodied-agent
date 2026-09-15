"""Choose a checkpoint on development physics before opening reserved test cases."""

import json
from pathlib import Path

import numpy as np

from .imitation import ContextPolicy
from .memory_policy import MemoryPolicy, digest
from .physics import PhysicsWorld
from .reactive_evaluation import run_case
from .temporal_data import read_json


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def verify_run(root):
    root = Path(root)
    run, config = read_json(root / "run.json"), read_json(root / "experiment.json")
    if run["status"] != "trained" or digest(root / "experiment.json") != run["experiment_sha256"]:
        raise ValueError("requires a complete run with unchanged experiment")
    if [c["epoch"] for c in run["candidates"]] != config["candidate_epochs"]:
        raise ValueError("incomplete candidate set")
    for candidate in run["candidates"]:
        if candidate["directory"] != f"epoch-{candidate['epoch']}":
            raise ValueError("invalid candidate path")
        policy = MemoryPolicy(root / candidate["directory"])
        if (
            policy.metadata["weights_sha256"] != candidate["weights_sha256"]
            or policy.metadata["experiment_sha256"] != run["experiment_sha256"]
            or policy.metadata["source_sha256"] != run["source_sha256"]
        ):
            raise ValueError("candidate provenance differs from run")
    return run, config


def verify_physics(policy):
    scene = Path(__file__).parent / "assets/tabletop.xml"
    if digest(scene) != policy.metadata["model_sha256"]:
        raise ValueError("physics model differs from checkpoint")
    world = PhysicsWorld()
    try:
        if not np.array_equal(policy.bounds, world.model.actuator_ctrlrange):
            raise ValueError("actuator bounds differ from checkpoint")
    finally:
        world.close()


def select_candidate(candidates):
    if not candidates:
        raise ValueError("no physical selection results")
    return min(candidates, key=lambda c: (-c["task_success"], c["validation_mse"], c["epoch"]))


def save_preview(output, name, frames, fps):
    frames[0].save(
        output / f"{name}.gif",
        save_all=True,
        append_images=frames[1:],
        duration=round(5000 / fps),
        loop=0,
    )


def select(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite development selection: {output}")
    run, config = verify_run(root)
    output.mkdir(parents=True)
    report = {
        "status": "running",
        "experiment_sha256": run["experiment_sha256"],
        "source_sha256": run["source_sha256"],
        "test_cases_executed": False,
        "selection_rule": config["selection"],
        "candidates": [],
    }
    write(output / "selection.json", report)
    try:
        for candidate in run["candidates"]:
            policy = MemoryPolicy(root / candidate["directory"])
            verify_physics(policy)
            results = []
            for case in config["validation_cases"]:
                result, frames = run_case(policy, case, "memory", config["max_seconds"])
                results.append(result)
                save_preview(
                    output,
                    f"{candidate['directory']}-{case['name']}",
                    frames,
                    policy.metadata["fps"],
                )
                print(json.dumps({"candidate": candidate["directory"], **result}), flush=True)
            entry = {
                **candidate,
                "task_success": sum(r["success"] for r in results),
                "task_count": len(results),
                "results": results,
            }
            report["candidates"].append(entry)
            write(output / "selection.json", report)
        chosen = select_candidate(report["candidates"])
        report.update(
            status="selected",
            selected=chosen["directory"],
            selected_weights_sha256=chosen["weights_sha256"],
            selected_task_success=chosen["task_success"],
            eligible_for_reserved_test=chosen["task_success"] > 0,
        )
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        write(output / "selection.json", report)
    return report


def selected_policy(root, selection_path):
    run, config = verify_run(root)
    selection = read_json(selection_path)
    if (
        selection["status"] != "selected"
        or selection["test_cases_executed"] is not False
        or selection["experiment_sha256"] != run["experiment_sha256"]
        or selection["source_sha256"] != run["source_sha256"]
    ):
        raise ValueError("complete development-only selection required")
    if [c["directory"] for c in selection["candidates"]] != [
        c["directory"] for c in run["candidates"]
    ]:
        raise ValueError("selection did not evaluate every candidate")
    for candidate, original in zip(selection["candidates"], run["candidates"], strict=True):
        if any(candidate[k] != original[k] for k in original):
            raise ValueError("candidate selection metadata differs from training")
        if [r["case"] for r in candidate["results"]] != [
            c["name"] for c in config["validation_cases"]
        ]:
            raise ValueError("selection cases differ from frozen development cases")
        if candidate["task_success"] != sum(r["success"] for r in candidate["results"]):
            raise ValueError("selection score disagrees with physical outcomes")
    chosen = select_candidate(selection["candidates"])
    if (
        selection["selected"] != chosen["directory"]
        or selection["selected_weights_sha256"] != chosen["weights_sha256"]
    ):
        raise ValueError("selection does not follow the frozen criterion")
    if chosen["task_success"] < 1:
        raise ValueError("at least one development success required before reserved test")
    return MemoryPolicy(Path(root) / selection["selected"]), selection, config


def evaluate(root, selection_path, baseline_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite test report: {output}")
    policy, selection, config = selected_policy(root, selection_path)
    baseline = ContextPolicy(baseline_path)
    for p in (policy, baseline):
        verify_physics(p)
    if policy.metadata["fps"] != baseline.metadata["fps"]:
        raise ValueError("controller clocks differ")
    output.mkdir(parents=True)
    report = {
        "status": "running",
        "selected": selection["selected"],
        "selection_sha256": digest(selection_path),
        "weights_sha256": policy.metadata["weights_sha256"],
        "experiment_sha256": policy.metadata["experiment_sha256"],
        "baseline_sha256": digest(Path(baseline_path) / "policy.npz"),
        "results": [],
        "limitations": "fixed red cube/camera; small interpolation test; evaluator truth detects goal; no learned completion",
    }
    write(output / "evaluation.json", report)
    try:
        for case in config["test"]:
            for mode in ("feedback", "memory"):
                result, frames = run_case(
                    policy if mode == "memory" else baseline, case, mode, config["max_seconds"]
                )
                report["results"].append(result)
                save_preview(output, f"{case['name']}-{mode}", frames, policy.metadata["fps"])
                print(json.dumps(result), flush=True)
                write(output / "evaluation.json", report)
        report["scores"] = {}
        for mode in ("feedback", "memory"):
            rows = [r for r in report["results"] if r["controller"] == mode]
            tasks = [r for r in rows if not r["requested_stop"]]
            stops = [r for r in rows if r["requested_stop"]]
            slips = [
                r
                for r in tasks
                if r["disturbance"]
                and r["disturbance"]["holding_before_force"]
                and r["contact_loss_after_force"]
            ]
            report["scores"][mode] = {
                "task_success": sum(r["success"] for r in tasks),
                "verified_pick_place": sum(r["quality"]["verified_pick_place"] for r in tasks),
                "tasks": len(tasks),
                "timeouts": sum(r["reason"] == "execution timeout" for r in tasks),
                "valid_slip_trials": len(slips),
                "recoveries": sum(r["recovery_success"] for r in slips),
                "user_stop_passed": sum(r["user_stop_passed"] for r in stops),
                "stop_cases": len(stops),
            }
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        write(output / "evaluation.json", report)
    return report
