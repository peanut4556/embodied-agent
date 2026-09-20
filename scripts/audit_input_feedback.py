"""Factorial reference-feature interventions in isolated development rollouts."""

import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", "outputs/hf-cache")


from embodied_agent.memory_evaluation import verify_physics
from embodied_agent.memory_policy import MemoryPolicy, digest, sequences
from embodied_agent.temporal_data import source_digest
from scripts.audit_action_chain import ObservedSession, run, summarize

MODES = ("actual", "reference_vision", "reference_joints", "reference_both")


def intervene(observations, reference, mode):
    if mode not in MODES:
        raise ValueError("unknown input intervention")
    if observations.shape != (1, 12) or reference.shape != (12,):
        raise ValueError("one current feature frame required")
    result = observations.copy()
    if mode in ("reference_vision", "reference_both"):
        result[0, :6] = reference[:6]
    if mode in ("reference_joints", "reference_both"):
        result[0, 6:11] = reference[6:11]
    # Contact remains actual in all four conditions; no simulator state is replaced.
    return result


class InterventionSession(ObservedSession):
    def __init__(self, policy, reference, mode):
        super().__init__(policy)
        self.reference, self.mode, self.inputs = reference, mode, []

    def predict(self, observations):
        tick = len(self.inputs)
        used = intervene(observations, self.reference[tick], self.mode)
        self.inputs.append({"actual": observations[0].tolist(), "used": used[0].tolist()})
        return super().predict(used)


def main():
    output = Path("outputs/evaluations/input-feedback-v1")
    if output.exists():
        raise FileExistsError(output)
    config_path = Path("config/correction-early-experiment.json")
    config = json.loads(config_path.read_text())
    base = Path("outputs/datasets/reactive-development-v2")
    refs = sequences(base, config["split"], "validation")
    models = [
        "outputs/models/memory-correction-v1/epoch-600",
        "outputs/models/memory-correction-v2/epoch-600",
    ]
    output.mkdir(parents=True)
    report = {
        "status": "running",
        "source_sha256": source_digest(base),
        "experiment_sha256": digest(config_path),
        "script_sha256": digest(__file__),
        "action_chain_sha256": digest(Path(__file__).with_name("audit_action_chain.py")),
        "horizon_seconds": 2,
        "test_executed": False,
        "models_retrained": False,
        "modes": MODES,
        "contact_input": "actual in every condition",
        "results": [],
    }
    try:
        for model in models:
            policy = MemoryPolicy(model)
            verify_physics(policy)
            if policy.metadata["fps"] != 25:
                raise ValueError("this frozen diagnostic requires 25 Hz")
            for case, ref in zip(config["validation_cases"], refs, strict=True):
                for mode in MODES:
                    session = InterventionSession(policy, ref["x"], mode)
                    rows = run(policy, case, ref, "limited", session_factory=lambda _, current=session: current)
                    for row, inputs in zip(rows, session.inputs, strict=True):
                        row["inputs"] = inputs
                    # Summarize numeric action-chain fields before adding input metadata.
                    score = summarize(
                        [{k: v for k, v in row.items() if k != "inputs"} for row in rows], 25
                    )
                    name = Path(model).parent.name + "-" + case["name"] + "-" + mode + ".json"
                    (output / name).write_text(json.dumps(rows) + "\n")
                    result = {
                        "model": model,
                        "weights_sha256": policy.metadata["weights_sha256"],
                        "case": case,
                        "mode": mode,
                        "trace": name,
                        "trace_sha256": digest(output / name),
                        **score,
                    }
                    report["results"].append(result)
                    print(json.dumps(result), flush=True)
        if source_digest(base) != report["source_sha256"]:
            raise ValueError("reference source changed")
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
