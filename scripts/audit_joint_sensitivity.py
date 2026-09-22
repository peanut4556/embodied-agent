"""Single-frame joint measurement pulses: fixed-input memory response and physical rollout."""

import json
from pathlib import Path

import numpy as np

from embodied_agent.memory_evaluation import verify_physics
from embodied_agent.memory_policy import MemoryPolicy, digest, sequences
from embodied_agent.temporal_data import source_digest
from scripts.audit_action_chain import ObservedSession, run, summarize


def pulse(observations, joint, amount, bounds):
    if observations.shape != (1, 12) or joint not in range(5) or not np.isfinite(amount):
        raise ValueError("invalid joint pulse")
    used = observations.copy()
    used[0, 6 + joint] = np.clip(used[0, 6 + joint] + amount, *bounds[joint])
    return used, float(used[0, 6 + joint] - observations[0, 6 + joint])


def replay(policy, observations, tick=None, joint=0, amount=0.0):
    session = policy.session()
    outputs, applied = [], 0.0
    for i, row in enumerate(observations):
        used = row[None]
        if i == tick:
            used, applied = pulse(used, joint, amount, policy.bounds)
        outputs.append(session.predict(used)[0])
    return np.stack(outputs), applied


class PulseSession(ObservedSession):
    def __init__(self, policy, joint, amount, tick=12):
        super().__init__(policy)
        self.joint, self.amount, self.tick = joint, amount, tick
        self.inputs = []

    def predict(self, observations):
        used = observations
        applied = 0.0
        if len(self.inputs) == self.tick and self.amount:
            used, applied = pulse(observations, self.joint, self.amount, self.bounds)
        self.inputs.append(
            {"actual": observations[0].tolist(), "used": used[0].tolist(), "applied": applied}
        )
        return super().predict(used)


def main():
    output = Path("outputs/evaluations/joint-sensitivity-v1")
    if output.exists():
        raise FileExistsError(output)
    prior_root = Path("outputs/evaluations/input-feedback-v1")
    prior = json.loads((prior_root / "audit.json").read_text())
    if prior["status"] != "complete":
        raise ValueError("complete input audit required")
    config_path = Path("config/correction-early-experiment.json")
    config = json.loads(config_path.read_text())
    base = Path("outputs/datasets/reactive-development-v2")
    if source_digest(base) != prior["source_sha256"]:
        raise ValueError("reference source changed")
    refs = {e["episode"]: e for e in sequences(base, config["split"], "validation")}
    report = {
        "status": "running",
        "script_sha256": digest(__file__),
        "source_sha256": prior["source_sha256"],
        "input_audit_sha256": digest(prior_root / "audit.json"),
        "config_sha256": digest(config_path),
        "test_executed": False,
        "models_retrained": False,
        "design": {
            "pulse_ticks": [5, 12, 24],
            "following_fixed_frames": 10,
            "arm_pulses_rad": [0.005, 0.01],
            "finger_pulses_m": [0.0005, 0.001],
            "signs": [-1, 1],
            "physical_tick": 12,
            "physical_pulse_rad": 0.01,
            "selection": "among arm joints, largest mean peak normalized gain over all fixed-input trials; lower index breaks ties",
        },
        "offline": [],
        "joint_scores": {},
        "physical": [],
    }
    output.mkdir(parents=True)
    try:
        for model in sorted({r["model"] for r in prior["results"]}):
            policy = MemoryPolicy(model)
            verify_physics(policy)
            if policy.metadata["fps"] != 25:
                raise ValueError("requires frozen 25Hz checkpoints")
            ranges = policy.bounds[:, 1] - policy.bounds[:, 0]
            source_rows = [
                r for r in prior["results"] if r["model"] == model and r["mode"] == "actual"
            ]
            if [r["case"] for r in source_rows] != config["validation_cases"]:
                raise ValueError("development cases differ")
            for source in source_rows:
                if (
                    source["weights_sha256"] != policy.metadata["weights_sha256"]
                    or digest(prior_root / source["trace"]) != source["trace_sha256"]
                ):
                    raise ValueError("input trace provenance changed")
                rows = json.loads((prior_root / source["trace"]).read_text())
                x = np.array([r["inputs"]["actual"] for r in rows], dtype=np.float32)
                baseline, _ = replay(policy, x)
                if not np.allclose(
                    baseline, np.array([r["prediction"] for r in rows]), atol=1e-7, rtol=0
                ):
                    raise ValueError("baseline recurrent replay differs")
                for tick in report["design"]["pulse_ticks"]:
                    for joint in range(5):
                        for magnitude in [0.005, 0.01] if joint < 3 else [0.0005, 0.001]:
                            for sign in (-1, 1):
                                changed, applied = replay(
                                    policy, x[: tick + 11], tick, joint, sign * magnitude
                                )
                                delta = np.abs(changed[tick:] - baseline[tick : tick + 11])
                                response = np.max(delta / ranges, axis=1)
                                denominator = abs(applied) / ranges[joint]
                                report["offline"].append(
                                    {
                                        "model": model,
                                        "weights_sha256": policy.metadata["weights_sha256"],
                                        "case": source["case"]["name"],
                                        "tick": tick,
                                        "joint": joint,
                                        "requested": sign * magnitude,
                                        "applied": applied,
                                        "normalized_response_by_lag": response.tolist(),
                                        "gain_by_lag": (response / denominator).tolist()
                                        if denominator
                                        else None,
                                        "max_arm_action_delta_rad": float(delta[:, :3].max()),
                                        "max_finger_action_delta_m": float(delta[:, 3:].max()),
                                    }
                                )
            scores = {
                j: float(
                    np.mean(
                        [
                            max(r["gain_by_lag"])
                            for r in report["offline"]
                            if r["model"] == model
                            and r["joint"] == j
                            and r["gain_by_lag"] is not None
                        ]
                    )
                )
                for j in range(5)
            }
            selected = max(range(3), key=lambda j: (scores[j], -j))
            report["joint_scores"][model] = {
                "mean_peak_normalized_gain": scores,
                "selected_arm_joint": selected,
            }
            for source, ep in zip(source_rows, config["split"]["validation"], strict=True):
                for amount in (0.0, -0.01, 0.01):
                    session = PulseSession(policy, selected, amount)
                    rows = run(
                        policy,
                        source["case"],
                        refs[ep],
                        "limited",
                        session_factory=lambda _, current=session: current,
                    )
                    score = summarize(rows, 25)
                    for row, inputs in zip(rows, session.inputs, strict=True):
                        row["inputs"] = inputs
                    name = (
                        Path(model).parent.name
                        + "-"
                        + source["case"]["name"]
                        + "-"
                        + str(amount)
                        + ".json"
                    )
                    (output / name).write_text(json.dumps(rows) + "\n")
                    result = {
                        "model": model,
                        "case": source["case"],
                        "joint": selected,
                        "amount_rad": amount,
                        "trace": name,
                        "trace_sha256": digest(output / name),
                        **score,
                    }
                    report["physical"].append(result)
                    print(json.dumps(result), flush=True)
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
