"""Frozen parent/control/replay audit at the single known successful training position."""

import json
from pathlib import Path

import numpy as np
import torch

from embodied_agent.memory_evaluation import save_preview, verify_physics
from embodied_agent.memory_policy import MemoryPolicy, digest
from embodied_agent.reactive_evaluation import run_case
from embodied_agent.trajectory_divergence import action_decomposition, compare, events

ROOTS = {
    "parent": "outputs/models/state-distill-distilled/epoch-300",
    "control": "outputs/models/success-replay-control/epoch-300",
    "replay": "outputs/models/success-replay-replay/epoch-300",
}


class Recorder:
    def __init__(self, policy):
        self.policy = policy
        self.metadata, self.bounds = policy.metadata, policy.bounds
        self.observations, self.actions = [], []

    def session(self):
        owner, session = self, self.policy.session()

        class Session:
            metadata, bounds = owner.metadata, owner.bounds

            def predict(self, observations):
                action = session.predict(observations)
                owner.observations.extend(observations.tolist())
                owner.actions.extend(action.tolist())
                return action

        return Session()


def outputs(policy, observations):
    x = ((np.asarray(observations, dtype=np.float32) - policy.mean) / policy.scale).astype(
        np.float32
    )
    with torch.no_grad():
        y = policy.model(torch.from_numpy(x)[None])[0][0].numpy()
    return y * np.diff(policy.bounds).ravel() + policy.bounds[:, 0]


def main():
    output = Path("outputs/evaluations/replay-divergence-v1")
    if output.exists():
        raise FileExistsError(output)
    policies = {arm: MemoryPolicy(root) for arm, root in ROOTS.items()}
    for arm, policy in policies.items():
        verify_physics(policy)
        if (
            arm != "parent"
            and policy.metadata["pretrained_weights_sha256"]
            != policies["parent"].metadata["weights_sha256"]
        ):
            raise ValueError("parent mismatch")
        if (
            not np.array_equal(policy.bounds, policies["parent"].bounds)
            or policy.metadata["fps"] != 25
        ):
            raise ValueError("incompatible action space or clock")
    output.mkdir(parents=True)
    report = {
        "status": "running",
        "position": 0.302,
        "group": "train",
        "test_executed": False,
        "training_performed": False,
        "script_sha256": digest(__file__),
        "results": {},
        "comparisons": {},
    }
    traces, observations, raw = {}, {}, {}
    try:
        for arm, policy in policies.items():
            recorder, trace = Recorder(policy), []
            result, frames = run_case(
                recorder, {"name": "training-completed-302", "x": 0.302}, "memory", 24, trace
            )
            traces[arm], observations[arm] = trace, recorder.observations
            raw[arm] = outputs(policy, recorder.observations)
            prediction_error = np.max(
                np.abs(
                    np.clip(raw[arm], policy.bounds[:, 0], policy.bounds[:, 1]) - recorder.actions
                ),
                axis=0,
            )
            # CPU batched GRU and incremental GRU may accumulate small rounding differences.
            if np.any(prediction_error > np.array([1e-5] * 3 + [1e-7] * 2)):
                raise ValueError(
                    f"batch predictions differ from causal rollout: {prediction_error}"
                )
            for key in ("simulation_qpos", "simulation_qvel", "previous_command"):
                if not np.array_equal(trace[0][key], traces["parent"][0][key]):
                    raise ValueError("initial physics states differ")
            report["results"][arm] = {
                "weights_sha256": policy.metadata["weights_sha256"],
                "result": result,
                "events": events(trace, 25),
                "policy_frames": len(raw[arm]),
                "batch_stream_max_error": prediction_error.tolist(),
            }
            (output / f"{arm}-trace.json").write_text(
                json.dumps(
                    {
                        "trace": trace,
                        "observations": recorder.observations,
                        "raw_actions": raw[arm].tolist(),
                    },
                    indent=2,
                )
                + "\n"
            )
            save_preview(output, arm, frames, 25)
            print(json.dumps({"arm": arm, **report["results"][arm]}), flush=True)
        for arm in ("control", "replay"):
            n = min(len(raw["parent"]), len(raw[arm]))
            comparison = compare(traces["parent"][:n], traces[arm][:n])
            same = outputs(policies[arm], observations["parent"][:n])
            terms = action_decomposition(raw["parent"][:n], same, raw[arm][:n])
            comparison["decomposition_physical_actions"] = {k: v.tolist() for k, v in terms.items()}
            # Each term has different units for arms/fingers; never combine them.
            comparison["windows"] = {}
            contact = report["results"]["parent"]["events"]["first_finger_contact_tick"]
            for name, start, end in [
                ("approach", 0, contact),
                ("first_contact", max(0, contact - 10), min(n, contact + 11)),
                ("shared_active", 0, n),
            ]:
                comparison["windows"][name] = {
                    "start_tick": start,
                    "end_tick_exclusive": end,
                    **{
                        key: {
                            "arm_rmse_rad": float(np.sqrt(np.mean(value[start:end, :3] ** 2))),
                            "finger_rmse_m": float(np.sqrt(np.mean(value[start:end, 3:] ** 2))),
                        }
                        for key, value in terms.items()
                    },
                }
            report["comparisons"][arm] = comparison
        report["trace_sha256"] = {a: digest(output / f"{a}-trace.json") for a in ROOTS}
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
