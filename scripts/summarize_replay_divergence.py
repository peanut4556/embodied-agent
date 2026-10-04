"""Produce a compact checked report from the immutable rollout audit."""

import json
from pathlib import Path

import numpy as np

from embodied_agent.memory_policy import MemoryPolicy, digest
from embodied_agent.trajectory_divergence import action_decomposition


def summarize(root):
    root = Path(root)
    report = json.loads((root / "audit.json").read_text())
    if report["status"] != "complete":
        raise ValueError("incomplete audit")
    traces = {}
    for arm, expected in report["trace_sha256"].items():
        path = root / f"{arm}-trace.json"
        if digest(path) != expected:
            raise ValueError("trace changed")
        traces[arm] = json.loads(path.read_text())
    policy = MemoryPolicy("outputs/models/state-distill-distilled/epoch-300")
    if policy.metadata["weights_sha256"] != report["results"]["parent"]["weights_sha256"]:
        raise ValueError("parent changed")
    for arm, item in report["results"].items():
        trace = traces[arm]["trace"]
        raw = np.asarray(traces[arm]["raw_actions"])
        commanded = np.asarray([r["command"] for r in trace[: len(raw)]])
        bounded = np.clip(raw, policy.bounds[:, 0], policy.bounds[:, 1])
        limited = np.any(np.abs(commanded - bounded) > np.array([1e-5] * 3 + [1e-7] * 2), axis=1)
        item["rate_limited_ticks_before_parent_contact"] = np.flatnonzero(limited[:85]).tolist()
        item["contact_keyframes"] = [r for r in trace[75:90]]
    for arm, comp in report["comparisons"].items():
        n = comp["aligned_frames"]
        parent = np.asarray(traces["parent"]["raw_actions"][:n])
        same = parent + np.asarray(comp["decomposition_physical_actions"]["weight_residual"])
        own = np.asarray(traces[arm]["raw_actions"][:n])
        terms = action_decomposition(
            *(np.clip(v, policy.bounds[:, 0], policy.bounds[:, 1]) for v in (parent, same, own))
        )
        comp["bounded_action_windows"] = {}
        for name, window in comp["windows"].items():
            start, end = window["start_tick"], window["end_tick_exclusive"]
            comp["bounded_action_windows"][name] = {
                k: {
                    "arm_rmse_rad": float(np.sqrt(np.mean(v[start:end, :3] ** 2))),
                    "finger_rmse_m": float(np.sqrt(np.mean(v[start:end, 3:] ** 2))),
                }
                for k, v in terms.items()
            }
        comp["initial_command_delta"] = (
            np.asarray(traces[arm]["trace"][0]["command"]) - traces["parent"]["trace"][0]["command"]
        ).tolist()
        comp["selected_errors"] = {
            str(t): {k: v[t] for k, v in comp["series"].items()}
            for t in [0, 1, 2, 25, 50, 75, 77, 78, 79, 84, 85, 87]
        }
        del comp["series"], comp["decomposition_physical_actions"]
    report["audit_sha256"] = digest(root / "audit.json")
    report["regression_tests_passed"] = 7
    report["limitations"] = [
        "one previously successful training position; no validation or reserved tests",
        "time aligned comparisons are not matched physical states after divergence",
        "weight/history decomposition is algebraic, not causal attribution; history includes GRU hidden state",
        "bounded outputs precede executor rate limits; raw-output windows reported separately",
        "contact sampled at 25 Hz; sub-frame contact changes are not resolved",
        "no retraining or deployed policy replacement",
    ]
    return report


if __name__ == "__main__":
    report = summarize("outputs/evaluations/replay-divergence-v1")
    Path("docs/evaluations/replay-divergence-v1.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                a: {
                    "initial_command_delta": c["initial_command_delta"],
                    "bounded_contact_window": c["bounded_action_windows"]["first_contact"],
                }
                for a, c in report["comparisons"].items()
            },
            indent=2,
        )
    )
