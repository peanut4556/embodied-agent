"""Offline audit on recorded RGB-D observations; no new physics or policy training."""

import argparse
import json
from pathlib import Path

from embodied_agent.memory_policy import digest
from embodied_agent.release_review import ReleaseReview

INPUTS = [
    Path("docs/evaluations/release-late-window-v1.json"),
    Path("docs/evaluations/release-window-validation-v1.json"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for source in INPUTS:
        report = json.loads(source.read_text())
        if report["status"] != "complete" or report["observe_seconds"] != 6:
            raise ValueError("completed six-second records required")
        for case in report["cases"]:
            observations = case["observations"]
            review = ReleaseReview(observations[0]["time"])
            confirmed_at, confirmation_truth = None, None
            for observation in observations:
                review.step(observation["time"], lambda o=observation: o["xyz"])
                if review.status == "confirmed_in_tray" and confirmed_at is None:
                    confirmed_at = observation["time"] - observations[0]["time"]
                    confirmation_truth = observation["truth_inside"]
            assert review.initial.result() == case["confirmation_result"], "initial result changed"
            assert review.status != "pending", "incomplete review"
            expected = case["status"]
            if expected == "unconfirmed":
                expected = case["late_result"]["status"]
            assert review.status == expected, "unexpected review outcome"
            rows.append(
                {
                    "source": source.name,
                    "name": case["name"],
                    **review.result(),
                    "confirmation_elapsed_s": confirmed_at,
                    "truth_at_confirmation": confirmation_truth,
                    "recorded_observations": len(observations),
                }
            )
    result = {
        "status": "complete",
        "scope": "Offline replay of 42 recorded observation streams; not new physical trials.",
        "inputs_sha256": {str(p): digest(p) for p in INPUTS},
        "source_sha256": {
            str(p): digest(p)
            for p in (
                Path(__file__),
                Path("src/embodied_agent/release_review.py"),
                Path("src/embodied_agent/release_confirmation.py"),
            )
        },
        "cases": rows,
        "confirmed": sum(r["status"] == "confirmed_in_tray" for r in rows),
        "initial_confirmed": sum(
            r["initial_result"]["status"] == "confirmed_in_tray" for r in rows
        ),
        "rechecks": sum(r["recheck_result"] is not None for r in rows),
        "observations_used": sum(r["observation_count"] for r in rows),
        "observations_recorded": sum(r["recorded_observations"] for r in rows),
        "false_confirmations": sum(r["truth_at_confirmation"] is False for r in rows),
    }
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["false_confirmations"] == 0
    print({k: v for k, v in result.items() if k not in ("cases", "inputs_sha256", "source_sha256")})


if __name__ == "__main__":
    main()
