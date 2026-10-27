"""Deterministic timing counterexample; not a camera or physics experiment."""

import argparse
import json
from pathlib import Path

from embodied_agent.exposure_confirmation import ExposureClockMap, ExposureConfirmation
from embodied_agent.memory_policy import digest
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.release_confirmation import ReleaseConfirmation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases = []
    for moving in (False, True):
        delivery_based = ReleaseConfirmation(0)
        exposure_based = ExposureConfirmation(0, ExposureClockMap(100, 0))
        rows = []
        for i in range(7):
            delivery = i * 0.2
            exposure = i * 0.16 if moving else max(0, delivery - 0.1)
            point = (0.62 + 0.017 * delivery, 0, 0.04) if moving else (0.64, 0, 0.04)
            delivery_based.observe(delivery, point)
            exposure_based.step(delivery, 100 + delivery, StampedPosition(point, 100 + exposure))
            rows.append(
                {
                    "delivery": delivery,
                    "exposure": exposure,
                    "position": point,
                    "delivery_status": delivery_based.status,
                    "exposure_status": exposure_based.status,
                }
            )
        exposure_based.step(3, 103, None)
        cases.append(
            {
                "name": "moving" if moving else "stationary",
                "observations": rows,
                "delivery_result": delivery_based.result(),
                "exposure_result": exposure_based.result(),
            }
        )
    result = {
        "scope": "Synthetic timestamped positions only; no physical experiment.",
        "source_sha256": {
            str(p): digest(p)
            for p in [
                Path(__file__),
                Path("src/embodied_agent/exposure_confirmation.py"),
                Path("src/embodied_agent/release_confirmation.py"),
                Path("src/embodied_agent/observation_capture.py"),
            ]
        },
        "cases": cases,
    }
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    assert cases[0]["exposure_result"]["status"] == "confirmed_in_tray"
    assert cases[1]["delivery_result"]["status"] == "confirmed_in_tray"
    assert cases[1]["exposure_result"]["status"] == "unconfirmed"
    print("Stationary control confirmed; variable-delay motion counterexample rejected.")


if __name__ == "__main__":
    main()
