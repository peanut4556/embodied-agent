"""Validate an explicit external-observation bundle; never invoke calibration or networking."""

import argparse
import hashlib
import json
from pathlib import Path

from embodied_agent.external_observation_contract import (
    ContractError,
    import_observations,
    strict_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument(
        "--as-of", required=True, help="explicit timezone-aware evaluation timestamp"
    )
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    destination = args.report.resolve()
    if destination in (args.input.resolve(), args.policy.resolve()) or destination.is_relative_to(
        args.evidence_root.resolve()
    ):
        parser.error("report must not overwrite inputs, policy or evidence bundle")
    hashes = {}
    try:
        documents = []
        for label, path in (("input", args.input), ("policy", args.policy)):
            with path.open("rb") as stream:
                raw = stream.read(10_000_001)
            if len(raw) > 10_000_000:
                raise ContractError(f"{label}: maximum size is 10 MB")
            hashes[label] = hashlib.sha256(raw).hexdigest()
            documents.append(strict_json(raw.decode("utf-8")))
        report = import_observations(*documents, args.evidence_root, args.as_of)
    except (ContractError, OSError, UnicodeError) as exc:
        report = {
            "status": "rejected",
            "normalized": [],
            "errors": [{"index": None, "reason": str(exc)}],
            "eligible_for_offline_check": False,
            "calibration_certified": False,
            "correction_applied": False,
            "evidence_contents_authenticated": False,
        }
    report["source_sha256"] = hashes
    report["implementation_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            Path(__file__),
            Path(__file__).resolve().parents[1]
            / "src/embodied_agent/external_observation_contract.py",
        )
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"{report['status']}: {len(report['normalized'])} normalized, {len(report['errors'])} errors"
    )
    return 0 if report["status"] == "validated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
