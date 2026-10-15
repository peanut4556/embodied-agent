"""Revalidate an external bundle and evaluate strictly matched raw visual data offline."""

import argparse
import hashlib
import json
from pathlib import Path

from embodied_agent.external_joint_report import build_joint_report
from embodied_agent.external_observation_contract import ContractError, strict_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("observations", "policy", "visual", "evidence-root", "report"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()
    sources = {name: getattr(args, name) for name in ("observations", "policy", "visual")}
    destination = args.report.resolve()
    if destination in [p.resolve() for p in sources.values()] or destination.is_relative_to(
        args.evidence_root.resolve()
    ):
        parser.error("report must not overwrite source files or evidence")
    hashes = {}
    try:
        documents = {}
        for name, path in sources.items():
            with path.open("rb") as stream:
                raw = stream.read(20_000_001)
            if len(raw) > 20_000_000:
                raise ContractError(f"{name}: 20 MB limit exceeded")
            hashes[name] = hashlib.sha256(raw).hexdigest()
            documents[name] = strict_json(raw.decode("utf-8"))
        report = build_joint_report(
            documents["observations"],
            documents["policy"],
            args.evidence_root,
            documents["visual"],
            args.as_of,
        )
    except (ContractError, OSError, UnicodeError) as exc:
        report = {
            "status": "rejected",
            "results": [],
            "errors": [{"reason": str(exc)}],
            "calibration_certified": False,
            "correction_applied": False,
            "evidence_contents_authenticated": False,
        }
    report["source_sha256"] = hashes
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__),
        *[
            root / "src/embodied_agent" / f"{name}.py"
            for name in (
                "external_joint_report",
                "external_observation_contract",
                "external_reference",
                "fit_reliability",
                "exposure_uncertainty",
                "reference_fit_candidate",
                "reference_fit",
                "reference_statistics",
                "repeated_exposure",
                "multiplane_reference",
            )
        ],
    ]
    report["implementation_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["status"], report.get("summary", {}))
    return 0 if report["status"] == "evaluated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
