"""Build training-only verified-completion masks from checked autonomous traces."""

import json
from pathlib import Path

from embodied_agent.correction_data import payload_digest
from embodied_agent.future_targets import completion_flags
from embodied_agent.memory_policy import digest


def main():
    root = Path("outputs/evaluations/multistep-targets-v2")
    source = Path("outputs/datasets/corrections-onpolicy-v1")
    output = Path("outputs/datasets/completion-targets-v1.json")
    if output.exists():
        raise FileExistsError(output)
    audit = json.loads((root / "audit.json").read_text())
    manifest = json.loads((source / "recording.json").read_text())
    if audit["status"] != "complete" or payload_digest(source) != audit["source_sha256"]:
        raise ValueError("complete unchanged branch audit required")
    # Read only training labels; validation targets never enter this pack.
    pack = json.loads((root / "train-targets.json").read_text())
    if pack["group"] != "train" or pack["source_sha256"] != audit["source_sha256"]:
        raise ValueError("training target source mismatch")
    traces = {}
    trace_hashes = {}
    for ep in manifest["split"]["train"]:
        x = manifest["episodes"][ep]["scenario"]["x"]
        path = root / f"baseline-{round(x * 1000)}.json"
        expected = next(b["trace_sha256"] for b in audit["baselines"] if b["x"] == x)
        if digest(path) != expected:
            raise ValueError("baseline trace changed")
        traces[ep] = completion_flags(json.loads(path.read_text()))
        trace_hashes[str(ep)] = expected
    for row in pack["targets"]:
        ep = row["episode"]
        if row["group"] != "train" or ep not in manifest["split"]["train"]:
            raise ValueError("validation target cannot enter training pack")
        tick = manifest["episodes"][ep]["takeover"]["tick"]
        row["takeover_tick"] = tick
        row["autonomous_verified_complete"] = bool(traces[ep][tick + row["horizon_ticks"]])
    pack.update(
        weights_sha256=audit["weights_sha256"],
        horizons=audit["horizons_ticks"],
        input_targets_sha256=digest(root / "train-targets.json"),
        baseline_sha256=trace_hashes,
        generator_sha256=digest(__file__),
    )
    output.write_text(json.dumps(pack, indent=2) + "\n")
    print(
        "targets",
        len(pack["targets"]),
        "protected",
        sum(r["autonomous_verified_complete"] for r in pack["targets"]),
    )


if __name__ == "__main__":
    main()
