"""Real-process recovery checks with synthetic readers and explicit new tasks."""

import argparse
import functools
import hashlib
import json
import time
from pathlib import Path

from embodied_agent.capture_recovery import CaptureRecovery
from embodied_agent.observation_capture import StampedPosition
from embodied_agent.threaded_pair_capture import CapturedPair
from scripts.check_process_pair_capture import failure_factory, fast_factory, hung_factory
from scripts.evaluate_sampling_aliases import sources as parent_sources


class StaleReader:
    def __init__(self, token):
        self.token = token

    def __call__(self, req):
        p = StampedPosition((0.64, 0, 0.045), time.monotonic())
        return CapturedPair(self.token, p, p)


def stale_factory(token):
    return StaleReader(token)


def drive(manager, token):
    deadline = time.monotonic() + 4
    while manager.runtime.controller.status == "pending" and time.monotonic() < deadline:
        manager.step(token)
        time.sleep(0.0005)
    if manager.runtime.controller.status == "pending":
        manager.step(token, stop=True)


def run(mode):
    manager = CaptureRecovery(position_error_m=(0.001,) * 3, max_generations=2)
    try:
        old = manager.start(failure_factory if mode == "startup_to_fast" else hung_factory)
        if old is not None:
            drive(manager, old)
        old_token = manager.session
        old_backend = manager.backend
        factory = (
            functools.partial(stale_factory, old_token) if mode == "hang_to_stale" else fast_factory
        )
        token = manager.start(factory)
        assert token is not None and token != old_token
        assert old_backend.result()["workers_stopped"]
        assert old_backend.drain() == []
        assert manager.runtime.controller.requests == []
        assert manager.step(old_token, stop=True) is None
        assert manager.runtime.controller.status == "pending"
        drive(manager, token)
        if mode == "hang_to_stale":
            assert manager.runtime.controller.status == "unconfirmed"
            assert manager.runtime.controller.reason == "camera_session_changed"
        assert manager.close()
        return {"mode": mode, "result": manager.result()}
    finally:
        manager.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def sources():
        paths = (
            __file__,
            "scripts/check_process_pair_capture.py",
            "src/embodied_agent/capture_recovery.py",
            "src/embodied_agent/process_pair_capture.py",
            "src/embodied_agent/two_stage_runtime.py",
            "src/embodied_agent/threaded_pair_capture.py",
            "src/embodied_agent/two_stage_capture.py",
            "src/embodied_agent/fresh_witness.py",
        )
        return {
            **parent_sources(),
            **{p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        }

    before = sources()
    cases = [run(mode) for mode in ("hang_to_fast", "startup_to_fast", "hang_to_stale")]
    assert before == sources()
    path = args.output / "report.json"
    path.write_text(json.dumps({"sources": before, "cases": cases}, indent=2) + "\n")
    summary = {
        "kind": "explicit_process_recovery",
        "sources": before,
        "report_path": str(path),
        "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cases": [
            {
                "mode": c["mode"],
                "attempts": c["result"]["attempts"],
                "ignored_sessions": c["result"]["ignored_sessions"],
                "generations": [
                    {
                        "session_id": h["session_id"],
                        "generation": h["generation"],
                        "status": h["result"]["confirmation"]["status"]
                        if h["result"]
                        else "not_started",
                        "reason": h["result"]["confirmation"]["reason"]
                        if h["result"]
                        else "startup_failure",
                        "requests": len(h["result"]["confirmation"]["requests"])
                        if h["result"]
                        else 0,
                        "cleanup": h["cleanup"],
                    }
                    for h in c["result"]["history"]
                ],
            }
            for c in cases
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["cases"], indent=2))


if __name__ == "__main__":
    main()
