"""Run the explicit CI test manifest and persist structured unittest outcomes."""

import argparse
import json
import re
import time
import traceback
import unittest
from pathlib import Path


def load_suite(manifest):
    with Path(manifest).open() as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("manifest exceeds 64 KiB character limit")
    names = raw.split()
    if (
        not names
        or len(names) > 256
        or len(names) != len(set(names))
        or any(re.fullmatch(r"test_[a-z0-9_]+", name) is None for name in names)
    ):
        raise ValueError("manifest requires unique test module names")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in names:
        module_suite = loader.loadTestsFromName(name)
        if loader.errors:
            raise ValueError(f"module could not be loaded: {name}\n" + "\n".join(loader.errors))
        if module_suite.countTestCases() == 0:
            raise ValueError(f"module contains no test cases: {name}")
        suite.addTests(module_suite)
    return suite


def run_suite(suite, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    return _run_suite(suite, output)


def _run_suite(suite, output):
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "schema": 1,
        "successful": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "duration_s": time.monotonic() - started,
        "failures": [{"test": t.id(), "traceback": trace} for t, trace in result.failures],
        "errors": [{"test": t.id(), "traceback": trace} for t, trace in result.errors],
        "skipped": [{"test": t.id(), "reason": reason} for t, reason in result.skipped],
        "expected_failures": [
            {"test": t.id(), "traceback": trace} for t, trace in result.expectedFailures
        ],
        "unexpected_successes": [t.id() for t in result.unexpectedSuccesses],
    }
    (output / "tests.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "## Core regression results",
        "",
        f"Result: {'PASS' if report['successful'] else 'FAIL'}",
        "",
        (
            f"Tests: {result.testsRun}; failures: {len(result.failures)}; "
            f"errors: {len(result.errors)}; skipped: {len(result.skipped)}; "
            f"unexpected successes: {len(result.unexpectedSuccesses)}."
        ),
        "",
        "Detailed test identifiers and tracebacks are in the tests.json artifact.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return report


def run_manifest(manifest, output):
    """Reserve a fresh directory before loading tests; never overwrite old evidence."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    phase = "load"
    try:
        suite = load_suite(manifest)
        phase = "execute_or_report"
        return 0 if _run_suite(suite, output)["successful"] else 1
    except Exception as exc:  # noqa: BLE001 - runner boundary records all ordinary failures
        # Ordinary unittest failures already appear in tests.json. This records
        # runner-level exceptions; forced exits and KeyboardInterrupt propagate.
        report = {
            "schema": 1,
            "successful": False,
            "phase": phase,
            "cause_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
        (output / "runner-error.json").write_text(json.dumps(report, indent=2) + "\n")
        (output / "summary.md").write_text(
            "## Core regression results\n\nResult: RUNNER ERROR\n\n"
            "The suite did not complete normally. See runner-error.json and the job log.\n"
        )
        return 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="config/ci-core-tests.txt")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return run_manifest(args.manifest, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
