"""Run the explicit CI test manifest and persist structured unittest outcomes."""

import argparse
import json
import re
import time
import unittest
from pathlib import Path


def load_suite(manifest):
    names = Path(manifest).read_text().split()
    if (
        not names
        or len(names) != len(set(names))
        or any(re.fullmatch(r"test_[a-z0-9_]+", name) is None for name in names)
    ):
        raise ValueError("manifest requires unique test module names")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(names)
    if loader.errors:
        raise ValueError("test manifest contains modules that could not be loaded")
    if suite.countTestCases() == 0:
        raise ValueError("empty test suite")
    return suite


def run_suite(suite, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="config/ci-core-tests.txt")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    suite = load_suite(args.manifest)
    return 0 if run_suite(suite, args.output)["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
