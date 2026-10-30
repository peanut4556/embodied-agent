"""Audit saved recovery reports without opening devices or changing inputs."""

import argparse

from embodied_agent.recovery_audit import audit_files
from embodied_agent.recovery_report import write_recovery_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_files(args.inputs)
    write_recovery_report(args.output, report)
    print(f"Accepted {report['accepted_count']}; rejected {report['rejected_count']}")
    return 1 if report["rejected_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
