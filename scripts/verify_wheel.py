"""Install one trusted local wheel in a fresh environment and smoke-test it."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import venv
from pathlib import Path


def verify(wheel, output):
    wheel = Path(wheel).resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise ValueError("a local wheel file is required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "passed": False,
        "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "steps": [],
    }
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"

    def run(name, args, cwd):
        child = subprocess.run(
            [str(a) for a in args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        report["steps"].append(
            {
                "name": name,
                "exit_code": child.returncode,
                "stdout": child.stdout,
                "stderr": child.stderr,
            }
        )
        if child.returncode:
            raise RuntimeError(f"{name} failed")
        return child.stdout

    try:
        with tempfile.TemporaryDirectory(prefix="embodied-wheel-") as directory:
            root = Path(directory)
            venv.EnvBuilder(with_pip=True).create(root / "venv")
            binary = root / "venv" / ("Scripts" if os.name == "nt" else "bin")
            python = binary / "python"
            run(
                "install",
                [python, "-I", "-m", "pip", "install", "--no-index", "--no-deps", wheel],
                root,
            )
            run(
                "package-and-asset",
                [
                    python,
                    "-I",
                    "-c",
                    """import sys, pathlib
import embodied_agent
from importlib.resources import files
assert pathlib.Path(embodied_agent.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve())
asset = files('embodied_agent').joinpath('assets/tabletop.xml')
assert '<mujoco' in asset.read_text()
print('Installed package and tabletop asset verified')
""",
                ],
                root,
            )
            for entry in ("embodied-agent", "embodied-evidence-audit", "embodied-checkpoint"):
                run(entry + "-help", [binary / entry, "--help"], root)
            result = json.loads(
                run("mock-task", [binary / "embodied-agent", "red block box", "--json"], root)
            )
            if result.get("success") is not True or result.get("completed_steps") != 4:
                raise AssertionError("installed mock task did not complete four steps")
            report["passed"] = True
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        (output / "wheel-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps({"passed": verify(args.wheel, args.output)["passed"]}))
