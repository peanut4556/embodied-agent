"""Start a detached localhost console; logs and PID stay in ignored outputs/."""

import os
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

root = Path(__file__).resolve().parents[1]
url = "http://127.0.0.1:8765/"


def ready():
    try:
        with urlopen(url, timeout=1) as response:
            return b"Embodied Agent" in response.read()
    except (URLError, TimeoutError):
        return False


if ready():
    print(f"Console already running: {url}")
else:
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    with (output / "sim-web.log").open("ab") as log:
        process = subprocess.Popen(
            [str(root / ".venv/bin/python"), "-m", "embodied_agent.sim_app"],
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    (output / "sim-web.pid").write_text(str(process.pid))
    for _ in range(30):
        if ready():
            print(f"Console running: {url} (PID {process.pid})")
            break
        if process.poll() is not None:
            raise RuntimeError("Console failed; see outputs/sim-web.log")
        time.sleep(0.1)
    else:
        raise RuntimeError("Console startup timeout; see outputs/sim-web.log")
