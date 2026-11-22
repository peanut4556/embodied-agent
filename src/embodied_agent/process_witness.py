"""Optional single-owner witness process; timeout closes it, never retries a request."""

import math
import multiprocessing

from .models import Action, Observation


class WitnessTimeout(TimeoutError):
    pass


def _worker(connection, factory):
    try:
        reader = factory()
        connection.send(("ready", None))
        for _ in range(128):
            action = connection.recv()
            value = reader(action)
            if not isinstance(value, Observation):
                raise TypeError("witness must return Observation")
            connection.send(("observation", value))
    except EOFError:
        pass
    except Exception as exc:  # noqa: BLE001 - serialize worker failures across the IPC boundary
        try:
            connection.send(("error", type(exc).__name__))
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


class ProcessWitness:
    """Spawn-picklable factory; caller must close explicitly or use a context manager.

    Only response readiness is timed. Local IPC/OS termination are not hard realtime.
    This does not prove hardware release or survive arbitrary owner-process death.
    """

    def __init__(self, factory, *, timeout_s=0.05, startup_timeout_s=5):
        for value in (timeout_s, startup_timeout_s):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= 5
            ):
                raise ValueError("witness timeout must be finite seconds in (0, 5]")
        self.timeout_s, self.calls, self.closed = timeout_s, 0, False
        self.exitcode = None
        ctx = multiprocessing.get_context("spawn")
        self.connection, child = ctx.Pipe()
        self.process = ctx.Process(target=_worker, args=(child, factory), daemon=True)
        try:
            self.process.start()
            child.close()
            kind, _ = self._receive(startup_timeout_s)
            if kind != "ready":
                raise RuntimeError("witness initialization failed")
        except BaseException:
            child.close()
            self.close()
            raise

    def _receive(self, timeout):
        if not self.connection.poll(timeout):
            raise WitnessTimeout("witness response timeout")
        return self.connection.recv()

    def __call__(self, action):
        if self.closed:
            raise RuntimeError("witness closed; explicit replacement required")
        if not isinstance(action, Action):
            raise TypeError("Action required")
        if self.calls >= 128:
            self.close()
            raise RuntimeError("witness request limit exhausted")
        self.calls += 1
        try:
            self.connection.send(action)
            kind, value = self._receive(self.timeout_s)
            if kind != "observation" or not isinstance(value, Observation):
                raise RuntimeError("witness read failed")
            return value
        except BaseException:
            self.close()
            raise

    def close(self):
        self.closed = True
        self.connection.close()
        if self.exitcode is not None:
            return True
        if self.process.pid is None:
            return True
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(0.1)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(0.1)
        if self.process.is_alive():
            return False
        self.exitcode = self.process.exitcode
        self.process.close()
        return True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
