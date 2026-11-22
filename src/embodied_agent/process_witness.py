"""Optional single-owner witness process; timeout closes it, never retries a request."""

import math
import multiprocessing
import time

from .models import Action, Observation


class WitnessFailure(RuntimeError):
    """Expected transport or worker failure with a bounded diagnostic code."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


class WitnessCancelled(Exception):
    pass


class WitnessTimeout(TimeoutError):
    pass


class WitnessDeadline(WitnessTimeout):
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
        self.cleanup_state = "open"
        self.terminate_attempts = 0
        self.kill_attempts = 0
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

    def _receive(self, timeout, stopped=None):
        deadline = time.monotonic() + timeout
        while True:
            if stopped is not None and stopped():
                raise WitnessCancelled("witness cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WitnessTimeout("witness response timeout")
            if self.connection.poll(min(remaining, 0.01)):
                if stopped is not None and stopped():
                    raise WitnessCancelled("witness cancelled")
                return self.connection.recv()

    def __call__(self, action, *, stopped=None, budget_s=None):
        if budget_s is not None and (
            isinstance(budget_s, bool)
            or not isinstance(budget_s, (int, float))
            or not math.isfinite(budget_s)
            or budget_s < 0
        ):
            raise ValueError("budget_s must be finite nonnegative seconds")
        limited = budget_s is not None and budget_s <= self.timeout_s
        timeout = min(self.timeout_s, budget_s) if budget_s is not None else self.timeout_s
        started = time.monotonic()
        if stopped is not None and not callable(stopped):
            raise TypeError("stopped must be callable")
        if self.closed:
            raise WitnessFailure("witness_closed")
        if not isinstance(action, Action):
            raise TypeError("Action required")
        if self.calls >= 128:
            self.close()
            raise WitnessFailure("witness_request_limit")
        try:
            if stopped is not None and stopped():
                raise WitnessCancelled("witness cancelled")
            if limited and timeout == 0:
                raise WitnessDeadline("confirmation budget exhausted")
            self.calls += 1
            self.connection.send(action)
            kind, value = self._receive(max(0, timeout - (time.monotonic() - started)), stopped)
            if kind != "observation" or not isinstance(value, Observation):
                raise WitnessFailure("witness_read_failed")
            return value
        except WitnessTimeout as exc:
            self.close()
            if limited:
                raise WitnessDeadline("confirmation budget exhausted") from exc
            raise
        except (EOFError, OSError) as exc:
            self.close()
            raise WitnessFailure("witness_disconnected") from exc
        except BaseException:
            self.close()
            raise

    def lifecycle_report(self):
        """Return a detached snapshot, not a promise of physical device release."""
        return {
            "state": self.cleanup_state,
            "closed": self.closed,
            "exitcode": self.exitcode,
            "requests": self.calls,
            "terminate_attempts": self.terminate_attempts,
            "kill_attempts": self.kill_attempts,
        }

    def close(self):
        self.closed = True
        self.cleanup_state = "unconfirmed"
        self.connection.close()
        if self.exitcode is not None:
            self.cleanup_state = "reaped"
            return True
        if self.process.pid is None:
            self.cleanup_state = "not_started"
            return True
        if self.process.is_alive():
            self.terminate_attempts += 1
            self.process.terminate()
        self.process.join(0.1)
        if self.process.is_alive():
            self.kill_attempts += 1
            self.process.kill()
            self.process.join(0.1)
        if self.process.is_alive():
            return False
        self.exitcode = self.process.exitcode
        self.process.close()
        self.cleanup_state = "reaped"
        return True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
