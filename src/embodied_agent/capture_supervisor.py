"""Opt-in process supervisor prototype for isolated blocking capture jobs."""

import math
import multiprocessing
import os
import threading
import time


def _watch_supervisor(connection):
    try:
        connection.recv_bytes()
    except (EOFError, OSError):
        pass
    os._exit(87)


def _run_supervised(job, connection):
    # This fallback does not replace the independent supervisor: SIGSTOP or
    # a native GIL hold can prevent this thread from running.
    threading.Thread(target=_watch_supervisor, args=(connection,), daemon=True).start()
    # Keep the read end open until process exit: closing it locally would
    # masquerade as supervisor loss to the monitor during normal completion.
    job()


def validate_runtime_limit(value):
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= 60
    ):
        raise ValueError("runtime_limit_s must be None or finite seconds in (0, 60]")


def _supervise(lifeline, reports, job, runtime_limit_s):
    ctx = multiprocessing.get_context("spawn")
    reader, writer = ctx.Pipe(duplex=False)
    child = ctx.Process(target=_run_supervised, args=(job, reader), daemon=True)
    started = time.monotonic()
    deadline = None if runtime_limit_s is None else started + runtime_limit_s
    try:
        child.start()
        reader.close()
        reports.send({"status": "started", "worker_pid": child.pid})
        owner_lost = False
        runtime_exceeded = False
        while child.is_alive():
            if deadline is not None and time.monotonic() >= deadline:
                runtime_exceeded = True
                break
            if lifeline.poll(0.01):
                try:
                    lifeline.recv_bytes()
                except (EOFError, OSError):
                    pass
                owner_lost = True
                break
        if child.is_alive():
            child.terminate()
            child.join(0.05)
        forced = child.is_alive()
        if forced:
            child.kill()
        child.join(1)
        result = {
            "status": "stopped" if not child.is_alive() else "cleanup_unconfirmed",
            "worker_pid": child.pid,
            "worker_exitcode": child.exitcode,
            "owner_connection_closed": owner_lost,
            "runtime_limit_exceeded": runtime_exceeded,
            "runtime_limit_s": runtime_limit_s,
            "elapsed_s": time.monotonic() - started,
            "kill_sent": forced,
            "hardware_cleanup_confirmed": False,
        }
        try:
            reports.send(result)
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if child.pid is not None:
            if child.is_alive():
                child.kill()
                child.join(1)
            if not child.is_alive():
                child.close()
        reader.close()
        writer.close()
        lifeline.close()
        reports.close()


class SupervisedCaptureJob:
    """One job owned by a separate supervisor, never by the caller alone.

    Not a drop-in ProcessPairCapture backend. This prototype establishes process
    ownership for blocking initialization/driver jobs; it does not transport frames.
    """

    def __init__(self, job, *, runtime_limit_s=None):
        if not callable(job):
            raise TypeError("spawn-picklable job required")
        validate_runtime_limit(runtime_limit_s)
        ctx = multiprocessing.get_context("spawn")
        reader, self._lifeline = ctx.Pipe(duplex=False)
        self._reports, writer = ctx.Pipe(duplex=False)
        self._process = ctx.Process(target=_supervise, args=(reader, writer, job, runtime_limit_s))
        self._events = []
        self._closed = False
        try:
            self._process.start()
        except Exception:
            self._lifeline.close()
            self._reports.close()
            self._process.close()
            raise
        finally:
            reader.close()
            writer.close()

    def snapshot(self):
        while self._reports.poll():
            try:
                self._events.append(self._reports.recv())
            except (EOFError, OSError):
                break
        final_report = bool(self._events and self._events[-1]["status"] == "stopped")
        alive = self._process.is_alive()
        return {
            "cleanup_status": "active"
            if alive
            else ("reported_stopped" if final_report else "supervisor_lost"),
            "supervisor_pid": self._process.pid,
            "supervisor_alive": self._process.is_alive(),
            "supervisor_exitcode": self._process.exitcode,
            "events": [dict(event) for event in self._events],
            "hardware_cleanup_confirmed": False,
        }

    def wait_started(self, timeout=5):
        if not 0 < timeout <= 5:
            raise ValueError("startup timeout must be within five seconds")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.snapshot()
            if result["events"]:
                return result
            if not result["supervisor_alive"]:
                break
            time.sleep(0.005)
        return self.snapshot()

    def close(self, timeout=2):
        if not 0 <= timeout <= 5:
            raise ValueError("close timeout must be within five seconds")
        if not self._closed:
            self._lifeline.close()
            self._closed = True
        self._process.join(timeout)
        # Never kill the supervisor before it can reap its own child.
        return self.snapshot()
