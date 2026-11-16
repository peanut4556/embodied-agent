"""Opt-in process supervisor prototype for isolated blocking capture jobs."""

import multiprocessing
import time


def _supervise(lifeline, reports, job):
    child = multiprocessing.get_context("spawn").Process(target=job, daemon=True)
    try:
        child.start()
        reports.send({"status": "started", "worker_pid": child.pid})
        owner_lost = False
        while child.is_alive():
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
        lifeline.close()
        reports.close()


class SupervisedCaptureJob:
    """One job owned by a separate supervisor, never by the caller alone.

    Not a drop-in ProcessPairCapture backend. This prototype establishes process
    ownership for blocking initialization/driver jobs; it does not transport frames.
    """

    def __init__(self, job):
        if not callable(job):
            raise TypeError("spawn-picklable job required")
        ctx = multiprocessing.get_context("spawn")
        reader, self._lifeline = ctx.Pipe(duplex=False)
        self._reports, writer = ctx.Pipe(duplex=False)
        self._process = ctx.Process(target=_supervise, args=(reader, writer, job))
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
        return {
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
