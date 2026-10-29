"""Experimental spawned-camera isolation. The child owns all device resources."""

import math
import multiprocessing
import time
import uuid

from .observation_capture import StampedPosition


def _serve(connection, session, factory):
    try:
        camera = factory()
        connection.send({"kind": "ready", "session": session})
        while True:
            request = connection.recv()
            if request["session"] != session:
                return
            try:
                value = camera.read()
                if value is not None and not isinstance(value, StampedPosition):
                    raise TypeError("camera process requires timestamped positions")
                if value is not None:
                    point = tuple(float(x) for x in value.position)
                    captured_at = float(value.captured_at)
                    if len(point) != 3 or not all(math.isfinite(x) for x in (*point, captured_at)):
                        raise ValueError("invalid camera process position")
                    value = StampedPosition(point, captured_at)
                packet = {"value": value, "error": None}
            except Exception as exc:  # noqa: BLE001 - isolate arbitrary driver exceptions
                packet = {"value": None, "error": f"{type(exc).__name__}: {exc}"[:512]}
            connection.send(
                {"kind": "result", "session": session, "request": request["request"], **packet}
            )
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:  # noqa: BLE001 - report camera initialization failure
        try:
            connection.send({"kind": "fatal", "session": session, "error": str(exc)[:512]})
        except (EOFError, BrokenPipeError):
            pass
    finally:
        connection.close()


class CameraProcessSession:
    """One pending request, isolated IPC per explicit restart; single owner thread.

    poll/request do not wait for a device read. close/restart may wait for two
    bounded shutdown intervals and belong in lifecycle management, not a servo loop.
    """

    def __init__(self, factory, *, timeout=0.25, startup_timeout=5.0, shutdown_timeout=0.2):
        if any(
            not math.isfinite(v) or v <= 0 for v in (timeout, startup_timeout, shutdown_timeout)
        ):
            raise ValueError("positive finite camera process timeouts required")
        self.factory = factory
        self.timeout, self.startup_timeout, self.shutdown_timeout = (
            timeout,
            startup_timeout,
            shutdown_timeout,
        )
        self.process = None
        self.connection = None
        self.last_exitcode = None
        self._start()

    def _start(self):
        context = multiprocessing.get_context("spawn")
        self.session = uuid.uuid4().hex
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_serve, args=(child, self.session, self.factory), daemon=True
        )
        try:
            self.process.start()
        except Exception:  # Release handles on spawn/pickling failure, then propagate.
            self.connection.close()
            self.process.close()
            self.process = None
            raise
        finally:
            child.close()
        self.status, self.error = "starting", None
        self.started = time.monotonic()
        self.request_id = 0
        self.requested_at = None

    def request(self):
        if self.status not in {"idle", "ready", "error"}:
            return False
        self.request_id += 1
        self.requested_at = time.monotonic()
        try:
            self.connection.send({"session": self.session, "request": self.request_id})
        except (BrokenPipeError, EOFError, OSError) as exc:
            self.status, self.error = "failed", str(exc)
            return False
        self.status, self.error = "pending", None
        return True

    def poll(self):
        if self.status not in {"starting", "pending"}:
            return None
        now = time.monotonic()
        deadline = (
            self.started + self.startup_timeout
            if self.status == "starting"
            else self.requested_at + self.timeout
        )
        if now >= deadline:
            self.status, self.error = "timed_out", "camera process deadline exceeded"
            return None
        if not self.connection.poll():
            if not self.process.is_alive():
                self.status, self.error = "failed", "camera process exited"
            return None
        try:
            packet = self.connection.recv()
        except (EOFError, OSError) as exc:
            self.status, self.error = "failed", str(exc)
            return None
        now = time.monotonic()
        if now >= deadline:
            self.status, self.error = "timed_out", "camera process deadline exceeded"
            return None
        if packet["session"] != self.session:
            self.status, self.error = "failed", "camera session mismatch"
        elif packet["kind"] == "fatal":
            self.status, self.error = "failed", packet["error"]
        elif self.status == "starting" and packet["kind"] == "ready":
            self.status = "idle"
        elif (
            self.status == "pending"
            and packet["kind"] == "result"
            and packet["request"] == self.request_id
        ):
            self.error = packet["error"]
            value = packet["value"]
            try:
                if self.error:
                    raise ValueError(self.error)
                if value is not None:
                    point = tuple(float(x) for x in value.position)
                    captured_at = float(value.captured_at)
                    if (
                        len(point) != 3
                        or not all(math.isfinite(x) for x in point)
                        or not math.isfinite(captured_at)
                        or not self.requested_at <= captured_at <= now
                    ):
                        raise ValueError("invalid or stale camera process observation")
                    value = StampedPosition(point, captured_at)
                self.status = "ready"
                return value
            except (ValueError, TypeError) as exc:
                self.status, self.error = "error", str(exc)
        else:
            self.status, self.error = "failed", "unexpected camera process packet"
        return None

    def close(self):
        if self.process is None:
            return
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(self.shutdown_timeout)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(self.shutdown_timeout)
        if self.process.is_alive():
            self.status = "cleanup_failed"
            raise RuntimeError("camera process could not be reaped; restart refused")
        self.last_exitcode = self.process.exitcode
        self.connection.close()
        self.process.close()
        self.process = None
        self.status = "closed"

    def restart(self, factory=None):
        self.close()
        if factory is not None:
            self.factory = factory
        self._start()
