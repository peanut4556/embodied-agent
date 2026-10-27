"""Experimental non-blocking capture boundary for independent camera drivers.

Never pass a renderer sharing mutable simulator state to the worker thread.
"""

import math
import threading
import time
from dataclasses import dataclass

from .release_observer import ReleaseObserver


@dataclass(frozen=True)
class StampedPosition:
    """Exposure time mapped by the driver into this host's monotonic clock."""

    position: tuple[float, float, float]
    captured_at: float


class ObservationCapture:
    """One in-flight call; consume each timely result once, quarantine late calls.

    read(cancel_event) must own its camera resources. Cancellation cannot forcibly
    interrupt native driver code; a stuck call prevents additional calls here.
    sample() and cancel() are intended for one controlling thread.
    """

    def __init__(self, read, *, timeout=0.25, clock=time.monotonic, require_timestamp=False):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("positive finite capture timeout required")
        self.read, self.timeout, self.clock = read, timeout, clock
        self.require_timestamp = require_timestamp
        self.cancel_event = threading.Event()
        self.lock = threading.Lock()
        self.job = None
        self.closed = False
        self.status = "idle"
        self.error = None
        self.last_captured_at = None
        self.last_age = None

    def _capture(self, job):
        point, error, captured_at = None, None, None
        try:
            value = self.read(self.cancel_event)
            returned_at = self.clock()
            if value is not None:
                if isinstance(value, StampedPosition):
                    captured_at = float(value.captured_at)
                    if (
                        not math.isfinite(captured_at)
                        or not job["start"] <= captured_at <= returned_at
                    ):
                        raise ValueError("capture timestamp outside request interval")
                    value = value.position
                elif self.require_timestamp:
                    raise ValueError("capture timestamp required")
                point = tuple(float(x) for x in value)
                if len(point) != 3 or not all(math.isfinite(x) for x in point):
                    raise ValueError("finite three-coordinate camera position required")
        except Exception as exc:  # noqa: BLE001 - isolate arbitrary camera-driver failures
            point = None
            captured_at = None
            error = f"{type(exc).__name__}: {exc}"
        with self.lock:
            job["done"] = True
            job["expired"] |= self.clock() - job["start"] >= self.timeout
            if not self.closed and not job["expired"]:
                job["point"], job["error"] = point, error
                job["captured_at"] = captured_at

    def sample(self):
        with self.lock:
            self.last_captured_at = self.last_age = None
            if self.closed:
                return None
            job = self.job
            if job is not None:
                job["expired"] |= self.clock() - job["start"] >= self.timeout
                if not job["done"]:
                    self.status = "timed_out" if job["expired"] else "pending"
                    return None
                self.job = None
                if not job["expired"]:
                    self.error = job["error"]
                    self.last_captured_at = job["captured_at"]
                    if self.last_captured_at is not None:
                        self.last_age = self.clock() - self.last_captured_at
                    self.status = (
                        "error" if self.error else ("ready" if job["point"] else "missing")
                    )
                    return list(job["point"]) if job["point"] is not None else None
                self.status = "timed_out"
            job = {
                "start": self.clock(),
                "done": False,
                "expired": False,
                "point": None,
                "error": None,
                "captured_at": None,
            }
            self.job = job
            self.error = None
            self.status = "pending"
            threading.Thread(target=self._capture, args=(job,), daemon=True).start()
            return None

    def request(self):
        """Prefetch only when no capture/result is outstanding (controller thread)."""
        if self.job is None and not self.closed:
            self.sample()

    def cancel(self):
        with self.lock:
            self.closed = True
            self.status = "cancelled"
            self.last_captured_at = self.last_age = None
            self.cancel_event.set()
            if self.job is not None:
                self.job["expired"] = True
                self.job["point"] = None


class CapturedReleaseObserver:
    """Explicit asynchronous adapter; leaves the default observer unchanged."""

    def __init__(self, capture, *, allow_recheck=False):
        self.capture = capture
        self.observer = ReleaseObserver(allow_recheck=allow_recheck)
        self.capture_records = []

    def step(self, timestamp, events, *, stop=False):
        if stop:
            self.capture.cancel()

        def sample():
            point = self.capture.sample()
            self.capture_records.append(
                {
                    "task_time": timestamp,
                    "captured_at": self.capture.last_captured_at,
                    "age_seconds": self.capture.last_age,
                    "capture_status": self.capture.status,
                    "error": self.capture.error,
                }
            )
            return point

        self.observer.step(timestamp, events, sample, stop=stop)
        if self.observer.status not in {"pending", "not_started"}:
            self.capture.cancel()
        elif self.observer.status == "pending":
            self.capture.request()
        return self.observer.status
