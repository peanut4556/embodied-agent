"""Experimental non-blocking capture boundary for independent camera drivers.

Never pass a renderer sharing mutable simulator state to the worker thread.
"""

import math
import threading
import time

from .release_observer import ReleaseObserver


class ObservationCapture:
    """One in-flight call; consume each timely result once, quarantine late calls.

    read(cancel_event) must own its camera resources. Cancellation cannot forcibly
    interrupt native driver code; a stuck call prevents additional calls here.
    sample() and cancel() are intended for one controlling thread.
    """

    def __init__(self, read, *, timeout=0.25, clock=time.monotonic):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("positive finite capture timeout required")
        self.read, self.timeout, self.clock = read, timeout, clock
        self.cancel_event = threading.Event()
        self.lock = threading.Lock()
        self.job = None
        self.closed = False
        self.status = "idle"
        self.error = None

    def _capture(self, job):
        point, error = None, None
        try:
            value = self.read(self.cancel_event)
            if value is not None:
                point = tuple(float(x) for x in value)
                if len(point) != 3 or not all(math.isfinite(x) for x in point):
                    raise ValueError("finite three-coordinate camera position required")
        except Exception as exc:  # noqa: BLE001 - isolate arbitrary camera-driver failures
            point = None
            error = f"{type(exc).__name__}: {exc}"
        with self.lock:
            job["done"] = True
            job["expired"] |= self.clock() - job["start"] >= self.timeout
            if not self.closed and not job["expired"]:
                job["point"], job["error"] = point, error

    def sample(self):
        with self.lock:
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
            self.cancel_event.set()
            if self.job is not None:
                self.job["expired"] = True
                self.job["point"] = None


class CapturedReleaseObserver:
    """Explicit asynchronous adapter; leaves the default observer unchanged."""

    def __init__(self, capture, *, allow_recheck=False):
        self.capture = capture
        self.observer = ReleaseObserver(allow_recheck=allow_recheck)

    def step(self, timestamp, events, *, stop=False):
        if stop:
            self.capture.cancel()
        self.observer.step(timestamp, events, self.capture.sample, stop=stop)
        if self.observer.status not in {"pending", "not_started"}:
            self.capture.cancel()
        elif self.observer.status == "pending":
            self.capture.request()
        return self.observer.status
