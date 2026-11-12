"""Bounded real worker threads for isolated, cooperative paired camera readers.

Never share mutable simulator/render state with these workers. Readers must own
or safely serialize their resources. A cancellation signal cannot kill native code.
"""

import math
import threading
import time
from collections import deque
from dataclasses import dataclass

from .observation_capture import StampedPosition


@dataclass(frozen=True)
class CapturedPair:
    session_id: str
    primary: StampedPosition
    secondary: StampedPosition


class ThreadedPairCapture:
    """At most two outstanding calls/results; no hidden executor work queue.

    submit/drain/cancel/reap belong to one control thread. Worker completion order
    is preserved. The read(request, cancel_event) callable must return CapturedPair.
    """

    def __init__(self, read, *, capacity=2):
        if not callable(read) or type(capacity) is not int or not 1 <= capacity <= 2:
            raise ValueError("callable reader and capacity of one or two required")
        self.read, self.capacity = read, capacity
        self.cancel_event = threading.Event()
        self.lock = threading.Lock()
        self.closed = False
        self.jobs, self.threads = {}, {}
        self.ready = deque()
        self.seen = set()
        self.discarded = 0

    def submit(self, request):
        with self.lock:
            if (
                self.closed
                or len(self.jobs) >= self.capacity
                or request.request_id in self.seen
                or len(self.seen) >= 36
            ):
                return False
            self.seen.add(request.request_id)
            self.jobs[request.request_id] = request
            thread = threading.Thread(target=self._read, args=(request,), daemon=True)
            self.threads[request.request_id] = thread
            try:
                thread.start()
            except RuntimeError:
                del self.jobs[request.request_id]
                del self.threads[request.request_id]
                raise
            return True

    def _read(self, request):
        pair, error = None, None
        try:
            pair = self.read(request, self.cancel_event)
            if not isinstance(pair, CapturedPair):
                raise TypeError("CapturedPair required")
        except Exception as exc:  # noqa: BLE001 - isolate arbitrary driver failures
            error = f"{type(exc).__name__}: {exc}"[:512]
        with self.lock:
            if self.closed:
                self.discarded += 1
            else:
                self.ready.append((request.request_id, pair, error))

    def drain(self):
        with self.lock:
            results = list(self.ready)
            self.ready.clear()
            for request_id, _, _ in results:
                self.jobs.pop(request_id, None)
            return results

    def cancel(self):
        with self.lock:
            self.closed = True
            self.cancel_event.set()
            self.discarded += len(self.ready)
            self.ready.clear()

    def reap(self, timeout=0.0):
        """Bounded wait for worker return; does not claim hardware cleanup."""
        if not math.isfinite(timeout) or not 0 <= timeout <= 0.25:
            raise ValueError("reap timeout must be between zero and 250 ms")
        deadline = time.monotonic() + timeout
        for thread in tuple(self.threads.values()):
            thread.join(max(0, deadline - time.monotonic()))
        with self.lock:
            if self.closed:
                for request_id, thread in self.threads.items():
                    if not thread.is_alive():
                        self.jobs.pop(request_id, None)
        return self.result()

    def result(self):
        with self.lock:
            live = [i for i, thread in self.threads.items() if thread.is_alive()]
            return {
                "closed": self.closed,
                "submitted": len(self.seen),
                "outstanding_slots": len(self.jobs),
                "ready_results": len(self.ready),
                "live_request_ids": live,
                "workers_stopped": not live,
                "discarded_results": self.discarded,
                "hardware_cleanup_confirmed": False,
            }
