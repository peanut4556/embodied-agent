"""Prewarmed spawned workers with isolated paired-reader ownership and bounded teardown."""

import math
import multiprocessing
import time

from .observation_capture import StampedPosition
from .threaded_pair_capture import CapturedPair


def _serve(connection, factory):
    try:
        reader = factory()
        connection.send(("ready", None))
        while True:
            req = connection.recv()
            try:
                pair = reader(req)
                if not isinstance(pair, CapturedPair):
                    raise TypeError("CapturedPair required")
                if not isinstance(pair.session_id, str) or len(pair.session_id) > 256:
                    raise ValueError("bounded session string required")
                points = []
                for value in (pair.primary, pair.secondary):
                    if not isinstance(value, StampedPosition) or len(value.position) != 3:
                        raise ValueError("timestamped three-coordinate pair required")
                    xyz = tuple(float(x) for x in value.position)
                    stamp = float(value.captured_at)
                    if not all(math.isfinite(x) for x in (*xyz, stamp)):
                        raise ValueError("finite paired coordinates required")
                    points.append(StampedPosition(xyz, stamp))
                pair = CapturedPair(pair.session_id, *points)
                connection.send(("result", (time.monotonic(), req.request_id, pair, None)))
            except Exception as exc:  # noqa: BLE001 - isolate reader failures
                connection.send(
                    (
                        "result",
                        (
                            time.monotonic(),
                            req.request_id,
                            None,
                            f"{type(exc).__name__}: {exc}"[:512],
                        ),
                    )
                )
    except (EOFError, BrokenPipeError, OSError):
        pass
    except Exception as exc:  # noqa: BLE001 - report factory failures
        try:
            connection.send(("fatal", str(exc)[:512]))
        except (EOFError, OSError):
            pass
    finally:
        connection.close()


class ProcessPairCapture:
    """Two persistent processes, one outstanding request each, no shared device.

    factory must be spawn-picklable and create an independent reader in each child.
    wait_ready belongs before the task clock starts. cancel sends terminate;
    reap may escalate to kill and never equates process exit with hardware cleanup.
    """

    def __init__(self, factory, *, capacity=2):
        if not callable(factory) or type(capacity) is not int or not 1 <= capacity <= 2:
            raise ValueError("callable factory and capacity of one or two required")
        self.closed, self.seen, self.slots = False, set(), []
        ctx = multiprocessing.get_context("spawn")
        try:
            for _ in range(capacity):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_serve, args=(child, factory), daemon=True)
                slot = {
                    "process": process,
                    "connection": parent,
                    "ready": False,
                    "request": None,
                    "pid": None,
                    "exitcode": None,
                    "killed": False,
                }
                self.slots.append(slot)
                try:
                    process.start()
                    slot["pid"] = process.pid
                finally:
                    child.close()
        except Exception:
            self.cancel()
            self.reap(0.25)
            raise

    def wait_ready(self, timeout=5.0):
        if not 0 < timeout <= 5:
            raise ValueError("startup timeout must be within five seconds")
        deadline = time.monotonic() + timeout
        while not self.closed and time.monotonic() < deadline:
            for slot in self.slots:
                if slot["ready"]:
                    continue
                pipe = slot["connection"]
                if pipe.poll():
                    try:
                        kind, _ = pipe.recv()
                    except (EOFError, OSError):
                        kind = "fatal"
                    if kind != "ready":
                        self.cancel()
                        return False
                    slot["ready"] = True
                elif not slot["process"].is_alive():
                    self.cancel()
                    return False
            if all(s["ready"] for s in self.slots):
                return True
            time.sleep(0.001)
        self.cancel()
        return False

    def submit(self, request):
        if self.closed or request.request_id in self.seen or len(self.seen) >= 36:
            return False
        for slot in self.slots:
            if slot["ready"] and slot["request"] is None and slot["process"].is_alive():
                slot["connection"].send(request)
                slot["request"] = request.request_id
                self.seen.add(request.request_id)
                return True
        return False

    def drain(self):
        if self.closed:
            return []
        results = []
        # Merge available packets by worker completion time, never by request ID.
        for slot in self.slots:
            request_id = slot["request"]
            if request_id is None:
                continue
            if slot["connection"].poll():
                try:
                    kind, packet = slot["connection"].recv()
                    if kind != "result" or packet[1] != request_id:
                        raise ValueError("worker request mismatch")
                    results.append(packet)
                except (EOFError, OSError, ValueError) as exc:
                    results.append((time.monotonic(), request_id, None, str(exc)[:512]))
                slot["request"] = None
            elif not slot["process"].is_alive():
                results.append((time.monotonic(), request_id, None, "capture process exited"))
                slot["request"] = None
        return [packet[1:] for packet in sorted(results, key=lambda p: p[0])]

    def cancel(self):
        if self.closed:
            return
        self.closed = True
        for slot in self.slots:
            slot["connection"].close()
            if slot["pid"] is not None and slot["process"].is_alive():
                slot["process"].terminate()

    def reap(self, timeout=0.0):
        if not 0 <= timeout <= 0.25:
            raise ValueError("reap timeout must be within 250 ms")
        start = time.monotonic()
        for slot in self.slots:
            p = slot["process"]
            if p is not None and slot["pid"] is not None:
                p.join(max(0, start + timeout / 2 - time.monotonic()))
        if self.closed:
            for slot in self.slots:
                p = slot["process"]
                if p is not None and slot["pid"] is not None and p.is_alive():
                    p.kill()
                    slot["killed"] = True
        for slot in self.slots:
            p = slot["process"]
            if p is None:
                continue
            if slot["pid"] is not None:
                p.join(max(0, start + timeout - time.monotonic()))
                if p.is_alive():
                    continue
                slot["exitcode"] = p.exitcode
            if self.closed:
                p.close()
                slot["process"] = None
                slot["request"] = None
        return self.result()

    def result(self):
        live = [
            s["pid"]
            for s in self.slots
            if s["process"] is not None and s["pid"] is not None and s["process"].is_alive()
        ]
        return {
            "backend": "spawned_process",
            "closed": self.closed,
            "ready": not self.closed and all(s["ready"] for s in self.slots),
            "submitted": len(self.seen),
            "live_pids": live,
            "workers_stopped": not live,
            "hardware_cleanup_confirmed": False,
            "workers": [
                {k: s[k] for k in ("pid", "exitcode", "request", "killed")} for s in self.slots
            ],
        }
