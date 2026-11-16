"""Prewarmed spawned workers with isolated paired-reader ownership and bounded teardown."""

import copy
import functools
import math
import multiprocessing
import os
import threading
import time

from .capture_supervisor import SupervisedCaptureJob
from .observation_capture import StampedPosition
from .threaded_pair_capture import CapturedPair


def _watch_owner(lifeline):
    try:
        # No heartbeat traffic: EOF comes from closure of the sole writer.
        lifeline.recv_bytes()
    except (EOFError, OSError):
        pass
    os._exit(86)


def _serve(connection, factory, trace_timing=False, lifeline=None):
    if lifeline is not None:
        threading.Thread(target=_watch_owner, args=(lifeline,), daemon=True).start()
    try:
        reader = factory()
        connection.send(("ready", None))
        while True:
            req = connection.recv()
            timing = {"reader_started_at": time.monotonic()} if trace_timing else None
            try:
                pair = reader(req)
                if timing is not None:
                    timing["reader_returned_at"] = time.monotonic()
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
                ready = time.monotonic()
                if timing is not None:
                    timing.update(
                        result_ready_at=ready,
                        primary_exposure_at=pair.primary.captured_at,
                        secondary_exposure_at=pair.secondary.captured_at,
                    )
                connection.send(("result", (ready, req.request_id, pair, None, timing)))
            except Exception as exc:  # noqa: BLE001 - isolate reader failures
                connection.send(
                    (
                        "result",
                        (
                            time.monotonic(),
                            req.request_id,
                            None,
                            f"{type(exc).__name__}: {exc}"[:512],
                            timing,
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


class _SupervisedProcess:
    """Process-shaped adapter; never kill the owner of an unreaped reader."""

    def __init__(self, target):
        self.target, self.job = target, None

    def start(self):
        self.job = SupervisedCaptureJob(self.target)

    @property
    def pid(self):
        return self.job._process.pid

    @property
    def exitcode(self):
        return self.job._process.exitcode

    def is_alive(self):
        return self.job._process.is_alive()

    def terminate(self):
        self.job.close(timeout=0)

    def join(self, timeout):
        self.job._process.join(timeout)

    def details(self):
        return self.job.snapshot()

    def close(self):
        if self.job is not None:
            self.job.close(timeout=0)
            self.job._reports.close()
            self.job._process.close()


class ProcessPairCapture:
    """Two persistent processes, one outstanding request each, no shared device.

    factory must be spawn-picklable and create an independent reader in each child.
    wait_ready belongs before the task clock starts. cancel sends terminate;
    reap may escalate to kill and never equates process exit with hardware cleanup.
    """

    def __init__(
        self, factory, *, capacity=2, trace_timing=False, owner_watchdog=False, supervised=False
    ):
        if not callable(factory) or type(capacity) is not int or not 1 <= capacity <= 2:
            raise ValueError("callable factory and capacity of one or two required")
        self.closed, self.seen, self.slots = False, set(), []
        if type(trace_timing) is not bool:
            raise ValueError("trace_timing must be boolean")
        if type(owner_watchdog) is not bool:
            raise ValueError("owner_watchdog must be boolean")
        if type(supervised) is not bool:
            raise ValueError("supervised must be boolean")
        self.supervised = supervised
        self.owner_watchdog = owner_watchdog
        self.trace_timing, self.timings = trace_timing, {}
        ctx = multiprocessing.get_context("spawn")
        try:
            for _ in range(capacity):
                parent, child = ctx.Pipe()
                life_reader, life_writer = (
                    ctx.Pipe(duplex=False) if owner_watchdog else (None, None)
                )
                process = (
                    _SupervisedProcess(
                        functools.partial(_serve, child, factory, trace_timing, life_reader)
                    )
                    if supervised
                    else ctx.Process(
                        target=_serve, args=(child, factory, trace_timing, life_reader), daemon=True
                    )
                )
                slot = {
                    "process": process,
                    "lifeline": life_writer,
                    "connection": parent,
                    "ready": False,
                    "request": None,
                    "pid": None,
                    "exitcode": None,
                    "killed": False,
                    "supervision": None,
                }
                self.slots.append(slot)
                try:
                    process.start()
                    slot["pid"] = process.pid
                finally:
                    child.close()
                    if life_reader is not None:
                        life_reader.close()
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
                submitted = time.monotonic() if self.trace_timing else None
                slot["connection"].send(request)
                if self.trace_timing:
                    self.timings[request.request_id] = {
                        "request_id": request.request_id,
                        "request_host_at": request.host_time,
                        "submitted_at": submitted,
                    }
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
                    if self.trace_timing:
                        row = self.timings[request_id]
                        row.update(packet[4] or {})
                        row["drained_at"] = time.monotonic()
                        row["error"] = packet[3]
                except (EOFError, OSError, ValueError) as exc:
                    results.append((time.monotonic(), request_id, None, str(exc)[:512]))
                slot["request"] = None
            elif not slot["process"].is_alive():
                results.append((time.monotonic(), request_id, None, "capture process exited"))
                slot["request"] = None
        return [packet[1:4] for packet in sorted(results, key=lambda p: p[0])]

    def cancel(self):
        if self.closed:
            return
        self.closed = True
        for slot in self.slots:
            slot["connection"].close()
            if slot["lifeline"] is not None:
                slot["lifeline"].close()
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
                if (
                    p is not None
                    and slot["pid"] is not None
                    and p.is_alive()
                    and not isinstance(p, _SupervisedProcess)
                ):
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
                if isinstance(p, _SupervisedProcess):
                    slot["supervision"] = p.details()
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
        supervision = (
            [
                (s["process"].details() if s["process"] is not None else s["supervision"])
                for s in self.slots
            ]
            if self.supervised
            else []
        )
        verified = all(
            detail
            and detail["events"]
            and detail["events"][-1]["status"] == "stopped"
            and detail["events"][-1]["worker_exitcode"] is not None
            for detail in supervision
        )
        return {
            "backend": "spawned_process",
            "supervised": self.supervised,
            "supervision": supervision,
            "owner_watchdog_enabled": self.owner_watchdog,
            "closed": self.closed,
            "ready": not self.closed and all(s["ready"] for s in self.slots),
            "submitted": len(self.seen),
            "live_pids": live,
            "workers_stopped": not live and verified,
            "hardware_cleanup_confirmed": False,
            "timing_trace_enabled": self.trace_timing,
            "timing_trace": copy.deepcopy(list(self.timings.values())),
            "workers": [
                {k: s[k] for k in ("pid", "exitcode", "request", "killed")} for s in self.slots
            ],
        }
