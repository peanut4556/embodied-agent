"""One caller-facing request protocol spanning candidate and fresh-witness stages."""

import copy
import math
from dataclasses import asdict, dataclass

from .fresh_witness import FreshWitnessGate
from .observation_capture import StampedPosition
from .phase_sweep_challenge import PhaseSweepChallengeController


@dataclass(frozen=True)
class StageCaptureRequest:
    request_id: int
    stage: str
    session_id: str
    task_time: float
    host_time: float


class TwoStageCaptureController:
    """Single controller thread; driver execution and cancellation stay external.

    Candidate success is never top-level success. Witness requests are serial,
    numbered after candidate requests, and must capture after their own request.
    A witness request has a fixed 30 ms task-clock timeout; no overlapping retry.
    """

    def __init__(self, start_time, clock_map, session_id, *, position_error_m):
        self.candidate = PhaseSweepChallengeController(
            start_time,
            clock_map,
            session_id,
            position_error_m=position_error_m,
        )
        self.gate = None
        self.stage = "candidate"
        self.status, self.reason = "pending", "collecting_candidate"
        self.mapping = clock_map
        self.last_task = self.last_host = None
        self.requests, self.pending = [], {}
        self.ignored_results = 0

    def _end(self, reason, status="unconfirmed"):
        self.status, self.reason = status, reason

    def _clock(self, now, host):
        if (
            not math.isfinite(now)
            or not math.isfinite(host)
            or now < self.candidate.check.start
            or (self.last_task is not None and now < self.last_task)
            or (self.last_host is not None and host < self.last_host)
        ):
            raise ValueError("monotonic finite controller clocks required")
        self.last_task, self.last_host = now, host
        m = self.mapping
        if (
            host < m.host_origin
            or abs(m.task_origin + (host - m.host_origin) * m.rate - now) > m.max_error
        ):
            self._end("controller_clock_mismatch")
            return False
        return True

    def _sync(self):
        if self.stage == "candidate":
            if self.candidate.status == "confirmed_in_tray":
                if self.pending:
                    self._end("candidate_has_outstanding_requests")
                else:
                    self.gate = FreshWitnessGate(self.candidate)
                    self.stage, self.reason = "witness", "candidate_withheld"
            elif self.candidate.status != "pending":
                self._end(self.candidate.reason, self.candidate.status)
        elif self.gate.status != "pending":
            self._end(self.gate.reason, self.gate.status)

    def _request(self, now, host, local_id):
        req = StageCaptureRequest(
            len(self.requests) + 1, self.stage, self.candidate.check.session, now, host
        )
        self.requests.append(asdict(req))
        self.pending[req.request_id] = (req, local_id)
        return req

    def poll(self, task_now, host_now, *, stop=False):
        if self.status != "pending":
            return None
        if stop:
            self._end("user_stop", "cancelled")
            return None
        if not self._clock(task_now, host_now):
            return None
        if self.stage == "candidate":
            req = self.candidate.poll(task_now, host_now)
            if req is not None:
                return self._request(task_now, host_now, req.request_id)
            self._sync()
            return None
        self.gate.tick(task_now)
        self._sync()
        if self.status != "pending":
            return None
        if self.pending:
            req = next(iter(self.pending.values()))[0]
            if task_now - req.task_time >= 0.03 - 1e-9:
                self._end("witness_request_timeout")
            return None
        if self.gate.last_exposure is not None and task_now < self.gate.last_exposure + 0.01 - 1e-9:
            return None
        if sum(r["stage"] == "witness" for r in self.requests) >= 5:
            self._end("witness_request_budget_exhausted")
            return None
        return self._request(task_now, host_now, None)

    def complete(self, request_id, task_now, host_now, primary, secondary, *, session_id):
        if self.status != "pending" or request_id not in self.pending:
            self.ignored_results += 1
            return False
        if not self._clock(task_now, host_now):
            return False
        req, local_id = self.pending[request_id]
        if req.stage == "candidate":
            consumed = self.candidate.complete(
                local_id, task_now, host_now, primary, secondary, session_id=session_id
            )
            if consumed:
                del self.pending[request_id]
            self._sync()
            return consumed
        if task_now - req.task_time >= 0.03 - 1e-9:
            self._end("witness_request_timeout")
            return False
        del self.pending[request_id]
        if any(not isinstance(p, StampedPosition) for p in (primary, secondary)):
            self._end("missing_timestamped_pair")
        elif any(
            isinstance(p.captured_at, bool)
            or not isinstance(p.captured_at, (int, float))
            or not math.isfinite(p.captured_at)
            or p.captured_at < req.host_time - 1e-9
            or p.captured_at > host_now
            for p in (primary, secondary)
        ):
            self._end("capture_outside_request_interval")
        else:
            self.gate.deliver(task_now, host_now, primary, secondary, session_id=session_id)
            self._sync()
        return True

    def result(self):
        return copy.deepcopy(
            {
                "status": self.status,
                "reason": self.reason,
                "stage": self.stage,
                "request_limit": 36,
                "requests": self.requests,
                "pending": [asdict(r) for r, _ in self.pending.values()],
                "cancel_request_ids": list(self.pending) if self.status != "pending" else [],
                "driver_cancel_required": self.status != "pending" and bool(self.pending),
                "ignored_results": self.ignored_results,
                "candidate": self.candidate.result(),
                "witness": self.gate.result() if self.gate else None,
            }
        )
