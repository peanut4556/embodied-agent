"""Opt-in advance reservations through candidate and fresh-witness confirmation."""

import math
from dataclasses import asdict, dataclass

from .phase_sweep_challenge import PhaseSweepChallengeController
from .scheduled_pair_capture import ScheduledPairRequest
from .serial_challenge import PairCaptureRequest
from .two_stage_capture import TwoStageCaptureController


def _valid_clock(now, host, start, last_task, last_host, mapping):
    if (
        not math.isfinite(now)
        or not math.isfinite(host)
        or now < start - 1e-9
        or (last_task is not None and now < last_task)
        or (last_host is not None and host < last_host)
    ):
        raise ValueError("monotonic finite reservation clocks required")
    return (
        host >= mapping.host_origin
        and abs(mapping.task_origin + (host - mapping.host_origin) * mapping.rate - now)
        <= mapping.max_error
    )


class ReservedCandidate(PhaseSweepChallengeController):
    """Same 31 exposure slots; reserve at most 20 ms before each slot.

    Request timeout still starts at actual issuance and includes the waiting
    period. Evidence deadlines and exposure windows are not extended.
    """

    lead_s = 0.02

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.check.clock_map.rate != 1.0:
            raise ValueError("reservation profile requires a unit-rate host clock map")

    def _clock(self, now, host):
        valid = _valid_clock(
            now,
            host,
            self.check.start - self.lead_s,
            self.last_task,
            self.last_host,
            self.check.clock_map,
        )
        self.last_task, self.last_host = now, host
        if not valid:
            self._fail("controller_clock_mismatch")
        return valid

    def poll(self, task_now, host_now, *, stop=False):
        if self.status != "pending":
            return None
        if stop:
            self._fail("user_stop", status="cancelled")
            return None
        if not self._clock(task_now, host_now) or self._expire(host_now):
            return None
        if task_now >= self.check.start:
            self.check.tick(task_now)
            self._sync()
        index = len(self.requests)
        if self.status != "pending" or index == len(self.schedule):
            return None
        due = self.schedule[index]
        if task_now < due - self.lead_s - 1e-9:
            return None
        if task_now > due:
            self._fail("reservation_submission_missed")
            return None
        if len(self.pending) >= self.capacity:
            self._fail("pipeline_capacity_exhausted")
            return None
        req = PairCaptureRequest(index + 1, self.check.session, task_now, host_now)
        self.requests.append(asdict(req))
        self.pending[req.request_id] = req
        self.peak_inflight = max(self.peak_inflight, len(self.pending))
        return req

    def result(self):
        return super().result() | {"reservation_lead_s": self.lead_s}


@dataclass(frozen=True)
class ReservedStageRequest(ScheduledPairRequest):
    stage: str
    task_time: float


class ReservedTwoStageController(TwoStageCaptureController):
    """36 requests maximum, candidate withheld until five NEW witness exposures.

    Begin polling 20 ms before start_time, after backend prewarm. The mapping's
    host origin must cover this reservation interval. Witnesses are reserved
    serially only after candidate completion, with 8 ms minimum dispatch lead.
    """

    def __init__(self, start_time, clock_map, session_id, *, position_error_m):
        super().__init__(start_time, clock_map, session_id, position_error_m=position_error_m)
        self.candidate = ReservedCandidate(
            start_time, clock_map, session_id, position_error_m=position_error_m
        )
        if clock_map.task_origin > start_time - self.candidate.lead_s + 1e-9:
            raise ValueError("clock map must cover the pre-start reservation interval")

    def _clock(self, now, host):
        valid = _valid_clock(
            now,
            host,
            self.candidate.check.start - self.candidate.lead_s,
            self.last_task,
            self.last_host,
            self.mapping,
        )
        self.last_task, self.last_host = now, host
        if not valid:
            self._end("controller_clock_mismatch")
        return valid

    def _request(self, now, host, local_id):
        if self.stage == "candidate":
            due = self.candidate.schedule[local_id - 1]
        else:
            due = max(now + 0.008, (self.gate.last_exposure or self.gate.start) + 0.01)
        target = self.mapping.host_origin + due - self.mapping.task_origin
        req = ReservedStageRequest(
            len(self.requests) + 1,
            self.candidate.check.session,
            host,
            target,
            target + 0.002,
            self.stage,
            now,
        )
        self.requests.append(asdict(req))
        self.pending[req.request_id] = (req, local_id)
        return req

    def poll(self, task_now, host_now, *, stop=False):
        if self.stage == "candidate" or stop or self.status != "pending":
            return super().poll(task_now, host_now, stop=stop)
        if not self._clock(task_now, host_now):
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
        if sum(r["stage"] == "witness" for r in self.requests) >= 5:
            self._end("witness_request_budget_exhausted")
            return None
        return self._request(task_now, host_now, None)

    def complete(self, request_id, task_now, host_now, primary, secondary, *, session_id):
        # Enforce the reservation again at the controller trust boundary. The
        # driver adapter alone is not evidence that a returned frame is valid.
        if self.status == "pending" and request_id in self.pending:
            req = self.pending[request_id][0]
            for point in (primary, secondary):
                stamp = getattr(point, "captured_at", None)
                if (
                    isinstance(stamp, bool)
                    or not isinstance(stamp, (int, float))
                    or not math.isfinite(stamp)
                    or not req.target_host_time - 1e-9 <= stamp <= req.latest_host_time + 1e-9
                ):
                    if self._clock(task_now, host_now):
                        self._end("reservation_exposure_rejected")
                    return False
        return super().complete(
            request_id, task_now, host_now, primary, secondary, session_id=session_id
        )

    def result(self):
        return super().result() | {"reservation_profile": "candidate_20ms_witness_8ms"}
