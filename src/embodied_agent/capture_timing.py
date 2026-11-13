"""Offline decomposition of a traced capture; never grants task admission."""

import math
from itertools import pairwise


def analyze_capture_timing(trace, *, scheduled_host_at, exposure_window_s=0.002):
    """Host monotonic seconds throughout; device clocks must already be mapped.

    Missing/crashed/cancelled requests remain incomplete. Timestamp order errors
    invalidate the decomposition instead of being clipped to plausible durations.
    The largest observed segment is descriptive, not proof of a root cause.
    """
    if (
        isinstance(exposure_window_s, bool)
        or not isinstance(exposure_window_s, (int, float))
        or not math.isfinite(exposure_window_s)
        or exposure_window_s < 0
    ):
        raise ValueError("finite nonnegative host-clock exposure window required")
    fields = (
        "request_host_at",
        "submitted_at",
        "reader_started_at",
        "primary_exposure_at",
        "secondary_exposure_at",
        "reader_returned_at",
        "result_ready_at",
        "drained_at",
    )
    result = {
        "request_id": trace.get("request_id"),
        "status": "incomplete",
        "missing": [k for k in fields if k not in trace],
        "predicts_future_feasibility": False,
    }
    if result["missing"]:
        return result
    values = [scheduled_host_at, *(trace[k] for k in fields)]
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in values
    ):
        return result | {"status": "invalid_timestamps"}
    target, request, submitted, started, primary, secondary, returned, ready, drained = values
    first, last = min(primary, secondary), max(primary, secondary)
    ordered = (target, request, submitted, started, first, last, returned, ready, drained)
    if any(b < a for a, b in pairwise(ordered)):
        return result | {"status": "invalid_timestamp_order"}
    if trace.get("error"):
        return result | {"status": "reader_error"}
    spans = {
        "caller_poll": request - target,
        "caller_dispatch": submitted - request,
        "ipc_and_worker_scheduling": started - submitted,
        "reader_to_exposure": last - started,
        "exposure_to_reader_return": returned - last,
        "validation": ready - returned,
        "result_ipc_and_drain": drained - ready,
    }
    pre_exposure = tuple(spans)[:4]
    lateness = last - target
    return result | {
        "status": "measured",
        "segments_s": spans,
        "exposure_lateness_s": lateness,
        "exposure_margin_s": exposure_window_s - lateness,
        "exposure_window_met": lateness <= exposure_window_s + 1e-9,
        "largest_pre_exposure_segment": max(pre_exposure, key=spans.get),
        "drain_age_s": drained - first,
        "total_s": drained - target,
    }
