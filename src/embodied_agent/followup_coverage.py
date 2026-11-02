"""Descriptive exposure coverage; no inference of physical safety."""


class FollowupCoverage:
    """Count intervals bracketed by consecutive valid exposures at most 250 ms apart.

    Missing, rejected or expired evidence breaks the chain. Single exposures
    contribute sample counts but no duration. No forward fill to the deadline.
    """

    def __init__(self, start, deadline):
        self.start, self.deadline = start, deadline
        self.counts = dict.fromkeys(("valid", "missing", "rejected", "expired"), 0)
        self.previous = self.first = self.last = None
        self.intervals = []

    def record(self, outcome, exposure=None):
        if outcome not in self.counts:
            raise ValueError("unknown coverage outcome")
        if outcome == "valid":
            if exposure is None or not self.start < exposure < self.deadline:
                raise ValueError("exposure outside follow-up window")
            if self.last is not None and exposure <= self.last:
                raise ValueError("ordered coverage exposures required")
            if self.previous is not None and exposure - self.previous <= 0.25:
                if self.intervals and self.intervals[-1][1] == self.previous:
                    self.intervals[-1][1] = exposure
                else:
                    self.intervals.append([self.previous, exposure])
            self.previous = self.last = exposure
            if self.first is None:
                self.first = exposure
        else:
            self.previous = None
        self.counts[outcome] += 1

    def result(self, status):
        duration = self.deadline - self.start
        covered = sum(end - start for start, end in self.intervals)
        cursor, longest = self.start, 0
        for start, end in self.intervals:
            longest = max(longest, start - cursor)
            cursor = end
        longest = max(longest, self.deadline - cursor)
        return {
            "schema_version": 1,
            "definition": "consecutive_valid_exposure_intervals",
            "planned_seconds": duration,
            "covered_seconds": covered,
            "uncovered_seconds": duration - covered,
            "coverage_fraction": covered / duration,
            "longest_uncovered_seconds": longest,
            "counts": dict(self.counts),
            "first_valid_exposure": self.first,
            "last_valid_exposure": self.last,
            "window_completed": status == "completed",
            "assessment": (
                "no_valid_observations"
                if self.first is None
                else "window_incomplete"
                if status != "completed"
                else "observations_with_gaps"
                if covered < duration
                else "window_bracketed"
            ),
        }
