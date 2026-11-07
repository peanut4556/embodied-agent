"""Seeded, budget-preserving permutation of bounded observation intervals."""

import random

from .observation_schedule import PHASE_SWEEP_INTERVALS_MS


def randomized_observation_times_ms(seed):
    """16 captures in 3 seconds, with gaps <=248 ms, independent of scene truth.

    The interval multiset is frozen; only its ordering changes. This is not
    uniform sampling over all valid schedules or a bound on missed motion.
    Store the returned times along with the seed for durable reproducibility.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("nonnegative integer schedule seed required")
    intervals = list(PHASE_SWEEP_INTERVALS_MS)
    random.Random(seed).shuffle(intervals)
    times = [0]
    for interval in intervals:
        times.append(times[-1] + interval)
    return tuple(times)
