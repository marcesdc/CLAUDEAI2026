"""
Per-play diversity guard for line sampling.

Without this guard, low-temperature softmax sampling can produce multiple
near-identical lines within a single play (mode collapse). The guard runs
a bounded rejection sampler: each candidate is compared against the lines
already chosen in the same play, and rejected if it shares more numbers
than `max_overlap` with any of them.

If the candidate distribution is so concentrated that no diverse line can
be found within `max_attempts` tries, the guard returns the last attempt
rather than hanging -- a `[diversity] gave up` message is the signal that
upstream temperature/probabilities are too sharp.
"""

from __future__ import annotations

import sys

import numpy as np


def is_too_similar(candidate: list[int],
                   existing: list[list[int]],
                   max_overlap: int) -> bool:
    """
    Return True if *candidate* shares more than *max_overlap* numbers with
    any line in *existing*.
    """
    cand_set = set(candidate)
    for line in existing:
        if len(cand_set & set(line)) > max_overlap:
            return True
    return False


def sample_diverse_line(rng: np.random.Generator,
                        main_probs: np.ndarray,
                        main_max: int,
                        main_count: int,
                        existing_lines: list[list[int]],
                        *,
                        max_overlap: int,
                        max_attempts: int = 20) -> list[int]:
    """
    Sample a single line of *main_count* unique numbers from a multinomial
    over `main_probs[:main_max]`, rejecting candidates that overlap an
    already-selected line by more than *max_overlap* numbers.

    Returns the first accepted candidate, or the last sampled candidate if
    all attempts violated the threshold (logs a warning to stderr).
    """
    probs = main_probs[:main_max]
    last_nums: list[int] = []
    for attempt in range(max_attempts):
        nums = sorted(
            int(v) + 1
            for v in rng.choice(main_max, size=main_count, replace=False, p=probs)
        )
        last_nums = nums
        if not existing_lines:
            return nums
        if not is_too_similar(nums, existing_lines, max_overlap):
            return nums

    print(
        f"[diversity] gave up after {max_attempts} attempts -- "
        f"distribution may be too concentrated (mode collapse signal).",
        file=sys.stderr,
    )
    return last_nums


def resolve_max_overlap(configured: int | None, main_count: int) -> int:
    """
    Resolve a configured max-overlap value. None means "main_count - 2",
    so e.g. LottoMax (7) allows up to 5 shared numbers, rejects 6-7.
    """
    if configured is None:
        return max(int(main_count) - 2, 0)
    return int(configured)
