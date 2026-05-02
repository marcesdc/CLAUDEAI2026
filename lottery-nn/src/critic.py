"""
Reflexion-style critic for lottery line sampling.

Computes empirical statistics over historical draws (sum percentiles, max
consecutive run, decade concentration) and rejects candidate lines that
fall outside plausible ranges. No LLM call -- pure numpy over ~140 rows.

Stats are recomputed each predict() call from the live CSV; caching them
in swarm_state.json was rejected because it would couple critic correctness
to state-file freshness.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _max_consecutive(line: Iterable[int]) -> int:
    """Length of the longest run of consecutive integers in *line*."""
    nums = sorted(int(n) for n in line)
    if not nums:
        return 0
    longest = 1
    current = 1
    for prev, curr in zip(nums, nums[1:]):
        if curr == prev + 1:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 1
    return longest


def _decade_concentration(line: Iterable[int]) -> int:
    """Max number of line entries falling in the same decade (1-10, 11-20, ...)."""
    nums = [int(n) for n in line]
    if not nums:
        return 0
    counts: dict[int, int] = {}
    for n in nums:
        d = (n - 1) // 10
        counts[d] = counts.get(d, 0) + 1
    return max(counts.values())


def compute_lottery_stats(df: pd.DataFrame, cfg: dict,
                          percentiles: tuple[int, int] = (5, 95)) -> dict:
    """
    Build a stats dict from historical draws, used by critic_filter().

    Returns: {sum_p_lo, sum_p_hi, max_consecutive_p_hi, decade_concentration_p_hi}
    """
    main_count = int(cfg["main_count"])
    cols       = [f"n{i}" for i in range(1, main_count + 1)]
    if any(c not in df.columns for c in cols):
        raise KeyError(f"[critic] DataFrame missing columns {cols}; got {list(df.columns)}")

    arr = df[cols].to_numpy(dtype=int)
    sums       = arr.sum(axis=1)
    cons_runs  = np.array([_max_consecutive(row) for row in arr])
    decade_max = np.array([_decade_concentration(row) for row in arr])

    p_lo, p_hi = percentiles
    return {
        "main_count":                 main_count,
        "sum_p_lo":                   float(np.percentile(sums, p_lo)),
        "sum_p_hi":                   float(np.percentile(sums, p_hi)),
        "max_consecutive_p_hi":       float(np.percentile(cons_runs, p_hi)),
        "decade_concentration_p_hi":  float(np.percentile(decade_max, p_hi)),
    }


def critic_filter(line: list[int], stats: dict) -> bool:
    """
    Return True if *line* passes all critic rules, False to reject.

    Rejection rules:
      - sum outside [sum_p_lo, sum_p_hi]
      - longest consecutive run exceeds historical p_hi
      - all numbers in same decade beyond historical p_hi
      - all-even or all-odd
    """
    nums = list(line)
    if not nums:
        return False

    s = sum(nums)
    if not (stats["sum_p_lo"] <= s <= stats["sum_p_hi"]):
        return False

    if _max_consecutive(nums) > stats["max_consecutive_p_hi"]:
        return False

    if _decade_concentration(nums) > stats["decade_concentration_p_hi"]:
        return False

    parities = {n % 2 for n in nums}
    if len(parities) == 1:           # all-even or all-odd
        return False

    return True
