"""
Statistical analysis helpers – useful for exploratory analysis before training.

Functions
---------
frequency_table   : count how often each number has appeared.
hot_cold          : identify 'hot' (recent) and 'cold' (overdue) numbers.
pair_frequency    : most common number pairs.
gap_analysis      : average gap between appearances of each number.
"""

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

import config

MAIN_MAX = config.LOTTERY["main_max"]
BONUS_MAX = config.LOTTERY["bonus_max"]


def frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with the draw frequency of each main ball."""
    main_cols = [c for c in df.columns if c.startswith("n")]
    counts = Counter()
    for row in df[main_cols].values:
        counts.update(int(v) for v in row)

    freq = pd.DataFrame(
        {"number": range(1, MAIN_MAX + 1),
         "count": [counts.get(i, 0) for i in range(1, MAIN_MAX + 1)]}
    )
    freq["pct"] = freq["count"] / len(df) * 100
    return freq.sort_values("count", ascending=False).reset_index(drop=True)


def hot_cold(df: pd.DataFrame, window: int = 30) -> dict:
    """
    Identify hot (drawn in last *window* draws) and cold (not drawn) numbers.
    """
    main_cols = [c for c in df.columns if c.startswith("n")]
    recent = df.tail(window)
    seen = set()
    for row in recent[main_cols].values:
        seen.update(int(v) for v in row)

    hot = sorted(seen)
    cold = sorted(set(range(1, MAIN_MAX + 1)) - seen)
    return {"hot": hot, "cold": cold}


def pair_frequency(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Return the *top_n* most common pairs of main numbers."""
    main_cols = [c for c in df.columns if c.startswith("n")]
    counts = Counter()
    for row in df[main_cols].values:
        counts.update(combinations(sorted(int(v) for v in row), 2))

    rows = [{"pair": p, "count": c} for p, c in counts.most_common(top_n)]
    return pd.DataFrame(rows)


def gap_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each ball 1..MAIN_MAX, compute average gap (draws between appearances).
    """
    main_cols = [c for c in df.columns if c.startswith("n")]
    N = len(df)
    gaps = {i: [] for i in range(1, MAIN_MAX + 1)}
    last_seen = {}

    for idx, row in enumerate(df[main_cols].values):
        for v in row:
            v = int(v)
            if v in last_seen:
                gaps[v].append(idx - last_seen[v])
            last_seen[v] = idx

    result = []
    for num in range(1, MAIN_MAX + 1):
        g = gaps[num]
        result.append({
            "number": num,
            "appearances": len(g) + (1 if num in last_seen else 0),
            "avg_gap": np.mean(g) if g else None,
            "last_seen_ago": N - 1 - last_seen[num] if num in last_seen else None,
        })
    return pd.DataFrame(result).sort_values("avg_gap").reset_index(drop=True)
