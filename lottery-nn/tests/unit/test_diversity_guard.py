"""
Tests for the per-play diversity guard (B2).

The guard runs a bounded rejection sampler that rejects candidate lines
overlapping any existing line in the same play by more than max_overlap.
"""

import numpy as np
import pytest

from src.diversity import is_too_similar, resolve_max_overlap, sample_diverse_line


# ---------------------------------------------------------------------------
# is_too_similar
# ---------------------------------------------------------------------------

def test_is_too_similar_overlap_threshold():
    a = [1, 2, 3, 4, 5, 6, 7]
    b = [1, 2, 3, 4, 5, 6, 99]   # 6 shared
    c = [1, 2, 3, 4, 50, 51, 52] # 4 shared
    assert is_too_similar(a, [b], max_overlap=5) is True
    assert is_too_similar(a, [c], max_overlap=5) is False


def test_is_too_similar_empty_existing():
    assert is_too_similar([1, 2, 3], [], max_overlap=2) is False


def test_is_too_similar_any_match_triggers():
    a = [1, 2, 3, 4, 5, 6, 7]
    fine = [10, 11, 12, 13, 14, 15, 16]
    bad  = [1, 2, 3, 4, 5, 6, 99]
    # Even with one good peer, one bad peer still flags as too similar.
    assert is_too_similar(a, [fine, bad], max_overlap=5) is True


# ---------------------------------------------------------------------------
# resolve_max_overlap
# ---------------------------------------------------------------------------

def test_resolve_max_overlap_default_lottomax():
    assert resolve_max_overlap(None, 7) == 5


def test_resolve_max_overlap_default_649():
    assert resolve_max_overlap(None, 6) == 4


def test_resolve_max_overlap_default_dailygrand():
    assert resolve_max_overlap(None, 5) == 3


def test_resolve_max_overlap_explicit_overrides():
    assert resolve_max_overlap(2, 7) == 2


# ---------------------------------------------------------------------------
# sample_diverse_line
# ---------------------------------------------------------------------------

def test_sample_diverse_line_no_existing_passes_first_attempt():
    rng = np.random.default_rng(0)
    probs = np.full(52, 1.0 / 52)
    nums = sample_diverse_line(rng, probs, 52, 7, [], max_overlap=5, max_attempts=20)
    assert len(nums) == 7
    assert len(set(nums)) == 7
    assert all(1 <= n <= 52 for n in nums)


def test_sample_diverse_line_avoids_duplicate_when_possible():
    rng = np.random.default_rng(123)
    probs = np.full(52, 1.0 / 52)
    existing = [[1, 2, 3, 4, 5, 6, 7]]
    nums = sample_diverse_line(rng, probs, 52, 7, existing, max_overlap=5, max_attempts=50)
    overlap = len(set(nums) & set(existing[0]))
    assert overlap <= 5, f"sampler returned {overlap}-overlap line: {nums}"


def test_sample_diverse_line_falls_back_after_max_attempts(capsys):
    """When probs are concentrated on exactly main_count numbers, every
    candidate is identical and the guard falls back without raising."""
    rng = np.random.default_rng(0)
    probs = np.zeros(52)
    probs[:7] = 1.0 / 7   # the only possible line is sorted({1..7})
    existing = [[1, 2, 3, 4, 5, 6, 7]]
    nums = sample_diverse_line(rng, probs, 52, 7, existing,
                                max_overlap=5, max_attempts=10)
    assert sorted(nums) == [1, 2, 3, 4, 5, 6, 7]
    captured = capsys.readouterr()
    assert "gave up" in captured.err


def test_sample_diverse_line_returns_unique_sorted():
    rng = np.random.default_rng(7)
    probs = np.full(49, 1.0 / 49)
    nums = sample_diverse_line(rng, probs, 49, 6, [], max_overlap=4, max_attempts=20)
    assert nums == sorted(nums)
    assert len(set(nums)) == 6


# ---------------------------------------------------------------------------
# Integration: lines within a play stay pairwise-distinct
# ---------------------------------------------------------------------------

def test_three_lines_pairwise_overlap_within_threshold():
    rng = np.random.default_rng(42)
    probs = np.full(52, 1.0 / 52)
    main_count, main_max, max_overlap = 7, 52, 5
    lines = []
    for _ in range(3):
        nums = sample_diverse_line(rng, probs, main_max, main_count,
                                    lines, max_overlap=max_overlap, max_attempts=30)
        lines.append(nums)
    assert len(lines) == 3
    for i in range(3):
        for j in range(i + 1, 3):
            shared = len(set(lines[i]) & set(lines[j]))
            assert shared <= max_overlap, \
                f"lines {i},{j} share {shared}: {lines[i]} vs {lines[j]}"
