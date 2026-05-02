"""
Tests for the tier-weighted Thompson reward in src.bandit.

Defaults stay binary-compatible with the pre-Phase-2 swarm state file:
when tier_weighted=False, alpha grows by exactly the hit count.
"""

import numpy as np
import pytest

from src.bandit import default_entry, tier_reward, update


def test_tier_reward_table_monotonic():
    """alpha_delta must be non-decreasing in hits."""
    deltas = [tier_reward(h, main_count=7)[0] for h in range(8)]
    assert deltas == sorted(deltas), f"alpha_deltas not monotonic: {deltas}"


def test_tier_reward_three_match_dominates_three_one_matches():
    """A single 3-match must reward more than three separate 1-matches."""
    a3, _ = tier_reward(3, main_count=7)
    a1, _ = tier_reward(1, main_count=7)
    assert a3 > 3 * a1, f"3-match alpha={a3} <= 3 x 1-match alpha={3*a1}"


def test_tier_reward_seven_match_capped():
    """alpha_delta is capped at 50 so a single jackpot can't dominate."""
    a7, _ = tier_reward(7, main_count=7)
    assert a7 == 50.0


def test_tier_reward_clamps_negative_hits():
    a, b = tier_reward(-3, main_count=7)
    assert a == 0.0
    assert b == 7.0


def test_tier_reward_clamps_overshoot_hits():
    a, b = tier_reward(15, main_count=7)
    a7, b7 = tier_reward(7, main_count=7)
    assert (a, b) == (a7, b7), "hits > main_count must clamp to main_count"


def test_tier_reward_dailygrand_main_count_5():
    a, b = tier_reward(5, main_count=5)
    assert a > 0
    assert b == 0.0   # all 5 hit -> zero misses


def test_update_backward_compat_flag_false():
    """When flag False, alpha grows by exactly hits (legacy behavior)."""
    weights = {"lottomax": default_entry()}
    np.random.seed(0)
    update(weights, "lottomax", hits=4, main_count=7, tier_weighted=False)
    assert weights["lottomax"]["alpha"] == default_entry()["alpha"] + 4
    assert weights["lottomax"]["beta"]  == default_entry()["beta"]  + 3


def test_update_default_is_legacy():
    """The default (no kwarg) must match tier_weighted=False."""
    w_default = {"lottomax": default_entry()}
    w_explicit = {"lottomax": default_entry()}
    np.random.seed(0)
    update(w_default,  "lottomax", hits=2, main_count=7)
    np.random.seed(0)
    update(w_explicit, "lottomax", hits=2, main_count=7, tier_weighted=False)
    # alpha/beta deterministic, weight depends on RNG seed
    assert w_default["lottomax"]["alpha"] == w_explicit["lottomax"]["alpha"]
    assert w_default["lottomax"]["beta"]  == w_explicit["lottomax"]["beta"]


def test_update_tier_weighted_amplifies_three_match():
    weights_legacy = {"lottomax": default_entry()}
    weights_tiered = {"lottomax": default_entry()}
    update(weights_legacy, "lottomax", hits=3, main_count=7, tier_weighted=False)
    update(weights_tiered, "lottomax", hits=3, main_count=7, tier_weighted=True)
    assert weights_tiered["lottomax"]["alpha"] > weights_legacy["lottomax"]["alpha"]


def test_update_state_keys_unchanged():
    """Schema drift guard: keys after update must match default_entry()."""
    weights = {"649": default_entry()}
    update(weights, "649", hits=1, main_count=6, tier_weighted=True)
    assert set(weights["649"].keys()) == set(default_entry().keys())


def test_update_draws_scored_increments():
    weights = {"dailygrand": default_entry()}
    for h in (0, 1, 2):
        update(weights, "dailygrand", hits=h, main_count=5, tier_weighted=True)
    assert weights["dailygrand"]["draws_scored"] == 3
