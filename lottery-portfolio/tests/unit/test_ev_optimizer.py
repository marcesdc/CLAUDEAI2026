"""Unit tests for src/ev_optimizer.py."""

import math

import numpy as np
import pytest

import config
from src import ev_optimizer as ev
from src import popularity_model as pm


# ---------------------------------------------------------------------------
# Hypergeometric helper
# ---------------------------------------------------------------------------

def test_hypergeometric_prob_sums_to_one():
    """Across all k, hypergeometric probabilities sum to 1 for each lottery."""
    for lottery in ["lottomax", "649", "dailygrand"]:
        cfg = config.LOTTERY_RULES[lottery]
        total = sum(
            ev.hypergeometric_prob(cfg["main_max"], cfg["main_count"], k)
            for k in range(cfg["main_count"] + 1)
        )
        assert math.isclose(total, 1.0, abs_tol=1e-12), f"{lottery}: total={total}"


def test_hypergeometric_prob_k_zero_is_non_trivial():
    """P(0 matches) = C(main_max-main_count, main_count) / C(main_max, main_count)."""
    p = ev.hypergeometric_prob(49, 6, 0)
    expected = math.comb(43, 6) / math.comb(49, 6)
    assert math.isclose(p, expected, rel_tol=1e-12)


def test_hypergeometric_prob_perfect_match():
    p = ev.hypergeometric_prob(49, 6, 6)
    expected = 1.0 / math.comb(49, 6)
    assert math.isclose(p, expected, rel_tol=1e-12)


def test_hypergeometric_prob_out_of_range():
    assert ev.hypergeometric_prob(49, 6, -1) == 0.0
    assert ev.hypergeometric_prob(49, 6, 7) == 0.0


# ---------------------------------------------------------------------------
# Bonus-match probability
# ---------------------------------------------------------------------------

def test_bonus_match_prob_lottomax_no_match_gives_all_remaining():
    cfg = config.LOTTERY_RULES["lottomax"]  # main=7, max=52
    # If we matched 0 main, all 7 picks are in the remaining pool of (52-7)=45
    p = ev.bonus_match_prob(cfg, tier_main=0)
    assert math.isclose(p, 7.0 / 45.0, rel_tol=1e-12)


def test_bonus_match_prob_lottomax_full_match_gives_zero():
    cfg = config.LOTTERY_RULES["lottomax"]
    # If we matched all 7 main, 0 of our picks remain in the pool
    p = ev.bonus_match_prob(cfg, tier_main=7)
    assert p == 0.0


def test_bonus_match_prob_dailygrand_independent_of_tier_main():
    cfg = config.LOTTERY_RULES["dailygrand"]  # bonus_max=7, separate pool
    expected = 1.0 / 7.0
    for k in range(cfg["main_count"] + 1):
        assert math.isclose(ev.bonus_match_prob(cfg, tier_main=k), expected,
                            rel_tol=1e-12)


# ---------------------------------------------------------------------------
# compute_ev structural invariants
# ---------------------------------------------------------------------------

def _uniform_params():
    return {k: 1.0 for k in pm.FEATURE_NAMES}


def test_compute_ev_structure_lottomax():
    params = _uniform_params()
    combo = [1, 2, 3, 4, 5, 6, 7]
    result = ev.compute_ev(combo, "lottomax", params,
                           jackpot=50_000_000.0, n_tickets_sold=8_000_000)
    for key in ("ev_gross", "ev_net", "ticket_cost", "p_jackpot",
                "e_other_jackpot_winners", "p_combination_given_params",
                "tier_breakdown"):
        assert key in result
    # ev_net = ev_gross - ticket_cost
    assert math.isclose(result["ev_net"],
                        result["ev_gross"] - result["ticket_cost"],
                        rel_tol=1e-12)
    # ticket_cost matches config
    assert result["ticket_cost"] == config.LOTTERY_RULES["lottomax"]["ticket_cost"]
    # tier_breakdown has one entry per tier defined in PRIZE_TIERS
    assert len(result["tier_breakdown"]) == len(config.PRIZE_TIERS["lottomax"])


def test_compute_ev_uniform_p_combination_is_one_over_binomial():
    """Under uniform popularity params, P(C | params) = 1 / C(main_max, main_count)."""
    params = _uniform_params()
    combo = [3, 12, 18, 24, 31, 40, 47]
    result = ev.compute_ev(combo, "lottomax", params,
                           jackpot=50_000_000.0, n_tickets_sold=8_000_000)
    expected = 1.0 / math.comb(52, 7)
    assert math.isclose(result["p_combination_given_params"], expected,
                        rel_tol=1e-9)


def test_compute_ev_p_jackpot_matches_hypergeometric_for_no_bonus_jackpot():
    """For 649 the jackpot has no bonus requirement: p_jackpot = P(6/6)."""
    params = _uniform_params()
    combo = [1, 2, 3, 4, 5, 6]
    result = ev.compute_ev(combo, "649", params,
                           jackpot=5_000_000.0, n_tickets_sold=4_000_000)
    assert math.isclose(result["p_jackpot"],
                        ev.hypergeometric_prob(49, 6, 6),
                        rel_tol=1e-12)


def test_compute_ev_jackpot_split_scales_with_n_tickets_sold():
    """Doubling n_tickets_sold roughly halves the effective jackpot payout."""
    params = _uniform_params()
    combo = [1, 2, 3, 4, 5, 6]
    a = ev.compute_ev(combo, "649", params, jackpot=5_000_000.0,
                      n_tickets_sold=1_000_000)
    b = ev.compute_ev(combo, "649", params, jackpot=5_000_000.0,
                      n_tickets_sold=2_000_000)
    # More expected other winners -> lower effective payout at the jackpot tier
    assert a["e_other_jackpot_winners"] < b["e_other_jackpot_winners"]
    jackpot_a = next(t for t in a["tier_breakdown"]
                     if t["payout_model"] == "jackpot_pari_mutuel")
    jackpot_b = next(t for t in b["tier_breakdown"]
                     if t["payout_model"] == "jackpot_pari_mutuel")
    assert jackpot_a["effective_payout"] > jackpot_b["effective_payout"]


def test_compute_ev_anti_popular_has_higher_jackpot_ev_than_popular():
    """A low-popularity combination should yield a larger jackpot EV contribution."""
    # Birthday-biased: numbers 1-31 are MORE popular (higher per-number weight)
    params = {"w_birthday": 2.0, "w_lucky7": 1.0,
              "w_round_decade": 1.0, "w_recent_winner": 1.0}
    popular = [1, 2, 3, 4, 5, 6]                     # all birthday numbers
    anti    = [35, 37, 38, 39, 41, 43]               # none in 1-31
    jackpot = 10_000_000.0
    n_sold  = 2_000_000
    r_pop  = ev.compute_ev(popular, "649", params, jackpot, n_sold)
    r_anti = ev.compute_ev(anti,    "649", params, jackpot, n_sold)
    # The anti-popular combination has lower P(C | params) -> fewer expected
    # other jackpot winners -> higher effective payout.
    assert r_anti["p_combination_given_params"] < r_pop["p_combination_given_params"]
    assert r_anti["e_other_jackpot_winners"] < r_pop["e_other_jackpot_winners"]
    anti_jackpot = next(t for t in r_anti["tier_breakdown"]
                        if t["payout_model"] == "jackpot_pari_mutuel")
    pop_jackpot = next(t for t in r_pop["tier_breakdown"]
                       if t["payout_model"] == "jackpot_pari_mutuel")
    assert anti_jackpot["contribution"] > pop_jackpot["contribution"]


def test_compute_ev_dailygrand_has_no_lower_parimutuel_tiers():
    """All Daily Grand non-GRAND tiers are fixed-payout."""
    params = _uniform_params()
    combo = [8, 17, 28, 37, 46]
    result = ev.compute_ev(combo, "dailygrand", params,
                           jackpot=7_000_000.0, n_tickets_sold=500_000)
    # Every tier should be either fixed or (a single) jackpot pari-mutuel
    models = {t["payout_model"] for t in result["tier_breakdown"]}
    assert "lower_pari_mutuel_uniform_approx" not in models


# ---------------------------------------------------------------------------
# best_combination
# ---------------------------------------------------------------------------

def test_best_combination_uniform_params_returns_smallest_indices():
    """When all per-number weights are equal (all=1), argsort stable -> indices 1..k."""
    params = _uniform_params()
    result = ev.best_combination("649", params, jackpot=5_000_000.0,
                                 n_tickets_sold=4_000_000)
    assert result["combination"] == [1, 2, 3, 4, 5, 6]
    assert result["lottery"] == "649"
    assert "ev" in result


def test_best_combination_avoids_high_weight_numbers():
    """With w_birthday > 1 (numbers 1-31 more popular), the min-popularity pick
    should skip all of 1-31 -- we want numbers with the smallest w_i."""
    params = {"w_birthday": 2.0, "w_lucky7": 1.0,
              "w_round_decade": 1.0, "w_recent_winner": 1.0}
    # For 649: main_max=49, main_count=6. Numbers 32-49 have weight 1.0 unless
    # they hit another feature. With only w_birthday active, 32-49 (18 numbers)
    # all tie at 1.0 and stable sort returns the smallest indices = 32..37.
    # Numbers 1-31 have weight 2.0 and should be excluded.
    result = ev.best_combination("649", params, jackpot=5_000_000.0,
                                 n_tickets_sold=4_000_000)
    for n in result["combination"]:
        assert n > 31, f"picked {n} which is in the birthday-biased range"
    assert len(result["combination"]) == 6
    assert len(set(result["combination"])) == 6   # unique


def test_best_combination_returns_sorted_ascending():
    params = _uniform_params()
    for lottery in ["lottomax", "649", "dailygrand"]:
        cfg = config.LOTTERY_RULES[lottery]
        result = ev.best_combination(lottery, params, jackpot=1_000_000.0,
                                     n_tickets_sold=cfg["main_max"] * 10_000)
        combo = result["combination"]
        assert combo == sorted(combo)
        assert len(combo) == cfg["main_count"]
        assert all(1 <= n <= cfg["main_max"] for n in combo)


# ---------------------------------------------------------------------------
# Grand Number recommendation (Daily Grand only)
# ---------------------------------------------------------------------------

def test_pick_grand_number_none_for_lottomax_and_649():
    assert ev.pick_grand_number("lottomax") is None
    assert ev.pick_grand_number("649") is None
    assert ev.pick_grand_number("lottomax", recent_grand=[1, 2, 3]) is None
    assert ev.pick_grand_number("649", recent_grand=[7]) is None


def test_pick_grand_number_default_when_no_history():
    # With no history we default to 1 (avoiding lucky-7 bias)
    assert ev.pick_grand_number("dailygrand") == 1
    assert ev.pick_grand_number("dailygrand", recent_grand=[]) == 1


def test_pick_grand_number_picks_least_recent():
    # History shows 1, 2, 3 most recently drawn (in order oldest-to-newest).
    # Numbers 4, 5, 6, 7 have never appeared -> should pick the smallest unseen = 4.
    g = ev.pick_grand_number("dailygrand", recent_grand=[1, 2, 3])
    assert g == 4


def test_pick_grand_number_prefers_oldest_seen_when_all_appeared():
    # All 7 numbers appear; the oldest (earliest in list) should win.
    # recent_grand is ordered oldest-first; pick_grand_number reverses it.
    # So after reversal: index 0 = most recent = 7 (last in list).
    # Oldest = 1 (first in list). Pick 1.
    g = ev.pick_grand_number("dailygrand", recent_grand=[1, 2, 3, 4, 5, 6, 7])
    assert g == 1


def test_best_combination_dailygrand_includes_grand_number():
    params = _uniform_params()
    result = ev.best_combination(
        "dailygrand", params, jackpot=7_000_000.0,
        n_tickets_sold=500_000,
        recent_grand=[7, 6, 5, 4, 3, 2, 1],
    )
    assert "grand_number" in result
    assert result["grand_number"] == 7  # 7 is the oldest in the reversed order


def test_best_combination_lottomax_grand_number_is_none():
    params = _uniform_params()
    result = ev.best_combination(
        "lottomax", params, jackpot=50_000_000.0, n_tickets_sold=8_000_000,
    )
    assert result["grand_number"] is None


def test_best_combination_649_grand_number_is_none():
    params = _uniform_params()
    result = ev.best_combination(
        "649", params, jackpot=5_000_000.0, n_tickets_sold=4_000_000,
    )
    assert result["grand_number"] is None


def test_best_combination_respects_recent_winning_bias():
    """With w_recent_winner > 1, recent numbers become LESS attractive (higher w)."""
    params = {"w_birthday": 1.0, "w_lucky7": 1.0,
              "w_round_decade": 1.0, "w_recent_winner": 5.0}
    recent = {1, 2, 3}
    result = ev.best_combination("649", params, jackpot=5_000_000.0,
                                 n_tickets_sold=4_000_000,
                                 recent_winning=recent)
    for n in result["combination"]:
        assert n not in recent


def test_best_combination_uses_default_tickets_sold_when_none():
    params = _uniform_params()
    result = ev.best_combination("lottomax", params, jackpot=50_000_000.0,
                                 n_tickets_sold=None)
    assert result["n_tickets_sold"] == config.DEFAULT_TICKETS_SOLD["lottomax"]


# ---------------------------------------------------------------------------
# uniform_random_combination baseline
# ---------------------------------------------------------------------------

def test_uniform_random_combination_is_reproducible_with_seed():
    params = _uniform_params()
    a = ev.uniform_random_combination("649", params, jackpot=5_000_000.0,
                                      n_tickets_sold=1_000_000, seed=123)
    b = ev.uniform_random_combination("649", params, jackpot=5_000_000.0,
                                      n_tickets_sold=1_000_000, seed=123)
    assert a["combination"] == b["combination"]
    assert a["strategy"] == "uniform_random"


def test_uniform_random_combination_shape_and_range():
    params = _uniform_params()
    for lottery in ["lottomax", "649", "dailygrand"]:
        cfg = config.LOTTERY_RULES[lottery]
        r = ev.uniform_random_combination(lottery, params, jackpot=1_000_000.0,
                                          n_tickets_sold=100_000, seed=7)
        combo = r["combination"]
        assert len(combo) == cfg["main_count"]
        assert len(set(combo)) == cfg["main_count"]
        assert all(1 <= n <= cfg["main_max"] for n in combo)
        assert combo == sorted(combo)
