"""Unit tests for src/popularity_model.py."""

import math

import numpy as np
import pytest

from src import popularity_model as pm


# ---------------------------------------------------------------------------
# Elementary symmetric polynomial
# ---------------------------------------------------------------------------

def test_elementary_symmetric_known_values():
    w = np.array([1.0, 2.0, 3.0, 4.0])
    e = pm.elementary_symmetric(w, 4)
    assert math.isclose(e[0], 1.0)
    assert math.isclose(e[1], 1 + 2 + 3 + 4)
    assert math.isclose(e[2], 1*2 + 1*3 + 1*4 + 2*3 + 2*4 + 3*4)
    assert math.isclose(e[3], 1*2*3 + 1*2*4 + 1*3*4 + 2*3*4)
    assert math.isclose(e[4], 1*2*3*4)


def test_elementary_symmetric_uniform_matches_binomial():
    """e_k([1,1,...,1]) = C(n, k)."""
    n = 20
    w = np.ones(n)
    for k in range(n + 1):
        expected = math.comb(n, k)
        e = pm.elementary_symmetric(w, k)
        assert math.isclose(e[k], expected), f"e_{k}({n}) = {e[k]} vs {expected}"


def test_elementary_symmetric_rejects_bad_k():
    w = np.ones(5)
    with pytest.raises(ValueError):
        pm.elementary_symmetric(w, -1)
    with pytest.raises(ValueError):
        pm.elementary_symmetric(w, 6)


# ---------------------------------------------------------------------------
# Per-number weights
# ---------------------------------------------------------------------------

def test_per_number_weights_uniform_params():
    w = pm.per_number_weights(50, {k: 1.0 for k in pm.FEATURE_NAMES})
    assert np.allclose(w, 1.0)


def test_per_number_weights_birthday_bias_applied_only_to_1_31():
    params = {"w_birthday": 1.5, "w_lucky7": 1.0, "w_round_decade": 1.0, "w_recent_winner": 1.0}
    w = pm.per_number_weights(50, params)
    # Number 1, 31: should be multiplied by 1.5 (for birthday)
    # Number 32: not birthday
    # But #7, #14, #21, #28 are also lucky7 -- we set that to 1.0 so no effect
    # #5, #10, #15, #20, #25, #30 are round_decade -- we set that to 1.0 so no effect
    # So: numbers 1-6, 8, 9, 11-13, 16-19, 22-24, 26, 27, 29, 31 should be 1.5
    # But #30 is in [1,31] AND round_decade -> 1.5 * 1.0 = 1.5 (round=1)
    # #31 is just birthday -> 1.5
    # #32 is neither -> 1.0
    assert w[0]  == 1.5   # number 1
    assert w[30] == 1.5   # number 31
    assert w[31] == 1.0   # number 32


def test_per_number_weights_compound_features():
    # Number 7 is lucky7, number 35 is lucky7 AND ends in 5
    params = {"w_birthday": 1.0, "w_lucky7": 2.0, "w_round_decade": 3.0, "w_recent_winner": 1.0}
    w = pm.per_number_weights(50, params)
    assert w[6]  == 2.0         # 7: only lucky7
    assert w[34] == 2.0 * 3.0   # 35: lucky7 AND round (ends in 5)


def test_per_number_weights_recent_bonus():
    params = {"w_birthday": 1.0, "w_lucky7": 1.0, "w_round_decade": 1.0, "w_recent_winner": 2.5}
    recent = {42, 48}
    w = pm.per_number_weights(50, params, recent_winning=recent)
    assert w[41] == 2.5
    assert w[47] == 2.5
    assert w[0]  == 1.0


# ---------------------------------------------------------------------------
# Partition and probability
# ---------------------------------------------------------------------------

def test_log_partition_uniform_matches_log_binomial():
    """When w_i == 1 for all i, Z = C(main_max, main_count)."""
    params = {k: 1.0 for k in pm.FEATURE_NAMES}
    for main_max, main_count in [(49, 6), (50, 7), (49, 5)]:
        logZ = pm.log_partition(main_max, main_count, params)
        assert math.isclose(logZ, math.log(math.comb(main_max, main_count)), rel_tol=1e-9)


def test_log_popularity_uniform_is_neg_log_binomial():
    params = {k: 1.0 for k in pm.FEATURE_NAMES}
    C = (1, 2, 3, 4, 5, 6)
    lp = pm.log_popularity(C, params, main_max=49, main_count=6)
    expected = -math.log(math.comb(49, 6))
    assert math.isclose(lp, expected, rel_tol=1e-9)


def test_log_popularity_biased_favors_biased_numbers():
    params = {"w_birthday": 2.0, "w_lucky7": 1.0, "w_round_decade": 1.0, "w_recent_winner": 1.0}
    C_low  = (1, 2, 3, 4, 5, 6)     # all birthday numbers
    C_high = (40, 41, 42, 43, 44, 45)  # no birthday numbers
    lp_low  = pm.log_popularity(C_low,  params, 49, 6)
    lp_high = pm.log_popularity(C_high, params, 49, 6)
    assert lp_low > lp_high


# ---------------------------------------------------------------------------
# Expected tier fraction
# ---------------------------------------------------------------------------

def test_expected_tier_fraction_uniform_matches_hypergeometric():
    """Under uniform params, P(match m of main_count) = hypergeometric."""
    params = {k: 1.0 for k in pm.FEATURE_NAMES}
    main_max, main_count = 49, 6
    W = [1, 2, 3, 4, 5, 6]
    for m in range(main_count + 1):
        p = pm.expected_tier_winner_fraction(W, m, main_max, main_count, params)
        expected = (math.comb(main_count, m) * math.comb(main_max - main_count, main_count - m)
                    / math.comb(main_max, main_count))
        assert math.isclose(p, expected, rel_tol=1e-9), f"m={m}: {p} vs {expected}"


def test_expected_tier_fraction_sums_to_one_under_uniform():
    params = {k: 1.0 for k in pm.FEATURE_NAMES}
    main_max, main_count = 49, 6
    W = list(range(1, main_count + 1))
    total = sum(
        pm.expected_tier_winner_fraction(W, m, main_max, main_count, params)
        for m in range(main_count + 1)
    )
    assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_expected_tier_fraction_sums_to_one_under_biased():
    """Even with non-uniform params, tier fractions over m must sum to 1."""
    params = {"w_birthday": 1.5, "w_lucky7": 1.2, "w_round_decade": 0.9, "w_recent_winner": 1.3}
    main_max, main_count = 49, 6
    W = [2, 7, 15, 22, 30, 42]
    total = sum(
        pm.expected_tier_winner_fraction(W, m, main_max, main_count, params)
        for m in range(main_count + 1)
    )
    assert math.isclose(total, 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def test_fit_uniform_data_recovers_priors_approximately(df_649, payouts_649_df):
    """With synthetic roughly-uniform winner counts, fit should stay near priors."""
    result = pm.fit(df_649, payouts_649_df, "649", n_tickets_sold=100_000)
    assert result.success
    # All fitted weights should be within the bounds set for L-BFGS-B (roughly [0.37, 2.72])
    for k in pm.FEATURE_NAMES:
        assert 0.3 < result.params[k] < 3.0
    assert result.n_draws > 0


def test_fit_handles_no_payout_data(df_649):
    """If no payouts match training draws, fit returns priors with success=False."""
    import pandas as pd
    empty = pd.DataFrame(columns=[
        "date", "lottery", "jackpot", "tier_main", "tier_bonus",
        "n_winners", "payout_per_winner",
    ])
    result = pm.fit(df_649, empty, "649", n_tickets_sold=100_000)
    assert not result.success
    for k in pm.FEATURE_NAMES:
        assert result.params[k] == pytest.approx(
            __import__("config").POPULARITY_PRIORS[k]
        )


def test_fit_produces_heldout_nll_when_data_allows(df_649, payouts_649_df):
    result = pm.fit(df_649, payouts_649_df, "649", n_tickets_sold=100_000,
                    heldout_fraction=0.3)
    # We have 60 synthetic draws -> 18 held-out. Expect a held-out NLL.
    assert result.heldout_nll is not None
    assert result.heldout_nll >= 0 or np.isfinite(result.heldout_nll)


# ---------------------------------------------------------------------------
# Recent-winning set helper
# ---------------------------------------------------------------------------

def test_recent_winning_set_respects_up_to_index(df_649):
    recent = pm._recent_winning_set(df_649, up_to_index=5, k=3)
    # Should contain main numbers from draws 2, 3, 4 (indices)
    assert isinstance(recent, set)
    assert len(recent) > 0
    # At index 0 -> empty set
    assert pm._recent_winning_set(df_649, up_to_index=0, k=5) == set()
