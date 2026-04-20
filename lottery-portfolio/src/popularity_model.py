"""
popularity_model.py -- parametric model of P(player picks combination C).

Problem framing
---------------
Lottery draws are i.i.d. uniform: we cannot predict winning numbers. What we CAN
predict is how *players* choose their ticket numbers -- which is non-uniform
because of birthdays, lucky numbers, and visual patterns on bet slips. By picking
combinations that fewer other players pick, a winning ticket has to split the
jackpot with fewer people, increasing expected $payout conditional on winning.

Model
-----
Per-number weight w_i (i in 1..main_max) is a product of feature indicators:

    w_i = w_birthday ** (i <= 31)
        * w_lucky7   ** (i % 7 == 0)
        * w_round    ** (i % 10 == 0 or i % 10 == 5)
        * w_recent   ** (i in recent_winning_numbers)

Combination weight is the product of its members' weights:

    weight(C) = prod_{i in C} w_i

Normalized probability:

    P(C | params) = weight(C) / Z           Z = e_k(w_1, ..., w_{main_max})

where e_k is the k-th elementary symmetric polynomial of the weights, computed in
O(N*k) time via the Newton recursion. For tier-k match calculations, we decompose
weighted subset sums into winners (subset of W) and losers (subset of non-W), each
computable with its own elementary symmetric polynomial.

Feature priors come from empirical research on lottery player behavior:
- Cook & Clotfelter (1993), "The Peculiar Scale Economies of Lotto"
- DeBoer (1990), "Lotto Sales Stagnation"
- Simon (1999), "An Analysis of the Distribution of Combinations Chosen by UK
  National Lottery Players"
- Skiena (2004), "Calculated Bets" (pp. 84-91 on number preferences)

Fitting
-------
Maximum-likelihood fit to observed per-tier winner counts across historical
draws, using Poisson NLL with Gaussian regularization toward the priors. Optimized
with scipy.optimize.minimize(method='L-BFGS-B') on log-weights to keep positivity.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config


# The features this model supports. Keep <= 5 for <= 100 draws of data.
FEATURE_NAMES = ["w_birthday", "w_lucky7", "w_round_decade", "w_recent_winner"]


# ---------------------------------------------------------------------------
# Per-number weight computation
# ---------------------------------------------------------------------------

def per_number_weights(
    main_max: int,
    params: dict,
    recent_winning: set[int] | None = None,
) -> np.ndarray:
    """Return an array w of shape (main_max,) with w[i-1] = weight for number i."""
    if recent_winning is None:
        recent_winning = set()

    w_birth  = float(params.get("w_birthday",      1.0))
    w_lucky  = float(params.get("w_lucky7",        1.0))
    w_round  = float(params.get("w_round_decade",  1.0))
    w_recent = float(params.get("w_recent_winner", 1.0))

    w = np.ones(main_max, dtype=np.float64)
    for i in range(1, main_max + 1):
        factor = 1.0
        if i <= 31:
            factor *= w_birth
        if i % 7 == 0:
            factor *= w_lucky
        if (i % 10 == 0) or (i % 10 == 5):
            factor *= w_round
        if i in recent_winning:
            factor *= w_recent
        w[i - 1] = factor
    return w


# ---------------------------------------------------------------------------
# Elementary symmetric polynomials (Newton recursion)
# ---------------------------------------------------------------------------

def elementary_symmetric(weights: np.ndarray, k: int) -> np.ndarray:
    """
    Return an array e of shape (k+1,) where e[j] = e_j(weights)
    for j = 0, 1, ..., k.

    Recursion: e_j(w_1..w_{i+1}) = e_j(w_1..w_i) + w_{i+1} * e_{j-1}(w_1..w_i)
    Runs in O(len(weights) * k) time.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if k > len(weights):
        raise ValueError(f"k={k} exceeds weights length {len(weights)}")

    e = np.zeros(k + 1, dtype=np.float64)
    e[0] = 1.0
    for i, w in enumerate(weights):
        # Iterate j from high to low so we use the previous-iteration e[j-1]
        for j in range(min(i + 1, k), 0, -1):
            e[j] += w * e[j - 1]
    return e


def log_partition(
    main_max: int,
    main_count: int,
    params: dict,
    recent_winning: set[int] | None = None,
) -> float:
    """Return log Z = log(e_{main_count}(per_number_weights))."""
    w = per_number_weights(main_max, params, recent_winning)
    e = elementary_symmetric(w, main_count)
    Z = e[main_count]
    return float(np.log(max(Z, 1e-300)))


# ---------------------------------------------------------------------------
# Popularity of a single combination
# ---------------------------------------------------------------------------

def log_popularity(
    combination: list[int] | tuple[int, ...],
    params: dict,
    main_max: int,
    main_count: int,
    recent_winning: set[int] | None = None,
) -> float:
    """Return log P(C | params) for a single combination."""
    if len(combination) != main_count:
        raise ValueError(f"Combination of length {len(combination)}, expected {main_count}")
    w = per_number_weights(main_max, params, recent_winning)
    log_w_C = float(np.sum(np.log(w[np.array(combination) - 1])))
    return log_w_C - log_partition(main_max, main_count, params, recent_winning)


# ---------------------------------------------------------------------------
# Expected winner count per tier
# ---------------------------------------------------------------------------

def expected_tier_winner_fraction(
    winning_numbers: list[int],
    tier_main: int,
    main_max: int,
    main_count: int,
    params: dict,
    recent_winning: set[int] | None = None,
) -> float:
    """
    Return the expected FRACTION of tickets that match exactly `tier_main` of
    the winning_numbers (ignoring bonus).

    P(match exactly m main numbers | uniform-conditional on params)
        = [e_m(w_W)  *  e_{main_count - m}(w_notW)]  /  e_{main_count}(w_all)

    where w_W are the weights of the winning numbers and w_notW are the rest.
    """
    W = np.array(sorted(winning_numbers))
    all_w = per_number_weights(main_max, params, recent_winning)

    # Indices 0-based
    mask = np.zeros(main_max, dtype=bool)
    mask[W - 1] = True
    w_W    = all_w[mask]
    w_notW = all_w[~mask]

    e_all  = elementary_symmetric(all_w,  main_count)[main_count]
    if e_all <= 0:
        return 0.0

    # Need elementary_symmetric of both halves at specific degrees
    e_W_full    = elementary_symmetric(w_W,    min(tier_main,              len(w_W)))
    e_notW_full = elementary_symmetric(w_notW, min(main_count - tier_main, len(w_notW)))

    if tier_main > len(w_W) or (main_count - tier_main) > len(w_notW):
        return 0.0
    numer = e_W_full[tier_main] * e_notW_full[main_count - tier_main]
    return float(numer / e_all)


# ---------------------------------------------------------------------------
# Negative log-likelihood for fitting
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    params:     dict
    nll:        float
    reg:        float
    total:      float
    n_draws:    int
    heldout_nll: float | None
    success:    bool
    message:    str


def _params_from_log_vec(log_vec: np.ndarray) -> dict:
    """Exponentiate log-vector into a params dict keyed by FEATURE_NAMES."""
    return {name: float(np.exp(log_vec[i])) for i, name in enumerate(FEATURE_NAMES)}


def _log_vec_from_params(params: dict) -> np.ndarray:
    return np.array([np.log(params.get(name, 1.0)) for name in FEATURE_NAMES])


def _nll_one_draw(
    winning_numbers: list[int],
    tier_counts: list[tuple[int, int]],   # [(tier_main, n_winners), ...]
    n_tickets_sold: float,
    params: dict,
    main_max: int,
    main_count: int,
    recent_winning: set[int] | None = None,
) -> float:
    """Poisson NLL for a single draw's per-tier winner counts."""
    total = 0.0
    for tier_main, n_winners in tier_counts:
        p = expected_tier_winner_fraction(
            winning_numbers, tier_main, main_max, main_count, params, recent_winning
        )
        lam = max(n_tickets_sold * p, 1e-9)
        # Poisson NLL (dropping constant log(n!) term)
        total += lam - n_winners * np.log(lam)
    return total


def _aggregate_tier_counts(payouts_draw: pd.DataFrame) -> list[tuple[int, int]]:
    """Given all payout rows for one draw, aggregate (tier_main -> total winners)."""
    if payouts_draw.empty:
        return []
    # Sum winners by tier_main (ignore bonus split -- we model main-only here)
    grouped = payouts_draw.groupby("tier_main")["n_winners"].sum().reset_index()
    return [(int(r.tier_main), int(r.n_winners)) for r in grouped.itertuples()]


def _recent_winning_set(draws_df: pd.DataFrame, up_to_index: int, k: int = 5) -> set[int]:
    """Return the set of main numbers from the k draws immediately before up_to_index."""
    if up_to_index <= 0:
        return set()
    start = max(0, up_to_index - k)
    recent = draws_df.iloc[start:up_to_index]
    nums: set[int] = set()
    for _, row in recent.iterrows():
        for col in row.index:
            if col.startswith("n") and col[1:].isdigit():
                nums.add(int(row[col]))
    return nums


def _total_objective(
    log_vec: np.ndarray,
    draws_df: pd.DataFrame,
    payouts_df: pd.DataFrame,
    lottery_cfg: dict,
    n_tickets_sold: float,
    prior_log_vec: np.ndarray,
    prior_sigma: float,
) -> float:
    """Regularized Poisson NLL over all draws with tier data."""
    params = _params_from_log_vec(log_vec)
    main_max   = lottery_cfg["main_max"]
    main_count = lottery_cfg["main_count"]

    total_nll = 0.0
    n_draws_used = 0

    # Group payouts by date for lookup
    payouts_by_date = dict(tuple(payouts_df.groupby("date"))) if not payouts_df.empty else {}

    main_cols = [f"n{i}" for i in range(1, main_count + 1)]

    for idx, (_, draw_row) in enumerate(draws_df.iterrows()):
        date = str(draw_row["date"])
        if date not in payouts_by_date:
            continue
        winning = [int(draw_row[c]) for c in main_cols]
        tier_counts = _aggregate_tier_counts(payouts_by_date[date])
        if not tier_counts:
            continue
        recent = _recent_winning_set(draws_df, idx, k=5)
        total_nll += _nll_one_draw(
            winning, tier_counts, n_tickets_sold,
            params, main_max, main_count, recent,
        )
        n_draws_used += 1

    # Gaussian prior regularization
    diff = log_vec - prior_log_vec
    reg = 0.5 * float(np.sum(diff * diff)) / (prior_sigma * prior_sigma)

    # Protect against zero-data edge case
    total_nll = total_nll if n_draws_used > 0 else 0.0
    return total_nll + reg


# ---------------------------------------------------------------------------
# Public fit function
# ---------------------------------------------------------------------------

def fit(
    draws_df: pd.DataFrame,
    payouts_df: pd.DataFrame,
    lottery: str,
    n_tickets_sold: float | None = None,
    heldout_fraction: float | None = None,
) -> FitResult:
    """
    Fit popularity-model params by maximum-likelihood matching of observed per-tier
    winner counts.

    Parameters
    ----------
    draws_df         : output of data_loader.load_draws(lottery)
    payouts_df       : output of data_loader.load_payouts(lottery)
    lottery          : 'lottomax' | '649' | 'dailygrand'
    n_tickets_sold   : fallback default from config if None
    heldout_fraction : last fraction of draws reserved for NLL eval (default config.HELDOUT_FRACTION)
    """
    lottery_cfg = config.LOTTERY_RULES[lottery]
    if n_tickets_sold is None:
        n_tickets_sold = config.DEFAULT_TICKETS_SOLD[lottery]
    if heldout_fraction is None:
        heldout_fraction = config.HELDOUT_FRACTION

    priors = {k: config.POPULARITY_PRIORS[k] for k in FEATURE_NAMES}
    prior_log_vec = _log_vec_from_params(priors)
    prior_sigma   = config.POPULARITY_PRIOR_SIGMA

    # Need >= 10 draws with payout data to fit reliably
    n_draws = len(draws_df)
    n_train = max(1, int(n_draws * (1 - heldout_fraction)))
    train_df = draws_df.iloc[:n_train].reset_index(drop=True)
    heldout_df = draws_df.iloc[n_train:].reset_index(drop=True)

    train_dates = set(train_df["date"].astype(str))
    heldout_dates = set(heldout_df["date"].astype(str))
    train_payouts   = payouts_df[payouts_df["date"].astype(str).isin(train_dates)]
    heldout_payouts = payouts_df[payouts_df["date"].astype(str).isin(heldout_dates)]

    if train_payouts.empty:
        # Not enough data -- return priors unchanged with success=False
        return FitResult(
            params=priors, nll=0.0, reg=0.0, total=0.0,
            n_draws=0, heldout_nll=None, success=False,
            message="No payout rows in training slice -- returning priors",
        )

    x0 = prior_log_vec.copy()
    bounds = [(-1.0, 1.0)] * len(FEATURE_NAMES)   # weights in [e^-1, e] ~= [0.37, 2.72]

    result = minimize(
        _total_objective, x0,
        args=(train_df, train_payouts, lottery_cfg, n_tickets_sold,
              prior_log_vec, prior_sigma),
        method="L-BFGS-B", bounds=bounds,
    )

    fitted_params = _params_from_log_vec(result.x)
    nll_val = _total_objective(
        result.x, train_df, train_payouts, lottery_cfg, n_tickets_sold,
        prior_log_vec, prior_sigma,
    )
    diff = result.x - prior_log_vec
    reg_val = 0.5 * float(np.sum(diff * diff)) / (prior_sigma ** 2)

    heldout_nll = None
    if not heldout_df.empty and not heldout_payouts.empty:
        heldout_nll = _total_objective(
            result.x, heldout_df, heldout_payouts, lottery_cfg, n_tickets_sold,
            prior_log_vec, 1e6,   # effectively remove regularization on held-out
        )

    return FitResult(
        params=fitted_params,
        nll=float(nll_val - reg_val),
        reg=float(reg_val),
        total=float(nll_val),
        n_draws=len(train_df),
        heldout_nll=float(heldout_nll) if heldout_nll is not None else None,
        success=bool(result.success),
        message=str(result.message),
    )
