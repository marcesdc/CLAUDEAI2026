"""
ev_optimizer.py -- choose the combination that maximizes expected payout per ticket.

Approach
--------
For a lottery with C(main_max, main_count) possible combinations, exhaustive
enumeration is infeasible for LottoMax (~100M) and tight for 6/49 (~14M). We
instead use the closed-form minimizer of per-combination popularity: under the
factored popularity model in popularity_model.py, the combination with MINIMUM
P(C | params) is simply the main_count numbers with the smallest per-number
weight w_i. That single sort IS the optimum for the jackpot-split-risk objective.

Since fixed-payout tiers (e.g. 4/- at $20 in LottoMax) contribute a constant
expected value regardless of C, and pari-mutuel tiers below the jackpot have
much smaller split-risk sensitivity to C, minimizing jackpot popularity
captures the dominant EV-shift signal.

For lower pari-mutuel tiers we model split-risk using a UNIFORM-player
approximation -- the expected number of other winners at tier k equals
n_tickets_sold * hypergeometric_prob(tier_k). This under-estimates the true
bias-induced popularity shift on those tiers, but (a) those tiers contribute a
much smaller share of jackpot-pool dollars and (b) the direction of the shift
aligns with the jackpot shift, so the minimum-popularity combination is still
a very good choice.

Bonus probability
-----------------
LottoMax / 6/49: the bonus ball is drawn from the remaining balls after the
main numbers, so P(bonus matches | we matched k main) = (main_count - k) /
(main_max - main_count).

Daily Grand: the Grand Number is drawn from its own separate pool (1-7),
independent of the main pool. P(bonus matches) = 1 / bonus_max regardless of
main match count.
"""

from math import comb
from typing import Optional

import numpy as np

import config
from src import popularity_model as pm


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------

def hypergeometric_prob(main_max: int, main_count: int, k: int) -> float:
    """P(match exactly k of main_count | uniform random draw of main_count from main_max)."""
    if k < 0 or k > main_count or k > main_max:
        return 0.0
    return (comb(main_count, k) * comb(main_max - main_count, main_count - k)
            / comb(main_max, main_count))


def bonus_match_prob(lottery_cfg: dict, tier_main: int) -> float:
    """P(bonus ball matches one of our remaining picks | we matched tier_main main)."""
    main_count = lottery_cfg["main_count"]
    main_max   = lottery_cfg["main_max"]
    bonus_max  = lottery_cfg["bonus_max"]

    # Daily Grand: bonus is independent, drawn from its own pool
    if lottery_cfg["bonus_col"] != "bonus":
        return 1.0 / bonus_max

    # LottoMax / 6/49: bonus is drawn from the remaining (main_max - main_count) balls
    remaining_pool = main_max - main_count
    if remaining_pool <= 0:
        return 0.0
    # We matched tier_main of our main_count numbers -> (main_count - tier_main) of our
    # picks are in the remaining pool. Each is equally likely to be the bonus draw.
    return (main_count - tier_main) / remaining_pool


# ---------------------------------------------------------------------------
# Pool allocation per lottery (fraction of total jackpot -> tier pool)
# ---------------------------------------------------------------------------
# These approximate the OLG Canadian Lotteries published prize-pool allocations.
# Only used for NON-jackpot pari-mutuel tiers. Jackpot itself uses the raw jackpot.

_POOL_FRACTION = {
    "lottomax": {
        # (tier_main, tier_bonus_required) -> fraction of the total prize pool for tier k
        (7, None):  1.00,   # jackpot -> uses jackpot value directly
        (6, True):  0.04,   # 6+Bonus pool ~= 4% of pools fund
        (6, False): 0.04,
        (5, None):  0.035,
    },
    "649": {
        (6, None):  1.00,
        (5, True):  0.05,
        (5, False): 0.05,
    },
    "dailygrand": {
        # Daily Grand's top tier is fixed-cash ($7M cash or annuity); not pari-mutuel
    },
}


def _is_parimutuel(lottery: str, tier_main: int, tier_bonus, payout) -> bool:
    """Return True iff the tier pays pari-mutuel (split among winners from a pool)."""
    return payout is None


def _pool_fraction_for_tier(lottery: str, tier_main: int, tier_bonus) -> float:
    """Return the fraction of the reported jackpot allocated to this tier's pool."""
    table = _POOL_FRACTION.get(lottery, {})
    key = (tier_main, tier_bonus) if (tier_main, tier_bonus) in table else (tier_main, None)
    return table.get(key, 0.0)


# ---------------------------------------------------------------------------
# Main EV computation
# ---------------------------------------------------------------------------

def compute_ev(
    combination: list[int],
    lottery: str,
    popularity_params: dict,
    jackpot: float,
    n_tickets_sold: float,
    recent_winning: Optional[set[int]] = None,
) -> dict:
    """
    Compute expected dollar payout for a single ticket holding `combination`.

    Returns dict:
        ev_gross    expected dollars before subtracting ticket cost
        ev_net      ev_gross - ticket_cost
        p_jackpot   hypergeometric probability of hitting the jackpot
        e_other_jackpot_winners  expected other winners at jackpot tier
        tier_breakdown  list of per-tier contribution dicts
    """
    cfg = config.LOTTERY_RULES[lottery]
    tiers = config.PRIZE_TIERS[lottery]
    main_max   = cfg["main_max"]
    main_count = cfg["main_count"]
    cost       = cfg["ticket_cost"]

    # Pre-compute P(C | params) for the jackpot split
    log_p_C = pm.log_popularity(combination, popularity_params,
                                main_max, main_count, recent_winning)
    p_C = float(np.exp(log_p_C))

    ev_gross = 0.0
    tier_breakdown = []
    e_other_jackpot = 0.0
    p_jackpot = 0.0

    for (tier_main, tier_bonus, fixed_payout) in tiers:
        p_main = hypergeometric_prob(main_max, main_count, tier_main)
        if tier_bonus is None:
            p_tier = p_main
        else:
            p_b = bonus_match_prob(cfg, tier_main)
            p_tier = p_main * (p_b if tier_bonus else (1.0 - p_b))

        if fixed_payout is not None:
            # Fixed payout tier
            # Daily Grand GRAND PRIZE: model as fixed-cash (DAILYGRAND_GRAND_CASH_VALUE)
            if (lottery == "dailygrand" and tier_main == main_count
                    and tier_bonus is True and fixed_payout is None):
                payout = config.DAILYGRAND_GRAND_CASH_VALUE
            else:
                payout = fixed_payout
            contribution = p_tier * payout
            tier_breakdown.append({
                "tier_main": tier_main, "tier_bonus": tier_bonus,
                "p_tier": p_tier, "payout_model": "fixed",
                "payout": payout, "contribution": contribution,
            })
            ev_gross += contribution
            continue

        # Pari-mutuel tier
        if tier_main == main_count and (tier_bonus is None or tier_bonus is True):
            # JACKPOT -- use exact popularity-based split risk
            pool = jackpot
            expected_other = n_tickets_sold * p_C
            effective_payout = pool / (1.0 + expected_other)
            contribution = p_tier * effective_payout
            e_other_jackpot = expected_other
            p_jackpot = p_tier
            tier_breakdown.append({
                "tier_main": tier_main, "tier_bonus": tier_bonus,
                "p_tier": p_tier, "payout_model": "jackpot_pari_mutuel",
                "pool": pool, "expected_other_winners": expected_other,
                "effective_payout": effective_payout, "contribution": contribution,
            })
            ev_gross += contribution
            continue

        # Lower pari-mutuel tier -- approximate split with uniform-player model
        pool_frac = _pool_fraction_for_tier(lottery, tier_main, tier_bonus)
        if pool_frac <= 0.0 or p_tier <= 0.0:
            tier_breakdown.append({
                "tier_main": tier_main, "tier_bonus": tier_bonus,
                "p_tier": p_tier, "payout_model": "pari_mutuel_unmodeled",
                "contribution": 0.0,
            })
            continue
        pool = pool_frac * jackpot
        expected_other = max(n_tickets_sold * p_tier, 0.0)
        effective_payout = pool / (1.0 + expected_other)
        contribution = p_tier * effective_payout
        tier_breakdown.append({
            "tier_main": tier_main, "tier_bonus": tier_bonus,
            "p_tier": p_tier, "payout_model": "lower_pari_mutuel_uniform_approx",
            "pool": pool, "expected_other_winners": expected_other,
            "effective_payout": effective_payout, "contribution": contribution,
        })
        ev_gross += contribution

    return {
        "ev_gross": ev_gross,
        "ev_net":   ev_gross - cost,
        "ticket_cost": cost,
        "p_jackpot": p_jackpot,
        "e_other_jackpot_winners": e_other_jackpot,
        "p_combination_given_params": p_C,
        "tier_breakdown": tier_breakdown,
    }


# ---------------------------------------------------------------------------
# Best combination (argmin-popularity closed form + EV annotation)
# ---------------------------------------------------------------------------

def pick_grand_number(
    lottery: str,
    recent_grand: Optional[list[int]] = None,
) -> Optional[int]:
    """Return a recommended Grand Number (1..bonus_max) for games where the
    player selects the bonus themselves. Returns None for LottoMax/6/49 because
    the bonus there is drawn by the machine and never appears on the ticket.

    Daily Grand: the Grand Number (1-7) IS player-picked. We have no popularity
    model for a 7-element pool (too small for reliable fitting on short histories),
    so we use a deterministic anti-chase heuristic: prefer the number that was
    drawn least recently in ``recent_grand``. Ties go to the smallest integer,
    which tends to avoid the lucky-7 bias players commonly apply. If no history
    is provided, defaults to 1.
    """
    cfg = config.LOTTERY_RULES[lottery]
    if cfg["bonus_col"] == "bonus":
        return None
    bonus_max = cfg["bonus_max"]
    if not recent_grand:
        return 1
    # recent_grand is ordered oldest-first; reverse so index 0 = most recent.
    last_seen: dict[int, int] = {}
    for i, g in enumerate(reversed(recent_grand)):
        g_int = int(g)
        if 1 <= g_int <= bonus_max and g_int not in last_seen:
            last_seen[g_int] = i
    # Score: never-seen beats any seen (use large sentinel); seen earlier beats seen later.
    def score(n: int) -> tuple[int, int]:
        seen_recency = last_seen.get(n, bonus_max * 1000)  # unseen -> large
        return (-seen_recency, n)
    return min(range(1, bonus_max + 1), key=score)


def best_combination(
    lottery: str,
    popularity_params: dict,
    jackpot: float,
    n_tickets_sold: Optional[float] = None,
    recent_winning: Optional[set[int]] = None,
    recent_grand: Optional[list[int]] = None,
) -> dict:
    """
    Return the min-popularity combination (one line) plus its EV estimate.

    Closed-form result: under the factored popularity model, P(C | params) is
    proportional to product of per-number weights. The argmin is achieved by
    picking the main_count numbers with the smallest per-number weights. Ties
    are broken by the natural ascending order of the number index.

    For lotteries where the player picks a bonus (Daily Grand only), also
    recommends a Grand Number via ``pick_grand_number``. LottoMax and 6/49
    return ``grand_number: None`` because the bonus there is machine-drawn.
    """
    cfg = config.LOTTERY_RULES[lottery]
    main_max   = cfg["main_max"]
    main_count = cfg["main_count"]

    if n_tickets_sold is None:
        n_tickets_sold = config.DEFAULT_TICKETS_SOLD[lottery]

    w = pm.per_number_weights(main_max, popularity_params, recent_winning)
    # Stable sort: argsort breaks ties in index order (ascending number)
    order = np.argsort(w, kind="stable")
    chosen_indices = sorted(int(i + 1) for i in order[:main_count])

    ev = compute_ev(
        chosen_indices, lottery, popularity_params,
        jackpot, n_tickets_sold, recent_winning,
    )

    grand = pick_grand_number(lottery, recent_grand)

    return {
        "lottery":        lottery,
        "combination":    chosen_indices,
        "grand_number":   grand,
        "popularity_params": popularity_params,
        "jackpot":        jackpot,
        "n_tickets_sold": n_tickets_sold,
        "recent_winning": sorted(recent_winning) if recent_winning else [],
        "ev":             ev,
    }


# ---------------------------------------------------------------------------
# Baselines for comparison
# ---------------------------------------------------------------------------

def uniform_random_combination(
    lottery: str,
    popularity_params: dict,
    jackpot: float,
    n_tickets_sold: Optional[float] = None,
    recent_winning: Optional[set[int]] = None,
    seed: Optional[int] = None,
) -> dict:
    """A uniform-random combination + its EV. Used as a baseline for backtest."""
    cfg = config.LOTTERY_RULES[lottery]
    if n_tickets_sold is None:
        n_tickets_sold = config.DEFAULT_TICKETS_SOLD[lottery]

    rng = np.random.default_rng(seed)
    pool = np.arange(1, cfg["main_max"] + 1)
    chosen = sorted(int(x) for x in rng.choice(pool, size=cfg["main_count"], replace=False))

    ev = compute_ev(
        chosen, lottery, popularity_params,
        jackpot, n_tickets_sold, recent_winning,
    )
    return {
        "lottery": lottery,
        "combination": chosen,
        "strategy": "uniform_random",
        "ev": ev,
    }
