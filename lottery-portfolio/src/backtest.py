"""
backtest.py -- rolling-origin realized-EV backtest.

For each draw t >= min_train:
    1. Fit the popularity model on draws[:t] + payouts up to t.
    2. Recommend best_combination for draw t using those params
       (closed-form min-popularity).
    3. Observe the actual draw at index t -- compute REALIZED dollar payout:
           realized = sum over tier:
               I[|C intersect W| == tier_main]
               * P(bonus_match | tier_main)
               * realized_payout_per_winner_at_tier
       where payout_per_winner is taken from payouts_df (if available) for
       pari-mutuel tiers -- jackpot / (n_winners_actual + 1) so our
       hypothetical ticket counts as an additional splitter. Fixed tiers
       use config.PRIZE_TIERS payouts directly.
    4. Do the same for a uniform_random baseline (seeded per-draw for
       reproducibility).
    5. Accumulate a DataFrame and compute paired statistics.

Why "expected" realized (soft over bonus-match coin flip)?
    The bonus match probability given k main matches is a deterministic
    fraction (main_count - k) / (main_max - main_count) for lottomax/649 and
    1/bonus_max for dailygrand. Using the soft probability yields a lower-
    variance estimator than sampling a single Bernoulli per draw, and keeps
    the backtest stable on our small (~100 draws/year/lottery) datasets.

Statistical test
----------------
Paired t-test on (realized_A - realized_B) across draws. H0: mean diff == 0.
Two-sided p-value reported. Because each draw is a single observation, the
significance threshold is informative only once we have several months of
real payout data.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import config
from src import data_loader, popularity_model as pm, ev_optimizer as ev


# ---------------------------------------------------------------------------
# Realized payout for a single ticket on a single draw
# ---------------------------------------------------------------------------

def _winning_numbers(row: pd.Series, main_count: int) -> list[int]:
    return [int(row[f"n{i}"]) for i in range(1, main_count + 1)]


def _payouts_for_date(payouts_df: pd.DataFrame, date: str) -> pd.DataFrame:
    if payouts_df.empty:
        return payouts_df
    return payouts_df[payouts_df["date"] == date]


def _realized_payout_per_winner(
    lottery: str, tier_main: int, tier_bonus, fixed_payout,
    payouts_for_draw: pd.DataFrame, jackpot: float,
) -> float:
    """Return the dollars-per-winner realized at this tier for this draw.

    Fixed tiers: use the config payout directly.
    Pari-mutuel tiers: observed pool_fraction * jackpot / (n_winners + 1).
        The +1 represents OUR hypothetical ticket being an additional splitter.
        If n_winners is unknown (missing row), we fall back to the model-based
        uniform-player approximation from ev_optimizer (so the backtest is not
        dominated by missing-data zeros).
    """
    if fixed_payout is not None:
        return float(fixed_payout)

    # Pari-mutuel tier
    pool_frac = ev._pool_fraction_for_tier(lottery, tier_main, tier_bonus)
    if pool_frac <= 0.0:
        return 0.0
    pool = pool_frac * float(jackpot) if tier_main != config.LOTTERY_RULES[lottery]["main_count"] \
        else float(jackpot)

    # Look up observed n_winners for this tier
    if not payouts_for_draw.empty:
        tier_bonus_str = "-" if tier_bonus is None else ("Y" if tier_bonus else "N")
        mask = (
            (payouts_for_draw["tier_main"] == tier_main)
            & (payouts_for_draw["tier_bonus"] == tier_bonus_str)
        )
        rows = payouts_for_draw[mask]
        if not rows.empty:
            n_winners = int(rows.iloc[0]["n_winners"])
            return pool / (n_winners + 1)

    # Fallback: uniform-player split estimate (same approximation as compute_ev)
    n_sold = config.DEFAULT_TICKETS_SOLD[lottery]
    p_tier = (ev.hypergeometric_prob(
        config.LOTTERY_RULES[lottery]["main_max"],
        config.LOTTERY_RULES[lottery]["main_count"],
        tier_main,
    ))
    expected_other = max(n_sold * p_tier, 0.0)
    return pool / (1.0 + expected_other)


def realized_payout(
    lottery: str,
    combination: list[int],
    draw_row: pd.Series,
    payouts_df: pd.DataFrame,
) -> float:
    """Expected realized dollars won by `combination` against a single historical draw.

    Soft over the bonus-match probability (see module docstring).
    """
    cfg = config.LOTTERY_RULES[lottery]
    main_count = cfg["main_count"]

    W = set(_winning_numbers(draw_row, main_count))
    k = len(set(combination) & W)

    payouts_for_draw = _payouts_for_date(payouts_df, str(draw_row["date"]))
    # Jackpot value: prefer observed; else fall back to a conservative default
    if not payouts_for_draw.empty and "jackpot" in payouts_for_draw.columns:
        jackpot = float(payouts_for_draw.iloc[0]["jackpot"])
    else:
        jackpot = 0.0  # Without jackpot info, pari-mutuel contribution is zero

    total = 0.0
    for (tier_main, tier_bonus, fixed_payout) in config.PRIZE_TIERS[lottery]:
        if k != tier_main:
            continue
        # Probability we also matched the bonus (given tier_main)
        if tier_bonus is None:
            p_b_contrib = 1.0
        else:
            p_b = ev.bonus_match_prob(cfg, tier_main)
            p_b_contrib = p_b if tier_bonus else (1.0 - p_b)

        payout = _realized_payout_per_winner(
            lottery, tier_main, tier_bonus, fixed_payout,
            payouts_for_draw, jackpot,
        )
        total += p_b_contrib * payout
    return total


# ---------------------------------------------------------------------------
# Rolling-origin loop
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    lottery: str
    per_draw: pd.DataFrame                # columns: date, realized_A, realized_B, diff, k_A, k_B
    mean_A: float = 0.0
    mean_B: float = 0.0
    mean_diff: float = 0.0
    std_diff: float = 0.0
    t_stat: float = 0.0
    p_value: float = 1.0
    n_draws: int = 0
    ticket_cost: float = 0.0
    messages: list[str] = field(default_factory=list)


def _paired_t_test(diffs: np.ndarray) -> tuple[float, float]:
    """Two-sided paired t-test. Returns (t_stat, p_value)."""
    from scipy import stats
    if len(diffs) < 2 or np.std(diffs, ddof=1) == 0.0:
        return 0.0, 1.0
    t_stat, p_value = stats.ttest_rel(diffs, np.zeros_like(diffs))
    return float(t_stat), float(p_value)


def run(
    lottery: str,
    min_train: int = 20,
    refit_every: int = 5,
    seed: int = 42,
    save_csv: bool = True,
) -> int:
    """Run rolling-origin backtest. Returns CLI exit code (0 success, 1 error).

    Compares strategy A = closed-form min-popularity (anti-popular) vs
    strategy B = uniform_random baseline. One ticket per draw per lottery.
    """
    draws   = data_loader.load_draws(lottery)
    payouts = data_loader.load_payouts(lottery)
    cfg     = config.LOTTERY_RULES[lottery]

    print("=" * 64)
    print(f"BACKTEST -- {cfg['name']}")
    print("=" * 64)
    print(f"  draws loaded:    {len(draws)}")
    print(f"  payouts loaded:  {len(payouts)}")
    print(f"  min_train:       {min_train}")
    print(f"  refit_every:     {refit_every}")
    print(f"  ticket cost:     ${cfg['ticket_cost']:.2f}")

    if draws.empty:
        print("[backtest] ERROR -- no draws data. Load historical data first.")
        return 1
    if len(draws) <= min_train:
        print(f"[backtest] ERROR -- need > {min_train} draws; have {len(draws)}.")
        return 1

    rng = np.random.default_rng(seed)
    records = []
    messages = []

    last_params = {k: float(v) for k, v in config.POPULARITY_PRIORS.items()}

    for i in range(min_train, len(draws)):
        if (i - min_train) % refit_every == 0:
            # Re-fit popularity on draws[:i] using matching payouts
            dates_so_far = set(draws.iloc[:i]["date"].astype(str))
            past_payouts = payouts[payouts["date"].astype(str).isin(dates_so_far)]
            try:
                result = pm.fit(draws.iloc[:i], past_payouts, lottery,
                                n_tickets_sold=config.DEFAULT_TICKETS_SOLD[lottery],
                                heldout_fraction=0.0)
                last_params = result.params
            except Exception as e:
                messages.append(f"fit at i={i} failed: {e!r} -- kept priors")

        recent = pm._recent_winning_set(draws, up_to_index=i, k=5)

        # Strategy A: closed-form min-popularity
        combo_A = ev.best_combination(
            lottery, last_params,
            jackpot=0.0,                 # jackpot unused for selection
            n_tickets_sold=config.DEFAULT_TICKETS_SOLD[lottery],
            recent_winning=recent,
        )["combination"]

        # Strategy B: uniform random baseline (seeded per draw for reproducibility)
        combo_B = sorted(int(x) for x in
                         rng.choice(np.arange(1, cfg["main_max"] + 1),
                                    size=cfg["main_count"], replace=False))

        draw_row = draws.iloc[i]
        realized_A = realized_payout(lottery, combo_A, draw_row, payouts)
        realized_B = realized_payout(lottery, combo_B, draw_row, payouts)

        W = set(int(draw_row[f"n{j}"]) for j in range(1, cfg["main_count"] + 1))
        records.append({
            "date":       str(draw_row["date"]),
            "i":          i,
            "combo_A":    " ".join(str(n) for n in combo_A),
            "combo_B":    " ".join(str(n) for n in combo_B),
            "k_A":        len(set(combo_A) & W),
            "k_B":        len(set(combo_B) & W),
            "realized_A": realized_A,
            "realized_B": realized_B,
            "diff":       realized_A - realized_B,
        })

    per_draw = pd.DataFrame(records)
    diffs = per_draw["diff"].to_numpy()
    t_stat, p_value = _paired_t_test(diffs)

    r = BacktestResult(
        lottery=lottery,
        per_draw=per_draw,
        mean_A=float(per_draw["realized_A"].mean()),
        mean_B=float(per_draw["realized_B"].mean()),
        mean_diff=float(diffs.mean()),
        std_diff=float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
        t_stat=t_stat,
        p_value=p_value,
        n_draws=len(per_draw),
        ticket_cost=float(cfg["ticket_cost"]),
        messages=messages,
    )

    # Report
    print()
    print(f"  n evaluated:     {r.n_draws}")
    print(f"  mean realized A (min-popularity): ${r.mean_A:.4f}")
    print(f"  mean realized B (uniform random): ${r.mean_B:.4f}")
    print(f"  mean diff (A - B):                ${r.mean_diff:.4f}")
    print(f"  std  diff:                        ${r.std_diff:.4f}")
    print(f"  paired t-stat:                    {r.t_stat:.3f}")
    print(f"  two-sided p-value:                {r.p_value:.4f}")
    print(f"  ticket cost (reference):          ${r.ticket_cost:.2f}")
    if r.messages:
        print("\n  messages:")
        for m in r.messages:
            print(f"    - {m}")

    if save_csv:
        out_path = config.BACKTEST_RESULTS
        out_path.parent.mkdir(parents=True, exist_ok=True)
        per_draw.to_csv(out_path, index=False)
        print(f"\n[backtest] per-draw results saved to {out_path}")

    # Honest verdict
    print()
    if r.p_value < 0.05 and r.mean_diff > 0:
        print("[backtest] VERDICT: statistically significant lift over uniform baseline.")
    elif r.mean_diff > 0:
        print("[backtest] VERDICT: positive lift but not statistically significant "
              "at alpha=0.05 (more data needed).")
    else:
        print("[backtest] VERDICT: NO lift over uniform baseline. "
              "Document this null result honestly in CLAUDE.md.")
    return 0
