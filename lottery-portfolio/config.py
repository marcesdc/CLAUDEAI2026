"""
config.py -- lottery rules, payout structures, and file paths for lottery-portfolio.

Single source of truth. Three lotteries: lottomax, 649, dailygrand.
Prize tables are based on OLG payout schedules (verify against current OLG site
before relying on for real-money decisions; lotteries occasionally restructure tiers).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Project paths (relative to this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / "data"

# Per-lottery draw CSVs (winning numbers only)
DRAWS_CSV = {
    "lottomax":   DATA_DIR / "draws_lottomax.csv",
    "649":        DATA_DIR / "draws_649.csv",
    "dailygrand": DATA_DIR / "draws_dailygrand.csv",
}

# Per-lottery payout CSVs (winner counts + jackpot per draw)
PAYOUTS_CSV = {
    "lottomax":   DATA_DIR / "payouts_lottomax.csv",
    "649":        DATA_DIR / "payouts_649.csv",
    "dailygrand": DATA_DIR / "payouts_dailygrand.csv",
}

PORTFOLIO_LOG    = DATA_DIR / "portfolio_log.csv"
REALIZED_EV_LOG  = DATA_DIR / "realized_ev_log.csv"
PORTFOLIO_STATE  = DATA_DIR / "portfolio_state.json"
BACKTEST_RESULTS = DATA_DIR / "backtest_results.csv"

# ---------------------------------------------------------------------------
# Lottery rules
# ---------------------------------------------------------------------------
# Each entry defines:
#   name           display name
#   main_count     numbers drawn from main pool
#   main_max       size of main pool
#   bonus_max      size of bonus pool (or None if no bonus)
#   bonus_col      column name in CSV for the bonus number
#   ticket_cost    CAD cost per single ticket line
#   draws_per_week typical schedule (informational)
LOTTERY_RULES = {
    "lottomax": {
        "name":           "LottoMax",
        "main_count":     7,
        "main_max":       52,   # Rule change: LottoMax pool expanded from 50 to 52
        "bonus_max":      52,   # Bonus drawn from the same expanded pool
        "bonus_col":      "bonus",
        "has_bonus":      False,  # Bonus is machine-drawn, NOT player-picked
        "ticket_cost":    5.00,
        "draws_per_week": 2,    # Tuesday + Friday
    },
    "649": {
        "name":           "Lotto 6/49",
        "main_count":     6,
        "main_max":       49,
        "bonus_max":      49,
        "bonus_col":      "bonus",
        "has_bonus":      False,  # Bonus is machine-drawn, NOT player-picked
        "ticket_cost":    3.00,
        "draws_per_week": 2,    # Wednesday + Saturday
    },
    "dailygrand": {
        "name":           "Daily Grand",
        "main_count":     5,
        "main_max":       49,
        "bonus_max":      7,
        "bonus_col":      "grand",
        "has_bonus":      True,  # Bonus (grand) IS player-picked
        "ticket_cost":    3.00,
        "draws_per_week": 2,    # Monday + Thursday (per OLG)
    },
}

# ---------------------------------------------------------------------------
# Prize tier structures
# ---------------------------------------------------------------------------
# Each tier is (matches_main, matches_bonus, payout). matches_bonus None means
# bonus is irrelevant for that tier. payout=None means split a pari-mutuel pool
# (jackpot) -- handled specially in ev_optimizer.
#
# These tables follow OLG's published prize structure as of early 2026. Verify
# against https://www.olg.ca before using for real-money decisions.

PRIZE_TIERS = {
    "lottomax": [
        # (main_match, bonus_match, payout_cad)
        (7, None, None),     # Jackpot (87% of pools fund) -- pari-mutuel
        (6, True, None),     # 2nd prize (4% of pools fund) -- pari-mutuel share
        (6, False, None),    # 3rd prize (4% of pools fund) -- pari-mutuel share
        (5, None, None),     # 4th prize (3.5% of pools fund) -- pari-mutuel share
        (4, None, 20.0),     # Fixed
        (3, True, 20.0),     # Fixed
        (3, None, 0.0),      # Free play (treat as 0 for EV)
        (2, True, 0.0),      # Free play
    ],
    "649": [
        (6, None,  None),    # Jackpot -- pari-mutuel
        (5, True,  None),    # 2nd prize (5/6 + bonus) -- pari-mutuel
        (5, False, None),    # 3rd prize (5/6 no bonus) -- pari-mutuel (disjoint from 2nd)
        (4, None, 85.0),     # Approx fixed (varies slightly by sales)
        (3, None, 10.0),     # Fixed
        (2, True,  5.0),     # Fixed (2/6 + bonus)
        (2, False, 0.0),     # Free play (2/6 no bonus)
    ],
    "dailygrand": [
        # main_count=5, bonus pool=7 (Grand Number 1-7)
        (5, True,  None),    # GRAND PRIZE: $1000/day for life or $7M cash -- treat as fixed
        (5, False, 100000.0),
        (4, True,  1000.0),
        (4, False, 500.0),
        (3, True,  100.0),
        (3, False, 20.0),
        (2, True,  10.0),
        (1, True,  4.0),
    ],
}

# For Daily Grand, the GRAND PRIZE has a cash-out option of ~$7M with no split risk
# (only one player can win the GRAND PRIZE per draw -- no pari-mutuel sharing).
# We model it as a fixed payout for EV math.
DAILYGRAND_GRAND_CASH_VALUE = 7_000_000.0

# ---------------------------------------------------------------------------
# Popularity model defaults (literature priors)
# ---------------------------------------------------------------------------
# See popularity_model.py for full citations. These are starting points;
# the fit() routine updates them via maximum likelihood with regularization
# back to these priors.
POPULARITY_PRIORS = {
    "w_birthday":         1.30,   # Numbers 1-31 over-chosen (DOB bias)
    "w_lucky7":           1.05,   # Multiples of 7 mildly over-chosen
    "w_consecutive":      1.02,   # Slight bonus per adjacent pair in C
    "w_arithmetic":       1.10,   # Arithmetic progressions are popular patterns
    "w_recent_winner":    1.05,   # Players chase recent winners
    "w_low_high_balance": 0.98,   # Mild penalty when all C in same half
    "w_round_decade":     1.03,   # Numbers ending in 0/5 mildly over-chosen
    "w_diagonal":         1.02,   # Bet-slip diagonals (1-7-13-19-25-31-37 etc.)
}

POPULARITY_PRIOR_SIGMA = 0.20   # Gaussian prior std for L2 regularization

# ---------------------------------------------------------------------------
# Strategy registry (for the bandit)
# ---------------------------------------------------------------------------
STRATEGIES = ["anti_popular", "uniform_random", "max_ev_search"]

# ---------------------------------------------------------------------------
# Default operational settings
# ---------------------------------------------------------------------------
LINES_PER_PLAY = 1               # We optimize one line per draw, per lottery
SEARCH_RESTARTS = 1000           # Local-search restarts in ev_optimizer
SEARCH_MAX_SWAPS = 200           # Per-restart swap budget
HELDOUT_FRACTION = 0.25          # Last 25% of draws used for held-out NLL eval

# Estimated tickets sold per draw -- fallback when OLG doesn't publish
# (used as initialization; popularity_model.fit() refines via tier counts)
DEFAULT_TICKETS_SOLD = {
    "lottomax":   8_000_000,
    "649":        4_000_000,
    "dailygrand":   500_000,
}
