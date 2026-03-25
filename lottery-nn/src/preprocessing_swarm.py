"""
Multi-lottery feature engineering for the swarm shared encoder.

All lotteries are encoded to the same input dimension (2 * POOL_MAX = 100)
by padding multi-hot and rolling-frequency vectors to POOL_MAX=50.

Target vectors (y_main, y_bonus) retain their natural size per lottery
so the per-lottery heads learn the right output distribution.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
POOL_MAX = 50   # largest pool across all games (LottoMax uses 1-50)
SEQ_LEN  = 10   # draw history window

# ---------------------------------------------------------------------------
# Lottery registry — single source of truth for all per-lottery settings
# ---------------------------------------------------------------------------
LOTTERY_CONFIGS = {
    "lottomax": {
        "id":          0,
        "name":        "Lotto Max",
        "main_count":  7,
        "main_max":    50,
        "bonus_max":   50,
        "bonus_col":   "bonus",
        "csv":         "data/draws.csv",
        "checkpoint":  "models/best_swarm.pt",   # shared checkpoint
    },
    "649": {
        "id":          1,
        "name":        "Lotto 6/49",
        "main_count":  6,
        "main_max":    49,
        "bonus_max":   49,
        "bonus_col":   "bonus",
        "csv":         "data/draws_649.csv",
        "checkpoint":  "models/best_swarm.pt",
    },
    "dailygrand": {
        "id":          2,
        "name":        "Daily Grand",
        "main_count":  5,
        "main_max":    49,
        "bonus_max":   7,
        "bonus_col":   "grand",
        "csv":         "data/draws_dailygrand.csv",
        "checkpoint":  "models/best_swarm.pt",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_lottery_df(lottery_name: str) -> pd.DataFrame:
    """Load and sort a lottery CSV, normalising the bonus column name."""
    cfg = LOTTERY_CONFIGS[lottery_name]
    df  = pd.read_csv(cfg["csv"], parse_dates=["date"])
    df  = df.sort_values("date").reset_index(drop=True)
    # Normalise bonus column so all lotteries use "bonus" internally
    if cfg["bonus_col"] != "bonus" and cfg["bonus_col"] in df.columns:
        df = df.rename(columns={cfg["bonus_col"]: "bonus"})
    return df


def build_features(df: pd.DataFrame, cfg: dict, seq_len: int = SEQ_LEN):
    """
    Build (X, y_main, y_bonus) for one lottery.

    X       : (N', seq_len, 2*POOL_MAX)   -- padded to POOL_MAX=50
    y_main  : (N', main_max)              -- natural size for this lottery
    y_bonus : (N', bonus_max)             -- natural size for this lottery
    """
    main_count = cfg["main_count"]
    main_max   = cfg["main_max"]
    bonus_max  = cfg["bonus_max"]

    main_cols   = [f"n{i}" for i in range(1, main_count + 1)]
    main_draws  = df[main_cols].values.astype(int)
    bonus_draws = df["bonus"].values.astype(int)

    N = len(main_draws)

    # Input: padded to POOL_MAX so all lotteries share the same feature space
    main_hot_padded = _multi_hot(main_draws, POOL_MAX)     # (N, 50)
    # Targets: natural size (no padding on outputs)
    y_hot           = _multi_hot(main_draws, main_max)     # (N, main_max)
    bonus_hot       = _one_hot(bonus_draws,  bonus_max)    # (N, bonus_max)

    X, y_main, y_bonus = [], [], []
    for i in range(seq_len, N):
        window = main_hot_padded[i - seq_len : i]          # (seq_len, 50)
        freq   = window.mean(axis=0, keepdims=True)
        aug    = np.concatenate(
            [window, np.tile(freq, (seq_len, 1))], axis=-1
        )                                                   # (seq_len, 100)
        X.append(aug)
        y_main.append(y_hot[i])
        y_bonus.append(bonus_hot[i])

    return (
        np.array(X,       dtype=np.float32),
        np.array(y_main,  dtype=np.float32),
        np.array(y_bonus, dtype=np.float32),
    )


def build_all_lottery_data(seq_len: int = SEQ_LEN) -> dict:
    """
    Load and preprocess all 3 lotteries.

    Returns
    -------
    dict: { lottery_name -> (X, y_main, y_bonus) }
    """
    result = {}
    for name, cfg in LOTTERY_CONFIGS.items():
        df = load_lottery_df(name)
        X, y_main, y_bonus = build_features(df, cfg, seq_len=seq_len)
        result[name] = (X, y_main, y_bonus)
        print(
            f"[swarm] {cfg['name']:12s}  draws={len(df)}  samples={len(X)}"
            f"  X={X.shape}  y_main={y_main.shape}  y_bonus={y_bonus.shape}"
        )
    return result


def split(X, y_main, y_bonus, val: float = 0.15, test: float = 0.05):
    """Chronological train / val / test split (no shuffling)."""
    N       = len(X)
    n_test  = max(1, int(N * test))
    n_val   = max(1, int(N * val))
    n_train = N - n_val - n_test

    tr = slice(0, n_train)
    v  = slice(n_train, n_train + n_val)
    te = slice(n_train + n_val, N)

    return (
        (X[tr], y_main[tr], y_bonus[tr]),
        (X[v],  y_main[v],  y_bonus[v]),
        (X[te], y_main[te], y_bonus[te]),
    )


def get_last_window(lottery_name: str, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Return the most recent preprocessed window for inference."""
    cfg        = LOTTERY_CONFIGS[lottery_name]
    df         = load_lottery_df(lottery_name)
    main_count = cfg["main_count"]
    main_cols  = [f"n{i}" for i in range(1, main_count + 1)]
    main_draws = df[main_cols].values.astype(int)
    main_hot   = _multi_hot(main_draws, POOL_MAX)
    window     = main_hot[-seq_len:]
    freq       = window.mean(axis=0, keepdims=True)
    aug        = np.concatenate(
        [window, np.tile(freq, (seq_len, 1))], axis=-1
    )
    return aug   # (seq_len, 2*POOL_MAX)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _multi_hot(arr: np.ndarray, max_val: int) -> np.ndarray:
    """arr: (N, k)  ->  (N, max_val) multi-hot (1-indexed)."""
    N   = len(arr)
    out = np.zeros((N, max_val), dtype=np.float32)
    for i, row in enumerate(arr):
        for v in row:
            if 1 <= int(v) <= max_val:
                out[i, int(v) - 1] = 1.0
    return out


def _one_hot(arr: np.ndarray, max_val: int) -> np.ndarray:
    """arr: (N,)  ->  (N, max_val) one-hot (1-indexed)."""
    N   = len(arr)
    out = np.zeros((N, max_val), dtype=np.float32)
    for i, v in enumerate(arr):
        if 1 <= int(v) <= max_val:
            out[i, int(v) - 1] = 1.0
    return out
