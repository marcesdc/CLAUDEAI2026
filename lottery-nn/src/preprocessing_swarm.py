"""
Multi-lottery feature engineering for the swarm shared encoder.

All lotteries are encoded to the same input dimension (2 * POOL_MAX = 104)
by padding multi-hot and rolling-frequency vectors to POOL_MAX=52.

Target vectors (y_main, y_bonus) retain their natural size per lottery
so the per-lottery heads learn the right output distribution.
"""

import numpy as np
import pandas as pd

from src.model_swarm import POOL_MAX, SEQ_LEN  # single source of truth for architecture dims

# ---------------------------------------------------------------------------
# Lottery registry — single source of truth for all per-lottery settings
# ---------------------------------------------------------------------------
LOTTERY_CONFIGS = {
    "lottomax": {
        "id":          0,
        "name":        "Lotto Max",
        "main_count":  7,
        "main_max":    52,     # updated 2026-04-14: range expanded from 50 to 52
        "bonus_max":   52,
        "bonus_col":   "bonus",
        "has_bonus":   False,  # bonus is drawn by the lottery; players don't select it
        "lines_per":   1,      # one set of 7 numbers per play
        "csv":         "data/draws.csv",
        "checkpoint":  "models/best_swarm.pt",   # shared checkpoint
    },
    "649": {
        "id":          1,
        "name":        "Lotto 6/49",
        "main_count":  6,
        "main_max":    49,
        "bonus_max":   49,      # model head size only -- not shown to users (has_bonus=False)
        "has_bonus":   False,   # bonus is randomly drawn by OLG; not available to players
        "bonus_col":   "bonus",
        "lines_per":   1,
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
        "has_bonus":   True,
        "lines_per":   1,
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
    bonus_col = cfg.get("bonus_col", "bonus")
    if bonus_col != "bonus" and bonus_col in df.columns:
        df = df.rename(columns={bonus_col: "bonus"})
    return df


def build_features(df: pd.DataFrame, cfg: dict, seq_len: int = SEQ_LEN):
    """
    Build (X, y_main, y_bonus) for one lottery -- deterministic, one window
    per target index in [seq_len, N). Byte-identical to the pre-Phase-2
    implementation; augmentation lives in build_split_features().

    X       : (N', seq_len, 2*POOL_MAX)   -- padded to POOL_MAX=52 (INPUT_DIM=104)
    y_main  : (N', main_max)              -- natural size for this lottery
    y_bonus : (N', bonus_max)             -- natural size for this lottery
    """
    main_count = cfg["main_count"]
    main_max   = cfg["main_max"]
    bonus_max  = cfg["bonus_max"]
    has_bonus  = cfg.get("has_bonus", True)

    main_cols  = [f"n{i}" for i in range(1, main_count + 1)]
    main_draws = df[main_cols].values.astype(int)

    N = len(main_draws)
    if N <= seq_len:
        raise ValueError(
            f"[swarm] Need more than seq_len={seq_len} draws to build features, "
            f"got {N}. Add more history or lower seq_len."
        )

    main_hot_padded = _multi_hot(main_draws, POOL_MAX)     # (N, 52)
    y_hot = _multi_hot(main_draws, main_max)               # (N, main_max)

    if has_bonus:
        bonus_draws = df["bonus"].values.astype(int)
        bonus_hot   = _one_hot(bonus_draws, bonus_max)
    else:
        bonus_hot   = np.zeros((N, bonus_max), dtype=np.float32)

    X, y_main, y_bonus = [], [], []
    for i in range(seq_len, N):
        window = main_hot_padded[i - seq_len : i]
        freq   = window.mean(axis=0, keepdims=True)
        aug    = np.concatenate([window, np.tile(freq, (seq_len, 1))], axis=-1)
        X.append(aug)
        y_main.append(y_hot[i])
        y_bonus.append(bonus_hot[i])

    return (
        np.array(X,       dtype=np.float32),
        np.array(y_main,  dtype=np.float32),
        np.array(y_bonus, dtype=np.float32),
    )


def build_split_features(df: pd.DataFrame, cfg: dict, seq_len: int = SEQ_LEN,
                         *,
                         val: float = 0.15,
                         test: float = 0.05,
                         augment: bool = False,
                         aug_factor: int = 3,
                         jitter_range: int = 2,
                         mixup_alpha: float = 0.0,
                         rng: np.random.Generator | None = None):
    """
    Build deterministic features, split chronologically into train/val/test,
    then optionally append jitter-augmented windows to the train slice ONLY.

    Augmented windows never touch val/test slices, so train/val/test target
    sets are disjoint by construction (the central leak-prevention invariant).

    Returns ((X_tr, y_main_tr, y_bonus_tr),
             (X_v,  y_main_v,  y_bonus_v),
             (X_te, y_main_te, y_bonus_te))
    """
    X, y_main, y_bonus = build_features(df, cfg, seq_len=seq_len)
    N = len(X)
    n_test  = max(1, int(N * test))
    n_val   = max(1, int(N * val))
    n_train = N - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            f"[swarm] Not enough samples to split: N={N}, n_val={n_val}, n_test={n_test}"
        )

    train = (X[:n_train],          y_main[:n_train],          y_bonus[:n_train])
    valid = (X[n_train:n_train+n_val], y_main[n_train:n_train+n_val], y_bonus[n_train:n_train+n_val])
    tests = (X[n_train+n_val:],    y_main[n_train+n_val:],    y_bonus[n_train+n_val:])

    if not augment or aug_factor <= 1:
        return train, valid, tests

    if rng is None:
        rng = np.random.default_rng(0)

    # Train target indices (in the original draw-row index space, 1-indexed by seq_len)
    # X[k] corresponds to target row index (seq_len + k). So train targets are
    # k in [0, n_train) -> draw indices [seq_len, seq_len + n_train).
    train_end_draw_idx = seq_len + n_train

    main_count = cfg["main_count"]
    main_cols  = [f"n{i}" for i in range(1, main_count + 1)]
    main_draws = df[main_cols].values.astype(int)
    main_hot_padded = _multi_hot(main_draws, POOL_MAX)
    y_hot = _multi_hot(main_draws, cfg["main_max"])

    has_bonus = cfg.get("has_bonus", True)
    bonus_max = cfg["bonus_max"]
    if has_bonus:
        bonus_hot = _one_hot(df["bonus"].values.astype(int), bonus_max)
    else:
        bonus_hot = np.zeros((len(df), bonus_max), dtype=np.float32)

    extras = []
    for i in range(seq_len, train_end_draw_idx):
        for _ in range(aug_factor - 1):
            jitter = int(rng.integers(-jitter_range, jitter_range + 1))
            start = max(0, min(i - seq_len + jitter, i - 1))
            length = i - start
            if length < 1:
                continue
            raw = main_hot_padded[start:i]
            if length < seq_len:
                pad = np.zeros((seq_len - length, POOL_MAX), dtype=np.float32)
                window = np.concatenate([pad, raw], axis=0)
            else:
                window = raw[-seq_len:]
            freq   = window.mean(axis=0, keepdims=True)
            aug_x  = np.concatenate([window, np.tile(freq, (seq_len, 1))], axis=-1).astype(np.float32)
            extras.append((aug_x, y_hot[i].astype(np.float32), bonus_hot[i].astype(np.float32)))

    if mixup_alpha > 0 and len(extras) >= 2:
        mixed = []
        for k in range(0, len(extras) - 1, 2):
            xa, ya_main, ya_bonus = extras[k]
            xb, yb_main, yb_bonus = extras[k + 1]
            lam = float(rng.beta(mixup_alpha, mixup_alpha))
            xa_mix = xa.copy()
            xb_mix = xb.copy()
            # Mix the freq channel only (last POOL_MAX dims). Targets stay hard.
            xa_mix[:, POOL_MAX:] = lam * xa[:, POOL_MAX:] + (1 - lam) * xb[:, POOL_MAX:]
            xb_mix[:, POOL_MAX:] = lam * xb[:, POOL_MAX:] + (1 - lam) * xa[:, POOL_MAX:]
            mixed.append((xa_mix, ya_main, ya_bonus))
            mixed.append((xb_mix, yb_main, yb_bonus))
        if len(extras) % 2 == 1:
            mixed.append(extras[-1])
        extras = mixed

    if extras:
        Xa = np.stack([e[0] for e in extras]).astype(np.float32)
        Ya = np.stack([e[1] for e in extras]).astype(np.float32)
        Ba = np.stack([e[2] for e in extras]).astype(np.float32)
        train = (
            np.concatenate([train[0], Xa]),
            np.concatenate([train[1], Ya]),
            np.concatenate([train[2], Ba]),
        )

    return train, valid, tests


def build_all_lottery_data(seq_len: int = SEQ_LEN) -> dict:
    """
    Load and preprocess all 3 lotteries.

    Returns
    -------
    dict: { lottery_name -> (X, y_main, y_bonus) }   -- the full deterministic
    feature array, suitable for downstream split() (legacy callsite).
    Augmentation is exposed separately via build_all_lottery_splits() so val
    and test stay deterministic.
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


def build_all_lottery_splits(seq_len: int = SEQ_LEN,
                              *,
                              val: float = 0.15,
                              test: float = 0.05,
                              augment: bool = False,
                              aug_factor: int = 3,
                              jitter_range: int = 2,
                              mixup_alpha: float = 0.0,
                              rng: np.random.Generator | None = None) -> dict:
    """
    Load all 3 lotteries and produce per-lottery (train, val, test) tuples
    via build_split_features(). Augmented windows are appended only to the
    train slice; val/test stay deterministic.

    Returns
    -------
    dict: { lottery_name -> (train, val, test) } where each entry is a tuple
    of (X, y_main, y_bonus) arrays.
    """
    result = {}
    for name, cfg in LOTTERY_CONFIGS.items():
        df = load_lottery_df(name)
        train, valid, tests = build_split_features(
            df, cfg, seq_len=seq_len,
            val=val, test=test,
            augment=augment, aug_factor=aug_factor,
            jitter_range=jitter_range, mixup_alpha=mixup_alpha,
            rng=rng,
        )
        result[name] = (train, valid, tests)
        print(
            f"[swarm] {cfg['name']:12s}  draws={len(df)}  "
            f"train={len(train[0])}  val={len(valid[0])}  test={len(tests[0])}"
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
    """Return the most recent preprocessed window for inference.

    Requires at least seq_len draws of history.
    """
    cfg        = LOTTERY_CONFIGS[lottery_name]
    df         = load_lottery_df(lottery_name)
    main_count = cfg["main_count"]
    main_cols  = [f"n{i}" for i in range(1, main_count + 1)]
    main_draws = df[main_cols].values.astype(int)
    if len(main_draws) < seq_len:
        raise ValueError(
            f"[swarm] get_last_window({lottery_name!r}) needs >= {seq_len} draws, "
            f"found {len(main_draws)}. Log more draws before predicting."
        )
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
