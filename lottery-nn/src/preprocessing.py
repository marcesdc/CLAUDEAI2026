"""
Feature engineering for lottery draws.

Each training sample is a fixed-length window of past draws (SEQUENCE_LEN)
represented as a multi-hot vector over the full number space, augmented with
frequency and recency statistics.

Shapes (after processing)
--------------------------
X : (N, SEQUENCE_LEN, main_max)   – multi-hot history window
y : (N, main_max)                  – multi-hot target (next main draw)
y_bonus : (N, bonus_max)           – one-hot bonus target
"""

import numpy as np
import pandas as pd

import config


MAIN_MAX = config.LOTTERY["main_max"]
BONUS_MAX = config.LOTTERY["bonus_max"]
MAIN_COUNT = config.LOTTERY["main_count"]
SEQ_LEN = config.SEQUENCE_LEN


def build_features(df: pd.DataFrame):
    """
    Convert a DataFrame of draws into (X, y_main, y_bonus) numpy arrays.
    """
    main_cols = [c for c in df.columns if c.startswith("n")]
    has_bonus = "bonus" in df.columns

    main_draws = df[main_cols].values.astype(int)   # (N, main_count)
    bonus_draws = df["bonus"].values.astype(int) if has_bonus else None

    N = len(main_draws)

    # --- multi-hot encoding for each draw ---------------------------------
    main_hot = _multi_hot(main_draws, MAIN_MAX)      # (N, MAIN_MAX)
    bonus_hot = _one_hot(bonus_draws, BONUS_MAX) if has_bonus else None  # (N, BONUS_MAX)

    # --- sliding-window sequences -----------------------------------------
    X, y_main, y_bonus = [], [], []

    for i in range(SEQ_LEN, N):
        window = main_hot[i - SEQ_LEN : i]           # (SEQ_LEN, MAIN_MAX)

        # Augment each step with rolling frequency over the window
        freq = window.mean(axis=0, keepdims=True)     # (1, MAIN_MAX)
        aug = np.concatenate([window, np.tile(freq, (SEQ_LEN, 1))], axis=-1)
        # aug shape: (SEQ_LEN, 2 * MAIN_MAX)

        X.append(aug)
        y_main.append(main_hot[i])
        if bonus_hot is not None:
            y_bonus.append(bonus_hot[i])

    X = np.array(X, dtype=np.float32)                # (N', SEQ_LEN, 2*MAIN_MAX)
    y_main = np.array(y_main, dtype=np.float32)       # (N', MAIN_MAX)
    y_bonus = np.array(y_bonus, dtype=np.float32) if y_bonus else None

    print(f"[preprocessing] X={X.shape}  y_main={y_main.shape}", end="")
    if y_bonus is not None:
        print(f"  y_bonus={y_bonus.shape}", end="")
    print()

    return X, y_main, y_bonus


def split(X, y_main, y_bonus=None, val=config.VAL_SPLIT, test=config.TEST_SPLIT):
    """Chronological train / val / test split (no shuffling)."""
    N = len(X)
    n_test = max(1, int(N * test))
    n_val = max(1, int(N * val))
    n_train = N - n_val - n_test

    slices = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, N),
    }

    def _cut(arr):
        if arr is None:
            return None, None, None
        return arr[slices["train"]], arr[slices["val"]], arr[slices["test"]]

    X_tr, X_v, X_te = _cut(X)
    ym_tr, ym_v, ym_te = _cut(y_main)
    yb_tr, yb_v, yb_te = _cut(y_bonus)

    print(
        f"[preprocessing] split -> train={n_train}  val={n_val}  test={n_test}"
    )
    return (X_tr, ym_tr, yb_tr), (X_v, ym_v, yb_v), (X_te, ym_te, yb_te)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _multi_hot(arr: np.ndarray, max_val: int) -> np.ndarray:
    """arr: (N, k)  →  out: (N, max_val)  (0-indexed internally)"""
    N = len(arr)
    out = np.zeros((N, max_val), dtype=np.float32)
    for i, row in enumerate(arr):
        for v in row:
            if 1 <= v <= max_val:
                out[i, v - 1] = 1.0
    return out


def _one_hot(arr: np.ndarray, max_val: int) -> np.ndarray:
    """arr: (N,)  →  out: (N, max_val)"""
    N = len(arr)
    out = np.zeros((N, max_val), dtype=np.float32)
    for i, v in enumerate(arr):
        if 1 <= v <= max_val:
            out[i, v - 1] = 1.0
    return out


def decode_multihot(vec: np.ndarray, top_k: int = config.LOTTERY["main_count"]) -> list[int]:
    """Return *top_k* 1-indexed ball numbers from a probability / multi-hot vector."""
    indices = np.argsort(vec)[::-1][:top_k]
    return sorted(int(i + 1) for i in indices)


def decode_onehot(vec: np.ndarray) -> int:
    """Return the 1-indexed ball number with highest probability."""
    return int(np.argmax(vec)) + 1
