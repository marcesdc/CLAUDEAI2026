"""
Tests for window-jitter data augmentation (B4) in src.preprocessing_swarm.

The central invariant is leak-prevention: augmented windows are appended
only to the train slice; val and test slices stay byte-identical to the
deterministic build, regardless of aug_factor.
"""

import numpy as np
import pytest

from src.preprocessing_swarm import (
    LOTTERY_CONFIGS,
    POOL_MAX,
    build_features,
    build_split_features,
)


# ---------------------------------------------------------------------------
# build_features: deterministic, byte-identical to legacy when augment=False
# ---------------------------------------------------------------------------

def test_build_features_unchanged_for_legacy_callers(df_lottomax):
    cfg = LOTTERY_CONFIGS["lottomax"]
    X, y_main, y_bonus = build_features(df_lottomax, cfg)
    assert X.shape[0] == len(df_lottomax) - 10  # SEQ_LEN=10
    assert X.shape[1] == 10
    assert X.shape[2] == 2 * POOL_MAX
    assert y_main.shape == (X.shape[0], cfg["main_max"])
    assert y_bonus.shape == (X.shape[0], cfg["bonus_max"])


# ---------------------------------------------------------------------------
# build_split_features: augment=False is a clean refactor of split()
# ---------------------------------------------------------------------------

def test_split_features_no_augment_matches_legacy_split(df_lottomax):
    """augment=False must produce the same train/val/test as build_features + split()."""
    from src.preprocessing_swarm import split
    cfg = LOTTERY_CONFIGS["lottomax"]
    X, y_main, y_bonus = build_features(df_lottomax, cfg)
    legacy_tr, legacy_v, legacy_te = split(X, y_main, y_bonus)

    new_tr, new_v, new_te = build_split_features(df_lottomax, cfg, augment=False)
    np.testing.assert_array_equal(new_tr[0], legacy_tr[0])
    np.testing.assert_array_equal(new_tr[1], legacy_tr[1])
    np.testing.assert_array_equal(new_tr[2], legacy_tr[2])
    np.testing.assert_array_equal(new_v[0],  legacy_v[0])
    np.testing.assert_array_equal(new_te[0], legacy_te[0])


# ---------------------------------------------------------------------------
# Augmentation increases train count, leaves val/test alone
# ---------------------------------------------------------------------------

def test_augmentation_increases_train_count_only(df_lottomax):
    cfg = LOTTERY_CONFIGS["lottomax"]
    base_tr, base_v, base_te = build_split_features(df_lottomax, cfg, augment=False)
    aug_tr,  aug_v,  aug_te  = build_split_features(
        df_lottomax, cfg, augment=True, aug_factor=3,
        rng=np.random.default_rng(0),
    )
    # Train slice grows; aug_factor=3 means up to 2 extras per train sample.
    assert len(aug_tr[0]) > len(base_tr[0]), \
        f"train did not grow: base={len(base_tr[0])} aug={len(aug_tr[0])}"
    # Val and test slices are byte-identical (deterministic, leak-free).
    np.testing.assert_array_equal(aug_v[0],  base_v[0])
    np.testing.assert_array_equal(aug_v[1],  base_v[1])
    np.testing.assert_array_equal(aug_te[0], base_te[0])
    np.testing.assert_array_equal(aug_te[1], base_te[1])


def test_augmentation_factor_3_triples_train_within_one(df_649):
    cfg = LOTTERY_CONFIGS["649"]
    base_tr, _, _ = build_split_features(df_649, cfg, augment=False)
    aug_tr,  _, _ = build_split_features(
        df_649, cfg, augment=True, aug_factor=3,
        rng=np.random.default_rng(0),
    )
    # Exactly 3x in the perfect case; we allow a small margin for jitter clamping.
    expected = 3 * len(base_tr[0])
    assert abs(len(aug_tr[0]) - expected) <= 2, \
        f"expected ~{expected} train samples, got {len(aug_tr[0])}"


def test_augmentation_factor_1_is_noop(df_lottomax):
    """aug_factor=1 means no extras even with augment=True."""
    cfg = LOTTERY_CONFIGS["lottomax"]
    base_tr, _, _ = build_split_features(df_lottomax, cfg, augment=False)
    aug_tr,  _, _ = build_split_features(
        df_lottomax, cfg, augment=True, aug_factor=1,
        rng=np.random.default_rng(0),
    )
    assert len(aug_tr[0]) == len(base_tr[0])


# ---------------------------------------------------------------------------
# Jitter clamping: window starts stay in valid range
# ---------------------------------------------------------------------------

def test_jitter_window_clamped_to_valid_range(df_dailygrand_normalised):
    cfg = LOTTERY_CONFIGS["dailygrand"]
    aug_tr, _, _ = build_split_features(
        df_dailygrand_normalised, cfg, augment=True, aug_factor=4,
        jitter_range=2, rng=np.random.default_rng(0),
    )
    X = aug_tr[0]
    # Every window has shape (seq_len, 2*POOL_MAX); padded windows still
    # match the contract -- so the only thing to verify is shape consistency
    # and finite values (no NaN from out-of-range slicing).
    assert X.ndim == 3
    assert X.shape[1] == 10
    assert X.shape[2] == 2 * POOL_MAX
    assert np.isfinite(X).all()


# ---------------------------------------------------------------------------
# Mixup safety: targets stay hard, only freq channel mixed
# ---------------------------------------------------------------------------

def test_mixup_does_not_touch_targets(df_lottomax):
    cfg = LOTTERY_CONFIGS["lottomax"]
    aug_tr, _, _ = build_split_features(
        df_lottomax, cfg, augment=True, aug_factor=3,
        mixup_alpha=0.4, rng=np.random.default_rng(0),
    )
    y_main = aug_tr[1]
    # Targets are multi-hot {0, 1}, never fractional even with mixup on.
    unique = np.unique(y_main)
    assert set(unique.tolist()).issubset({0.0, 1.0}), \
        f"mixup leaked into targets: unique values {unique}"


def test_mixup_freq_channel_remains_finite(df_lottomax):
    cfg = LOTTERY_CONFIGS["lottomax"]
    aug_tr, _, _ = build_split_features(
        df_lottomax, cfg, augment=True, aug_factor=3,
        mixup_alpha=0.4, rng=np.random.default_rng(0),
    )
    X = aug_tr[0]
    # Freq channel (last POOL_MAX dims) should stay in [0, 1] after mixing.
    freq = X[:, :, POOL_MAX:]
    assert np.isfinite(freq).all()
    assert freq.min() >= 0.0
    assert freq.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Determinism: same RNG seed produces same augmented output
# ---------------------------------------------------------------------------

def test_augmentation_deterministic_with_seed(df_lottomax):
    cfg = LOTTERY_CONFIGS["lottomax"]
    a, _, _ = build_split_features(
        df_lottomax, cfg, augment=True, aug_factor=3,
        rng=np.random.default_rng(42),
    )
    b, _, _ = build_split_features(
        df_lottomax, cfg, augment=True, aug_factor=3,
        rng=np.random.default_rng(42),
    )
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
