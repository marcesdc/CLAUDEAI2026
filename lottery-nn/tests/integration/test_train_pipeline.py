"""Integration tests for the single-lottery train/predict pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import pytest
import torch
import config
from datetime import datetime, timedelta
from src.preprocessing import build_features, split
from src.train import train


def _make_synthetic_df(n=20, seed=0):
    rng = np.random.default_rng(seed)
    main_max = config.LOTTERY["main_max"]
    main_count = config.LOTTERY["main_count"]
    records = []
    date = datetime(2026, 1, 1)
    for _ in range(n):
        main = sorted(int(v) + 1 for v in rng.choice(main_max, size=main_count, replace=False))
        records.append({"date": date.strftime("%Y-%m-%d"),
                        **{f"n{i+1}": v for i, v in enumerate(main)}})
        date += timedelta(days=3)
    return pd.DataFrame(records)


def test_full_single_lottery_pipeline(tmp_path):
    df = _make_synthetic_df(n=config.SEQUENCE_LEN + 10)
    X, y_main, y_bonus = build_features(df)
    assert X.shape[0] > 0
    train_data, val_data, _ = split(X, y_main, y_bonus)
    ckpt = str(tmp_path / "test_model.pt")
    model, history = train(train_data, val_data, epochs=2, checkpoint=ckpt)
    assert len(history["train_loss"]) > 0
    assert Path(ckpt).exists()


def test_train_accepts_bare_filename_checkpoint(tmp_path, monkeypatch):
    """Codex M2 guard: --checkpoint best.pt (no dir) must not crash
    on os.makedirs('', exist_ok=True)."""
    monkeypatch.chdir(tmp_path)
    df = _make_synthetic_df(n=config.SEQUENCE_LEN + 10, seed=2)
    X, y_main, y_bonus = build_features(df)
    train_data, val_data, _ = split(X, y_main, y_bonus)
    model, _ = train(train_data, val_data, epochs=1, checkpoint="bare.pt")
    assert (tmp_path / "bare.pt").exists()


def test_predict_after_train(tmp_path, monkeypatch):
    from src import feedback
    monkeypatch.setattr(feedback, "PRED_LOG", str(tmp_path / "pred_log.csv"))
    from src.predict import predict

    df = _make_synthetic_df(n=config.SEQUENCE_LEN + 10, seed=1)
    X, y_main, y_bonus = build_features(df)
    train_data, val_data, _ = split(X, y_main, y_bonus)
    ckpt = str(tmp_path / "predict_test.pt")
    model, _ = train(train_data, val_data, epochs=2, checkpoint=ckpt)

    window = X[-1]
    plays = predict(model, window, n=1)
    assert len(plays) == 1
    assert len(plays[0]["lines"][0]) == config.LOTTERY["main_count"]
