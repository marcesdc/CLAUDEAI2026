"""Unit tests for src/data_loader.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import pytest
import config
from src.data_loader import load_draws, generate_synthetic, _normalize_columns, _validate


def test_generate_synthetic_shape(tmp_path):
    path = str(tmp_path / "draws.csv")
    df = generate_synthetic(n_draws=50, save_path=path)
    assert len(df) == 50
    assert "date" in df.columns
    assert "n1" in df.columns


def test_generate_synthetic_range(tmp_path):
    path = str(tmp_path / "draws.csv")
    df = generate_synthetic(n_draws=20, save_path=path)
    main_max = config.LOTTERY["main_max"]
    for col in [f"n{i}" for i in range(1, config.LOTTERY["main_count"] + 1)]:
        assert df[col].between(1, main_max).all()


def test_load_draws_from_csv(tmp_path):
    path = tmp_path / "draws.csv"
    rows = [{"date": "2026-01-01", **{f"n{i}": i * 3 for i in range(1, 8)}},
            {"date": "2026-01-04", **{f"n{i}": i * 3 + 1 for i in range(1, 8)}}]
    pd.DataFrame(rows).to_csv(path, index=False)
    df = load_draws(str(path))
    assert len(df) == 2
    assert "n1" in df.columns


def test_normalize_columns_aliases():
    df = pd.DataFrame([{"date": "2026-01-01",
                        **{f"number{i}": i for i in range(1, 8)}}])
    df_norm = _normalize_columns(df)
    assert "n1" in df_norm.columns
    assert "number1" not in df_norm.columns


def test_normalize_columns_grand_unchanged():
    df = pd.DataFrame([{"n1": 1, "n2": 2, "grand": 3, "date": "2026-01-01"}])
    df_norm = _normalize_columns(df)
    assert "grand" in df_norm.columns


def test_validate_raises_missing_col():
    df = pd.DataFrame([{"date": "2026-01-01", "n1": 1}])
    with pytest.raises(ValueError, match="missing"):
        _validate(df)


def test_validate_raises_out_of_range():
    row = {"date": "2026-01-01", **{f"n{i}": 1 for i in range(1, 8)}}
    row["n1"] = 99
    df = pd.DataFrame([row])
    with pytest.raises(ValueError, match="outside"):
        _validate(df)


def test_load_draws_fallback_synthetic(monkeypatch, tmp_path):
    synthetic_path = str(tmp_path / "synthetic.csv")
    monkeypatch.setattr(config, "RAW_CSV", synthetic_path)
    df = load_draws(str(tmp_path / "does_not_exist.csv"))
    assert len(df) > 0
    assert "n1" in df.columns
