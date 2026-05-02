"""
Tests for src.checkpoint_meta -- the architectural compatibility sidecar
that prevents cryptic shape mismatches when a stale checkpoint is loaded
after the lottery pool size or head sizes change.
"""

import json
import sys

import pytest

from src.checkpoint_meta import (
    _meta_path,
    assert_compatible,
    read_meta,
    write_meta,
)


def test_write_then_read_round_trip(tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"dummy")

    meta_file = write_meta(
        str(ckpt),
        pool_max=52,
        main_head_sizes=[52, 49, 49],
        bonus_head_sizes=[52, 49, 7],
    )

    meta = read_meta(str(ckpt))
    assert meta is not None
    assert meta["pool_max"] == 52
    assert meta["main_head_sizes"] == [52, 49, 49]
    assert meta["bonus_head_sizes"] == [52, 49, 7]
    assert "saved_at" in meta
    assert meta_file.exists()
    assert meta_file == _meta_path(str(ckpt))


def test_assert_compatible_passes_on_match(tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"dummy")
    write_meta(str(ckpt), pool_max=52, main_head_sizes=[52, 49, 49], bonus_head_sizes=[52, 49, 7])

    # Must not raise
    assert_compatible(str(ckpt), 52, [52, 49, 49], [52, 49, 7])


def test_assert_compatible_raises_on_pool_mismatch(tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"dummy")
    write_meta(str(ckpt), pool_max=50, main_head_sizes=[50, 49, 49], bonus_head_sizes=[50, 49, 7])

    with pytest.raises(RuntimeError) as excinfo:
        assert_compatible(str(ckpt), 52, [52, 49, 49], [52, 49, 7])

    msg = str(excinfo.value)
    assert "pool_max" in msg
    assert "joint-train" in msg


def test_assert_compatible_raises_on_main_head_mismatch(tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"dummy")
    write_meta(str(ckpt), pool_max=52, main_head_sizes=[50, 49, 49], bonus_head_sizes=[52, 49, 7])

    with pytest.raises(RuntimeError) as excinfo:
        assert_compatible(str(ckpt), 52, [52, 49, 49], [52, 49, 7])

    assert "main_head_sizes" in str(excinfo.value)


def test_assert_compatible_raises_on_bonus_head_mismatch(tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"dummy")
    write_meta(str(ckpt), pool_max=52, main_head_sizes=[52, 49, 49], bonus_head_sizes=[52, 49, 5])

    with pytest.raises(RuntimeError) as excinfo:
        assert_compatible(str(ckpt), 52, [52, 49, 49], [52, 49, 7])

    assert "bonus_head_sizes" in str(excinfo.value)


def test_assert_compatible_warn_only_on_missing_sidecar(tmp_path, capsys):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"dummy")
    # No sidecar written

    # Must NOT raise -- legacy checkpoint path
    assert_compatible(str(ckpt), 52, [52, 49, 49], [52, 49, 7])

    captured = capsys.readouterr()
    # Warning must go to stderr, not stdout
    assert "no sidecar" in captured.err
    assert captured.out == ""


def test_read_meta_returns_none_when_missing(tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"dummy")
    assert read_meta(str(ckpt)) is None


def test_meta_path_for_pt_extension(tmp_path):
    p = tmp_path / "best.pt"
    expected = tmp_path / "best.pt.meta.json"
    assert _meta_path(str(p)) == expected


def test_meta_path_creates_parent_dir(tmp_path):
    nested = tmp_path / "models" / "deeper" / "best.pt"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"dummy")
    write_meta(str(nested), pool_max=52, main_head_sizes=[52], bonus_head_sizes=[])
    assert (tmp_path / "models" / "deeper" / "best.pt.meta.json").exists()
