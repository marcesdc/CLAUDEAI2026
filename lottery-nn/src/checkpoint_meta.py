"""
Checkpoint metadata sidecar for shape-compatibility checking.

When a checkpoint is saved, we drop a small JSON file next to it recording
the architectural dimensions that the weights were trained against:

    models/best_swarm.pt        -- the actual torch.save() weights
    models/best_swarm.meta.json -- {pool_max, main_head_sizes, bonus_head_sizes, saved_at}

On load, assert_compatible() compares the sidecar against the current
constants in src.model_swarm. If they disagree (e.g., LottoMax pool was 50
when the checkpoint was saved but is now 52), we raise RuntimeError with
a clear message instead of letting PyTorch produce a cryptic shape mismatch.

Legacy checkpoints saved before this module existed have no sidecar; we
warn-only and proceed so existing models keep loading until retrained.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _meta_path(ckpt_path: str) -> Path:
    """Return the sidecar path for a checkpoint (sibling .meta.json)."""
    p = Path(ckpt_path)
    return p.with_suffix(p.suffix + ".meta.json") if p.suffix else p.with_suffix(".meta.json")


def write_meta(ckpt_path: str,
               pool_max: int,
               main_head_sizes: list[int],
               bonus_head_sizes: list[int]) -> Path:
    """Write the sidecar JSON alongside *ckpt_path*. Returns the sidecar path."""
    sidecar = _meta_path(ckpt_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pool_max":         int(pool_max),
        "main_head_sizes":  [int(s) for s in main_head_sizes],
        "bonus_head_sizes": [int(s) for s in bonus_head_sizes],
        "saved_at":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return sidecar


def read_meta(ckpt_path: str) -> dict | None:
    """Load and return the sidecar dict, or None if no sidecar exists."""
    sidecar = _meta_path(ckpt_path)
    if not sidecar.exists():
        return None
    with open(sidecar, encoding="utf-8") as f:
        return json.load(f)


def assert_compatible(ckpt_path: str,
                      expected_pool_max: int,
                      expected_main_head_sizes: list[int],
                      expected_bonus_head_sizes: list[int]) -> None:
    """
    Verify a checkpoint's sidecar matches current architecture.

    - No sidecar present  : print a warning and proceed (legacy checkpoint).
    - Sidecar mismatches  : raise RuntimeError with retraining hint.
    - Sidecar matches     : silent.
    """
    meta = read_meta(ckpt_path)
    if meta is None:
        print(
            f"[checkpoint_meta] WARNING: no sidecar for '{ckpt_path}'. "
            f"Cannot verify architectural compatibility -- if load fails with a "
            f"shape mismatch, retrain with: python main_swarm.py joint-train",
            file=sys.stderr,
        )
        return

    mismatches = []
    if int(meta.get("pool_max", -1)) != int(expected_pool_max):
        mismatches.append(f"pool_max: checkpoint={meta.get('pool_max')} current={expected_pool_max}")
    if list(meta.get("main_head_sizes", [])) != list(expected_main_head_sizes):
        mismatches.append(
            f"main_head_sizes: checkpoint={meta.get('main_head_sizes')} "
            f"current={expected_main_head_sizes}"
        )
    if list(meta.get("bonus_head_sizes", [])) != list(expected_bonus_head_sizes):
        mismatches.append(
            f"bonus_head_sizes: checkpoint={meta.get('bonus_head_sizes')} "
            f"current={expected_bonus_head_sizes}"
        )

    if mismatches:
        joined = "\n  - ".join(mismatches)
        raise RuntimeError(
            f"[checkpoint_meta] '{ckpt_path}' is incompatible with current architecture:\n"
            f"  - {joined}\n"
            f"Retrain with: python main_swarm.py joint-train"
        )
