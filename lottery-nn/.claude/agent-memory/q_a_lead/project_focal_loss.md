---
name: focal_loss_implementation
description: Focal loss added in Phase 2 -- implementation details, test coverage, guard rails, and known edge cases
type: project
---

Focal loss (`src/focal_loss.py`) replaced `nn.BCEWithLogitsLoss` for main-number targets in both `src/train.py` and `main_swarm.py`. Config keys `FOCAL_GAMMA=2.0` and `FOCAL_ALPHA=0.25` are the active values as of 2026-04-02.

**Why:** Down-weight easy negatives (non-drawn numbers) to sharpen training signal on the hard positives (drawn numbers). Based on Lin et al. 2017 RetinaNet focal loss, adapted to multi-label binary targets.

**How to apply:** When reviewing train.py or main_swarm.py loss changes, check that `config.FOCAL_GAMMA` and `config.FOCAL_ALPHA` are passed as keyword args. `FOCAL_ALPHA = None` is a valid config value (disables alpha weighting) -- the config comment now documents this explicitly.

Guards added 2026-04-02 (follow-up commit):
- `gamma < 0` raises ValueError immediately (tested: test_negative_gamma_raises)
- `alpha` outside (0, 1) raises ValueError immediately (tested: test_alpha_out_of_range_raises)
- Both boundary values (alpha=0.0, alpha=1.0) are correctly rejected by the strict (0 < alpha < 1) check

Known edge cases verified GREEN:
- gamma=0 reduces to BCE exactly
- alpha=None skips alpha branch cleanly
- all-zero or all-one targets produce finite non-NaN loss
- shape mismatch raises RuntimeError immediately (PyTorch built-in)
- gamma=-1 raises ValueError (new guard)
- alpha=1.5 raises ValueError (new guard)
- alpha=0.0 raises ValueError (new guard -- boundary is excluded)

Open warnings (non-critical, carried forward):
- Large gamma (>5) silently zeroes gradient on confident predictions -- no guard
- `nn.CrossEntropyLoss()` re-instantiated per epoch in `_run_joint_epoch` (pre-existing, not introduced here)

Style/docstring fixes verified 2026-04-02:
- `src/focal_loss.py`: docstring corrected to `alpha : (0, 1)` (open interval, matching the guard `0 < alpha < 1`)
- `tests/unit/test_focal_loss.py`: `import pytest` moved to module-level top (was inside test functions -- invalid style)

All 61 tests passed on all three QA runs. QA verdict: GREEN (original focal loss commit, guard follow-up, and style/docstring fix pass).
