---
name: project_codex_bugfixes_2026_04_04
description: Four Codex-review bug fixes 2026-04-04: predict.py bonus_max, _prune_and_finetune threshold, _run_joint_epoch masks, data_loader path guard. GREEN verdict.
type: project
---

Four bugs fixed on 2026-04-04. All 85 tests pass. Verdict: GREEN.

**Fix 1 -- src/predict.py line 23:**
`config.LOTTERY["bonus_max"]` changed to `config.LOTTERY.get("bonus_max")`.
BONUS_MAX is now None for LottoMax (has_bonus=False). Safe because the
bonus_probs guard on line 79 (`if bonus_probs is not None`) fires before
`rng.choice(BONUS_MAX, ...)` is reached. Model returns bonus_logits=None
when has_bonus=False, so bonus_probs stays None.
Residual warning: if a caller somehow passes a has_bonus=True model with
a config that omits bonus_max, BONUS_MAX=None reaches rng.choice and crashes.
Config currently cannot produce this state, but no assert guards it.

**Fix 2 -- main_swarm.py _prune_and_finetune:**
Signature gained `pretrain_best_val=float("inf")` param. `best_val` now
initialises to `pretrain_best_val` instead of float("inf"). Pruned subnetwork
only overwrites checkpoint if it genuinely improves on pre-prune best_val.
cmd_joint_train passes `best_val` as the new arg (line 144-146).
Correct and complete. No issues.

**Fix 3 -- main_swarm.py _run_joint_epoch:**
Gained `masks=None` param. `apply_masks(model, masks)` called after each
`optimizer.step()` inside the per-batch loop when masks is not None. Lazy
import (`from src.pruning import apply_masks`) placed inside the branch so
non-pruning paths incur no import overhead. The redundant standalone
apply_masks call that was previously placed after the epoch in
_prune_and_finetune was removed. Correct placement -- masks are applied
per-step, not per-epoch.
val epoch call: `_run_joint_epoch(model, val_loaders, optimizer=None,
train=False)` passes no masks, which is correct (masks only needed on train).

**Fix 4 -- src/data_loader.py:**
Removed _PROJECT_ROOT variable and the is_relative_to() path-restriction
check that was blocking user-supplied absolute paths (e.g., C:\Users\...\draws.xlsx).
No replacement security check was added. This is intentional -- the file is
used in a local single-user context. No issues.

**Open warnings (new from this batch):**
- predict.py: BONUS_MAX=None is not asserted before use. If config has_bonus=True
  but bonus_max is absent, rng.choice(None, ...) raises TypeError with no message.
  Suggestion: add `assert BONUS_MAX is not None` before line 80, gated on a
  `config.LOTTERY.get("has_bonus", True)` check.

**How to apply:** When reviewing future predict.py changes, verify the
bonus_probs / BONUS_MAX None-safety chain holds. When reviewing _run_joint_epoch
changes, verify masks arg is threaded correctly to all call sites (train=True
only).
