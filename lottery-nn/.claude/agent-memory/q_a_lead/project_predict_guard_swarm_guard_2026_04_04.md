---
name: predict_guard_and_swarm_guard_2026_04_04
description: QA 2026-04-04: predict.py bonus_max assert at import + main_swarm.py prune checkpoint guard -- GREEN, 85/85 pass
type: project
---

Two hardening fixes applied 2026-04-04 after prior QA warnings.

**Why:** Both were latent crash paths identified in the codex bugfixes session that day.

Fix 1 -- src/predict.py lines 24-25:
  Assert BONUS_MAX is not None at import time when has_bonus=True.
  Does not fire for current config (has_bonus=False on LottoMax) so no regression risk.
  Correctly placed: module-level, runs once on import, before any function call.

Fix 2 -- main_swarm.py _prune_and_finetune lines 288-290:
  Path(SWARM_CHECKPOINT).exists() guard returns pretrain_best_val early instead of
  raising FileNotFoundError when checkpoint is absent.
  Logically sound: in normal flow the checkpoint always exists because cmd_joint_train
  saves it before calling _prune_and_finetune, so this guard only triggers if the
  file is manually deleted between steps.

**How to apply:** Both paths remain untested by the automated suite (no test exercises
has_bonus=True misconfiguration or missing-checkpoint-during-prune). Open suggestion
carried forward: add targeted unit tests for these two guards.

Test run: 85/85 passed in 1.42s. Verdict GREEN.
