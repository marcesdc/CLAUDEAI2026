---
name: Codex bug-fix batch 2026-04-23b
description: Five Codex fixes: H2a/H2b short-history guards, H1 feedback silent-fallback removal, M1 cmd_log numbers guard, M2 train.py empty-dirname guard -- GREEN 120/120
type: project
---

Five Codex fixes landed 2026-04-23 (second batch, distinct from the load_draws incident fix).

Fixes:
- H2a: preprocessing.build_features raises ValueError (N <= SEQUENCE_LEN); split() raises when N < 3 or n_train <= 0.
- H2b: preprocessing_swarm.build_features raises ValueError (N <= seq_len); get_last_window raises ValueError (< seq_len draws).
- H1: feedback.score_last_prediction no longer falls back silently to the most-recent pred when draw_date has no match -- returns empty DataFrame and prints message.
- M1: main.cmd_log checks args.numbers is None and calls sys.exit() with clear message before any downstream call.
- M2: train.py guards os.makedirs with `if ckpt_dir:` so bare filename checkpoints (e.g. best.pt) don't raise WinError 3.

Test count: 120/120 passed (was 113 before this batch; +7 new tests).

**Why:** Codex external review identified silent-failure paths that would be hard to diagnose in production (shape-mismatch stack traces, corrupted score_log, Windows directory error).

**How to apply:** All five guards now have direct test coverage. The H1 omit-draw_date path (no arg -> uses last row of pred_log) is intentional and still safe -- it only fires when caller omits draw_date AND the last saved pred_date matches, which is the common happy path.

Open items: None -- all five fixes verified GREEN.
