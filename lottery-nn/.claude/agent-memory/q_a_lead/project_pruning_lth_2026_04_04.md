---
name: project_pruning_lth_2026_04_04
description: LTH magnitude pruning 2026-04-04: src/pruning.py + main_swarm.py --prune flag. RED->GREEN after two post-RED fixes on 2026-04-04.
type: project
---

LTH-style magnitude pruning added on 2026-04-04 (src/pruning.py, tests/unit/test_pruning.py,
main_swarm.py --prune/--prune-percent/--prune-epochs/--prune-patience args).

**Final status (2026-04-04 re-run):** GREEN -- 85/85 tests pass.
Actual test count: 85 total (16 pruning tests, not 17 as originally reported -- the RED report overcounted by 1).

**Fixes applied after RED verdict:**
1. src/pruning.py: early return for percent==0.0 -- returns all-ones masks, skips threshold
   computation entirely. Fixes test_prune_zero_percent_keeps_all.
2. main_swarm.py lines 2, 20, 574: UTF-8 em-dashes (0xE2 0x80 0x94) replaced with ` -- `
   (ASCII double-hyphen). Verified with binary scan -- no em-dash bytes remain.

**Why:** percent=0.0 cutoff_idx=0 caused the minimum-magnitude weight to be zeroed (mask uses
abs > cutoff, not >=). Early return is cleaner than changing the comparison operator. em-dashes
would crash Windows cp1252 terminal on --help output.

**Open warnings (carried from RED, still unresolved):**
- _prune_and_finetune has no try/except around torch.load(); FileNotFoundError possible if
  initial training produced no checkpoint. Low probability but uncaught.

**How to apply:** When reviewing pruning-related PRs, check percent=0.0 boundary condition and
ASCII compliance of any new docstrings/argparse descriptions added to main_swarm.py.
