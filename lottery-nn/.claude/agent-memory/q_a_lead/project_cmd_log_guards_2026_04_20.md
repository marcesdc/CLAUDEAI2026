---
name: cmd_log validation guards 2026-04-20
description: main_swarm.py cmd_log() -- uniqueness check, bonus rejection for no-bonus lotteries, lottomax added to log choices -- GREEN verdict
type: project
---

Change landed 2026-04-20. cmd_log() now validates:
- Duplicate numbers rejected (len(set(args.numbers)) != main_count)
- --bonus on no-bonus lottery (LottoMax, 649) rejected with clear OLG explanation
- LottoMax added to p_log choices (was previously 649/dailygrand only)
- Docstring updated to reflect all-three-lottery log support

Verdict: GREEN -- 91/91 tests passing.

Open warnings after this change:
- W1: --bonus 0 on no-bonus lottery produces bonus-rejection message rather than value-error message (correct outcome, minor UX seam)
- W2: Validation ordering fires duplicate check before bonus check -- user may not see both errors simultaneously when both are wrong
- W3: Docstring Examples block only shows dailygrand log example; no LottoMax example

Open suggestions:
- S3 (priority): Add unit tests for cmd_log validation in test_swarm_guards.py -- currently zero direct test coverage for new guards

**Why:** Functionality confirmed correct by manual test runs; test gap is the main follow-up.
**How to apply:** If user adds cmd_log tests, check against the 8 cases listed in S3 in the QA report.
