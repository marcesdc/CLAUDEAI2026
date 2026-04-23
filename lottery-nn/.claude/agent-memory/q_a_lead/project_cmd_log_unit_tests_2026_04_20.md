---
name: cmd_log unit tests 2026-04-20
description: 15 new unit tests for cmd_log() validation added to test_swarm_guards.py -- GREEN verdict, 106/106 passing
type: project
---

Change landed 2026-04-20. 15 new tests + 1 new fixture added to tests/unit/test_swarm_guards.py:
- redirect_log_paths fixture monkeypatches LOTTERY_CONFIGS[name]["csv"], SWARM_STATE_FILE, SWARM_PRED_LOG to tmp_path
- 11 rejection tests (count, dup x2, bonus-on-no-bonus x4, range, bonus missing/0/above-max for dailygrand, invalid date)
- 4 accept tests (lottomax, 649, dailygrand with grand col, duplicate-date skip)

Verdict: GREEN -- 106/106 tests passing (17/17 in this file).

Fixture safety confirmed:
- monkeypatch.setitem(LOTTERY_CONFIGS[name], "csv", ...) targets the inner per-lottery dict directly
- LOTTERY_CONFIGS imported in test file and main_swarm.py are the SAME dict object (id-identical)
- pytest monkeypatch reverts setitem after each test -- no cross-test leakage
- All file writes in cmd_log path (csv_path, SWARM_STATE_FILE) are fully redirected to tmp_path
- SWARM_PRED_LOG is read-only in cmd_log path (_score_swarm_prediction reads it; no write path from cmd_log)
- No artifacts written outside tmp_path

Open warnings:
- W1: xdist parallelism would create a LOTTERY_CONFIGS dict-mutation race; not installed so not a current risk
- W2: Lower boundary n=0 and ISO-extended date "2026-04-20T00:00:00" not covered in rejection tests

Open suggestions:
- S1: Add test_cmd_log_rejects_number_zero (n=0) to cover lower bound of range check
- S2: Add test_cmd_log_rejects_iso_extended_date ("2026-04-20T00:00:00") for completeness
- S3: Add xdist incompatibility note to fixture docstring if parallel test execution ever considered
- S4: Close open S3 from 2026-04-20 cmd_log guards memory (now resolved -- tests exist)

**Why:** Prior QA identified zero direct test coverage for cmd_log guards; this batch closes that gap.
**How to apply:** S3 (add cmd_log tests) is now resolved. Remaining open items are S1/S2 (boundary values).
