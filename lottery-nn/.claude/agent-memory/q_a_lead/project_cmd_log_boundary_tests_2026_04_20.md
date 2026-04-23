---
name: project_cmd_log_boundary_tests_2026_04_20
description: Two boundary tests added 2026-04-20 to close S1/S2 from prior QA session -- GREEN 108/108
type: project
---

Two low-priority follow-up tests added to tests/unit/test_swarm_guards.py (no production code changed):

- test_cmd_log_rejects_iso_extended_date: confirms strptime("%Y-%m-%d") rejects "2026-04-20T00:00:00" (ISO datetime with time component)
- test_cmd_log_rejects_number_zero: confirms lower-bound check rejects 0 (below 1..main_max range)
- redirect_log_paths fixture docstring expanded to document pytest-xdist incompatibility (monkeypatch on shared LOTTERY_CONFIGS dict not safe under parallel workers)

Both new tests PASSED. Full suite 108/108 in 2.65s on Python 3.14 / Windows / CPU.

**Why:** Closed the open S1 (lower-bound zero) and S2 (ISO datetime) suggestions from the 2026-04-20 cmd_log unit test session.

**How to apply:** S1 and S2 are now closed. No open suggestions remain for cmd_log input guards.
