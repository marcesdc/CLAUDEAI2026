---
name: Final QA pass 2026-04-23 (S4 closed, monitor.py lazy-import comment)
description: S4/W1-deferred cmd_log bootstrap hardened + monitor.py comment. All items closed. GREEN 112/112.
type: project
---

Final QA pass 2026-04-23: S4 (cmd_log bootstrap) and S1-new (monitor.py lazy-import) both closed.
112/112 tests pass. No open items.

**Why:** S4 was the last remaining silent-overwrite risk -- cmd_log would silently create a 1-row
CSV if the file was missing. Now replaced with sys.exit(1) + echo-recipe message. The S1-new item
was a missing explanation for the lazy import in monitor.py::get_last_logged_date() that could have
caused a future developer to promote it to top-level and break the Playwright subprocess path.

**How to apply:**

Hardening pattern confirmed across all write sites:
- main_swarm.py cmd_log: FileNotFoundError -> sys.exit(1) + echo-recipe (line 236-243)
- src/feedback.py log_draw: raises FileNotFoundError (line 57-62)
- Both guarded by assert len(df_new) > len(df) append guard
- _save_swarm_prediction / _save_prediction: use "if Path.exists() -> concat else fresh" (safe --
  these are append-only prediction logs, not real-data history files)
- src/data_loader.py generate_synthetic: intentional -- only called explicitly from main.py data cmd

Test infrastructure:
- _HEADERS dict in test_swarm_guards.py verified to match cmd_log row_cols output exactly:
    lottomax:   date,n1,n2,n3,n4,n5,n6,n7
    649:        date,n1,n2,n3,n4,n5,n6
    dailygrand: date,n1,n2,n3,n4,n5,grand
- _seed_header() helper writes header-only CSV before each happy-path test
- test_cmd_log_rejects_missing_csv regression guard: verifies sys.exit(1), "refusing to bootstrap"
  in output, AND file NOT created -- reverting the fix would break this test

No open items carried forward.
