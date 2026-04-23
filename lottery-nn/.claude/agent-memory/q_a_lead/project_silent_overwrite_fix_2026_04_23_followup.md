---
name: Follow-up QA pass 2026-04-23 (W1/W2/S1/S2/S3 closed)
description: All 5 open items from previous pass closed. One new deferred item: main_swarm.py cmd_log bootstrap pattern (S4). GREEN 111/111.
type: project
---

Follow-up pass 2026-04-23: all W1/W2/S1/S2/S3 items confirmed closed, 111/111 tests pass.

**Why:** W1/W2 were silent-overwrite footguns; S1 was Windows makedirs("") crash; S2 was missing regression tests; S3 was monitor.py column-normalization bypass.

**How to apply:** One open item carried forward as S4:

S4 (DEFERRED): main_swarm.py cmd_log lines 236-243 has the same bootstrap pattern as the W1 fix --
if the CSV is missing it creates a 1-row file. This is intentional for the swarm path (CSVs are
expected to be absent on first log for a new lottery), but creates inconsistency with feedback.py.
User acknowledged and deferred -- do NOT raise as a new critical in future passes unless the
behavior changes. If hardened in a future pass, the fix should add a FileNotFoundError similar
to feedback.py but with a clear docstring explaining first-use semantics.

No new criticals, no regressions.
