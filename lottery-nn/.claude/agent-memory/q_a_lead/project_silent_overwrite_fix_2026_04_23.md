---
name: silent_overwrite_fix_2026_04_23
description: Incident 2026-04-23 fix: load_draws() hardened against silent synthetic overwrite of real draws.csv -- GREEN 109/109
type: project
---

load_draws() now raises FileNotFoundError instead of silently calling generate_synthetic() when csv_path is missing.
SYNTHETIC_CSV constant added to src/data_loader.py; generate_synthetic() default save_path changed from config.RAW_CSV
to SYNTHETIC_CSV. Two regression guard tests replace old fallback test. All 109 tests pass.

**Why:** On a fresh machine with no draws.csv, the old code silently created data/draws.csv with 2000 synthetic rows.
Any subsequent real-data CSV placed at that path was overwritten without warning.

**How to apply:** Any future change to data_loader.py or generate_synthetic() must not restore the old fallback.
The two regression tests (test_load_draws_raises_when_missing, test_generate_synthetic_default_path_is_safe) are the
canonical guards -- if either is deleted or weakened, that is a RED flag.

Open warnings from this QA run:
- W1: feedback.py::log_draw() bootstrap-creates config.RAW_CSV when missing (intentional but structural cousin of the bug).
- W2: agent/monitor.py::get_last_logged_date() reads RAW_CSV via pd.read_csv() directly, bypassing load_draws() guards.

Open suggestions:
- S1: os.makedirs("", ...) in generate_synthetic() raises unhelpful OS error on bare-filename save_path. Add `or "."` guard.
- S2: Add test_generate_synthetic_bare_filename to pin that edge case behavior.
- S3: Route agent/monitor.py::get_last_logged_date() through load_draws() or _normalize_columns() for consistency.
