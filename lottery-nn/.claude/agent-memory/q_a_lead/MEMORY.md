# QA Lead Memory Index

- [project_focal_loss.md](project_focal_loss.md) -- Focal loss added 2026-04-02: config keys, edge cases, GREEN verdict, open warnings
- [project_code_quality_batch_2026_04_03.md](project_code_quality_batch_2026_04_03.md) -- Code quality batch 2026-04-03: utils.py extraction, vectorized preprocessing, path guard -- GREEN
- [project_pruning_lth_2026_04_04.md](project_pruning_lth_2026_04_04.md) -- LTH pruning 2026-04-04: src/pruning.py + main_swarm.py --prune flag -- RED (1 test fail, non-ASCII in docstrings)
- [project_codex_bugfixes_2026_04_04.md](project_codex_bugfixes_2026_04_04.md) -- Codex bug fixes 2026-04-04: predict.py bonus_max, prune threshold, masks-per-step, path guard -- GREEN
- [project_predict_guard_swarm_guard_2026_04_04.md](project_predict_guard_swarm_guard_2026_04_04.md) -- predict.py bonus_max assert + main_swarm.py prune checkpoint guard -- GREEN 85/85
- [project_cmd_log_guards_2026_04_20.md](project_cmd_log_guards_2026_04_20.md) -- cmd_log() uniqueness + bonus-rejection guards, lottomax in log choices -- GREEN 91/91, S3 open (add cmd_log unit tests)
- [project_cmd_log_unit_tests_2026_04_20.md](project_cmd_log_unit_tests_2026_04_20.md) -- 15 new cmd_log unit tests + redirect_log_paths fixture -- GREEN 106/106, fixture safe, S1/S2 open (boundary values)
- [project_cmd_log_boundary_tests_2026_04_20.md](project_cmd_log_boundary_tests_2026_04_20.md) -- S1/S2 closed: iso-extended date + number-zero rejection tests, fixture docstring updated -- GREEN 108/108
- [project_silent_overwrite_fix_2026_04_23.md](project_silent_overwrite_fix_2026_04_23.md) -- Incident 2026-04-23 fix: load_draws() raises FileNotFoundError, SYNTHETIC_CSV constant -- GREEN 109/109, W1/W2/S1-S3 open
- [project_silent_overwrite_fix_2026_04_23_followup.md](project_silent_overwrite_fix_2026_04_23_followup.md) -- Follow-up 2026-04-23: W1/W2/S1/S2/S3 all closed, GREEN 111/111, S4 deferred (main_swarm.py bootstrap, intentional first-use)
- [project_cmd_log_bootstrap_final_2026_04_23.md](project_cmd_log_bootstrap_final_2026_04_23.md) -- Final pass 2026-04-23: S4 closed + monitor.py comment, all write-sites audited -- GREEN 112/112, NO open items
