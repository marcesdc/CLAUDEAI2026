# CLAUDE.md

PyTorch swarm predictor for OLG Lotto Max + 6/49 + Daily Grand. Python 3.14 setup is in global rules.

## Commands

```bash
pip install -r requirements.txt

# Single-lottery (LottoMax) -- main.py
C:\Python314\python.exe main.py {data|train|predict|evaluate}
C:\Python314\python.exe main.py log --date YYYY-MM-DD --numbers n1 .. n7 [--no-retrain]

# Swarm (all 3 lotteries) -- main_swarm.py
C:\Python314\python.exe main_swarm.py joint-train [--prune] [--prune-percent 0.65]
C:\Python314\python.exe main_swarm.py predict --lottery {lottomax|649|dailygrand}
C:\Python314\python.exe main_swarm.py log --lottery <name> --date YYYY-MM-DD --numbers ... [--bonus N]
C:\Python314\python.exe main_swarm.py status
```

`--bonus` only applies to Daily Grand (player-picked Grand 1-7). LottoMax/6/49 bonus is machine-drawn and not part of plays.
**After each real draw:** `log` → optional `joint-train` → `predict`. Full retrain only when adding a batch of historical rows.

## Per-lottery shape (single source of truth: `src/preprocessing_swarm.py::LOTTERY_CONFIGS`)

| Lottery     | main_count | main_max | has_bonus | bonus_col |
|-------------|------------|----------|-----------|-----------|
| lottomax    | 7          | 52       | False     | -         |
| 649         | 6          | 49       | False     | -         |
| dailygrand  | 5          | 49       | True (1-7)| `grand`   |

Other knobs in `config.py`: `LINES_PER_PLAY`, `SEQUENCE_LEN`, `EPOCHS`/`PATIENCE`, `TEMPERATURE`, `NUM_PLAYS`, `FOCAL_GAMMA`, `FOCAL_ALPHA`.

## Data

CSVs under `data/` — `date` is any pandas-parseable format:
- `draws.csv` (LottoMax): `date, n1..n7`
- `draws_649.csv`: `date, n1..n6`
- `draws_dailygrand.csv`: `date, n1..n5, grand`

`data_loader._normalize_columns()` renames legacy `number1..number7` to `n1..n7` on load. Without these files, `main.py data` generates synthetic draws.

`data/swarm_state.json` is auto-updated by every `joint-train` and `log`. `data/predictions_log.csv` + `data/score_log.csv` track predictions and hit scores.

## Architecture (high-level — read `src/model_swarm.py` for details)

- `SharedLotteryTransformer` — Pre-LN backbone + 3 per-lottery head pairs (~253K params). Inputs padded to `POOL_MAX=52` (2x52=104 features/timestep). `lottery_id` embedding (0/1/2) added every timestep. Round-robin batches per epoch.
- Loss: focal loss for main numbers (`src/focal_loss.py`); CrossEntropyLoss × 0.3 for bonus head (Daily Grand only).
- Single-lottery alternative: `LotteryTransformer` / `LotteryLSTM` in `src/model.py`.

## Active gotchas (don't get burned twice)

- **OLG page needs JavaScript** — plain `WebFetch` returns empty; gateway APIs need auth. Use Playwright MCP (`agent/run.py monitor|watch`).
- **Daily Grand bonus column is `grand`** in the CSV; `load_lottery_df()` renames it internally — do not rename in CSV.
- **PyTorch nested-tensor warning on swarm train** (`enable_nested_tensor is True, but ... norm_first was True`) — non-fatal Pre-LN noise. Ignore.
- **Cleaning scripts must `assert len(df) == expected` before writing CSVs** — a regex cleaner once wiped `draws_649.csv` to header-only with no backup.
- **LottoMax pool is 1-52 (2026 rule change), not 1-50** — `C(52,7) = 133,784,560`. Old hardcoded `50`/`43` are stale.

## Phase status

Phase 2 in progress (focal loss + bandit + QA team done; Actor-Critic + Reflexion + Thompson-sampling agent weights still pending). Check `git log` for current state. Phase 3 (DeepAR / probabilistic LSTM / N-BEATS / Pyro) not started.

## QA Agents — MANDATORY

`.claude/agents/`: `q_a_lead` (orchestrator) → `q_a1` (review + tests) + `q_a2` (perf + edge cases) in parallel, returns GREEN/RED.

Claude must invoke `@q_a_lead` automatically after every code change to `src/`, root scripts, `tests/`, or `config.py`, and before declaring the task done. Do not report complete until q_a_lead returns GREEN.

```
@q_a_lead I just changed <one-line description>
```

Manual test runs: `C:\Python314\python.exe -m pytest tests/ -v --tb=short` (also `tests/unit/`, `tests/integration/`, or `--cov=src --cov-report=term-missing`). Tests use synthetic fixtures from `tests/conftest.py` and run on CPU only.

## Agent Interface

```bash
C:\Python314\python.exe agent/run.py {chat|monitor|watch}
```

`monitor`/`watch` use `@playwright/mcp@latest` to render the OLG JS page and auto-log new draws. Deps: `pip install claude-agent-sdk anyio` + Node/`npx`.
