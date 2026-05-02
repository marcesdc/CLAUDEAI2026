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

`--bonus` is for Daily Grand only (player-picked Grand 1-7). LottoMax/6/49 bonus is machine-drawn.
`--force` on `log` overwrites an existing date in place (1:1 row replace; 2026-04-23 missing-CSV guard still wins).
**After each real draw:** `log` → optional `joint-train` → `predict`. Full retrain only when adding a batch of historical rows.

Per-lottery shape lives in `src/preprocessing_swarm.py::LOTTERY_CONFIGS` (main_count / main_max / has_bonus / bonus_col). Hyperparameters + feature flags live in `config.py`.

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

## Active gotchas

- **OLG page needs JavaScript.** Plain `WebFetch` returns empty; use Playwright MCP via `agent/run.py monitor|watch`.
- **Daily Grand bonus CSV column is `grand`** (renamed to `bonus` in-memory by `load_lottery_df()`). Don't rename in the CSV.
- **PyTorch `enable_nested_tensor` warning on swarm train** — non-fatal Pre-LN noise; ignore.
- **Never silently overwrite `data/draws*.csv`.** Clean/regen scripts must assert row count before writing (incident 2026-04-23).
- **LottoMax pool is 1-52 since the 2026 rule change** — `C(52,7) = 133,784,560`. Any hardcoded `50` is stale.
- **Checkpoint metadata sidecar** (`models/best.meta.json` / `models/best_swarm.meta.json`) — written on save, asserted on load. Pool/head-size mismatch raises `RuntimeError`. Missing sidecar = legacy checkpoint, warn-only.

## Phase / state

Current state lives in `git log`, `data/swarm_state.json`, and the project memory at `~/.claude/projects/d--AI---2026-CLAUDEAI2026/memory/`. Don't re-document it here.

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
