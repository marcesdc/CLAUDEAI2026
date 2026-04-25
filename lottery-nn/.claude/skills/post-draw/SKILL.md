---
name: post-draw
description: Canonical post-draw workflow (log -> optional retrain -> predict). Use when the user has a new real draw result and wants to record it and refresh the swarm's predictions in one shot.
disable-model-invocation: true
---

# /post-draw -- canonical post-draw workflow

CLAUDE.md prescribes the post-draw sequence as: `log` -> optional `joint-train` -> `predict`. This skill bundles all three so the user can run them with one command.

## Inputs to gather (ask the user once, then proceed end-to-end)

- `lottery`   : `lottomax` | `649` | `dailygrand`
- `date`      : `YYYY-MM-DD` (default to today if omitted)
- `numbers`   : the drawn main numbers
   - LottoMax  -> 7 ints in 1..52
   - 6/49      -> 6 ints in 1..49
   - DailyGrand -> 5 ints in 1..49
- `bonus`     : Daily Grand only -- the player-picked Grand 1..7
- `retrain?`  : default **No**. Only retrain when the user explicitly says so or is adding a batch of historical rows.

If anything is missing, ask once before running anything.

## Steps

1. Append the draw to history:

   ```bash
   C:\Python314\python.exe main_swarm.py log --lottery <lottery> --date <date> --numbers <n1> <n2> ... [--bonus N]
   ```

2. If the user requested a retrain:

   ```bash
   C:\Python314\python.exe main_swarm.py joint-train
   ```

3. Print fresh predictions for that lottery:

   ```bash
   C:\Python314\python.exe main_swarm.py predict --lottery <lottery>
   ```

4. Briefly summarise: hits scored vs the new draw (printed by `log`), whether a retrain ran, and the top predicted lines.

## Guardrails

- Default to **No** retrain. Per CLAUDE.md: "Full retrain only when adding a batch of historical rows."
- For LottoMax / 6/49 do NOT pass `--bonus` -- bonus is machine-drawn and not part of plays.
- For Daily Grand `--bonus` is mandatory and is the player-picked Grand 1..7.
- Use `C:\Python314\python.exe` -- never `python` (resolves to 3.12 with no torch).
- `data/draws*.csv` is now write-protected by the `guard_csv_writes` PreToolUse hook. The `log` subcommand goes through Python's `pandas.to_csv` (not Claude's Edit/Write tool), so it bypasses the hook cleanly.

## When to skip this skill

- The user only wants to *predict* without logging a new draw -> use `predict-plays` skill instead.
- The user only wants to *log* without re-predicting -> use `log-draw` skill or `--no-retrain`.
- The user is bulk-loading historical draws -> use `agent/run.py monitor` (Playwright) instead.
