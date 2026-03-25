# CLAUDEAI2026 — Project Root

This is the root workspace for all AI/lottery prediction projects.

## Active Projects
- `lottery-nn/` — Lotto Max neural network predictor (main project, see `lottery-nn/CLAUDE.md`)
- Future: `lottery-649/` — Lotto 6/49 agent
- Future: `lottery-dailygrand/` — Daily Grand agent

## Python Environment
- Always use `C:\Python314\python.exe`
- Never use `python` (resolves to 3.12, missing torch and other deps)

## Architecture Direction
Multi-agent swarm system: three per-lottery agents sharing a single encoder backbone.
See `lottery-nn/CLAUDE.md` for full architecture details.

## Swarm Design Principles (from book research 2026-03-24)
- Shared encoder trained jointly on all 3 lotteries (lottery_id embedding)
- Per-lottery output heads (different ball pool sizes)
- Shared swarm state: `data/swarm_state.json`
- Actor-Critic play generation: generate candidates → score → accept or regenerate
- Agent weights updated after each scored draw via Thompson sampling
- All three lotteries have ~1 year of data only (~100 draws each) — shared encoder is critical

## Skills
Browse https://skills.sh/ at the start of each new sub-project.
User runs: `npx skills add <owner/repo>`
