# CLAUDEAI2026 — Project Root

Workspace for AI/lottery prediction projects. Python 3.14 + skills.sh workflow are in global rules.

## Active Projects
- `lottery-nn/` — neural-network predictor for LottoMax + 6/49 + Daily Grand (swarm). See `lottery-nn/CLAUDE.md`.
- `lottery-portfolio/` — ticket-portfolio EV optimizer across the same three games. See `lottery-portfolio/CLAUDE.md`.

## Cross-cutting principle

Each lottery has ~100-140 draws of history. Both projects rely on cross-game information sharing (a shared encoder in `lottery-nn`; literature-derived priors in `lottery-portfolio`) to compensate for the small-data setup.
