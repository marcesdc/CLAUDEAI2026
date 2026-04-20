# CLAUDEAI2026 — Project Root

Workspace for AI/lottery prediction projects. Python 3.14 + skills.sh workflow are in global rules.

## Active Projects
- `lottery-nn/` — neural-network predictor for LottoMax + 6/49 + Daily Grand (swarm). See `lottery-nn/CLAUDE.md`.
- `lottery-portfolio/` — EV optimization / portfolio sizing across the same three games.

## Swarm Design Principles (from book research 2026-03-24)
- Shared encoder trained jointly on all 3 lotteries via a `lottery_id` embedding.
- Per-lottery output heads (different ball pool sizes).
- Shared swarm state in `lottery-nn/data/swarm_state.json`.
- Actor-Critic play generation: generate candidates -> score -> accept or regenerate.
- Agent weights updated after each scored draw via Thompson sampling.
- Each lottery has only ~100 draws of history — the shared encoder is what makes the small-data setup viable.
