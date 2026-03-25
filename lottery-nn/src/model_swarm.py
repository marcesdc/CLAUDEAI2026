"""
Shared encoder for the multi-lottery swarm.

Architecture
------------
One transformer backbone trained jointly on all 3 lotteries.
A lottery_id embedding conditions the model on which game is being played.
Per-lottery output heads handle different pool and bonus sizes.

Lottery IDs
-----------
  0 = Lotto Max    (7 from 50, bonus 1-50)
  1 = Lotto 6/49   (6 from 49, bonus 1-49)
  2 = Daily Grand  (5 from 49, grand  1-7)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Constants shared across all lotteries
# ---------------------------------------------------------------------------
POOL_MAX       = 50          # largest pool size (LottoMax uses 1-50)
SEQ_LEN        = 10          # draw history window
INPUT_DIM      = 2 * POOL_MAX   # multi-hot + rolling-freq, padded to POOL_MAX
NUM_LOTTERIES  = 3

# Per-lottery output head sizes  [LottoMax, 6/49, DailyGrand]
MAIN_HEAD_SIZES  = [50, 49, 49]
BONUS_HEAD_SIZES = [50, 49,  7]


# ---------------------------------------------------------------------------
# Shared encoder model
# ---------------------------------------------------------------------------

class SharedLotteryTransformer(nn.Module):
    """
    Transformer encoder shared across all three lotteries.

    All lotteries are projected to the same INPUT_DIM feature space.
    A lottery_id embedding is added to each timestep so the backbone
    knows which game it is processing.
    Separate output heads are used for main numbers and bonus per lottery.

    Forward convention: all samples in a batch must belong to the SAME lottery.
    """

    def __init__(
        self,
        embed_dim:  int = 64,
        num_heads:  int = 4,
        num_layers: int = 3,
        ff_dim:     int = 256,
        dropout:    float = 0.1,
    ):
        super().__init__()

        # --- Shared backbone ---
        self.input_proj    = nn.Linear(INPUT_DIM, embed_dim)
        self.lottery_embed = nn.Embedding(NUM_LOTTERIES, embed_dim)
        self.pos_embed     = nn.Parameter(torch.zeros(1, SEQ_LEN, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool    = _AttentionPool(embed_dim)

        # --- Per-lottery heads ---
        self.main_heads = nn.ModuleList([
            _make_head(embed_dim, ff_dim, dropout, size)
            for size in MAIN_HEAD_SIZES
        ])
        self.bonus_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, size),
            )
            for size in BONUS_HEAD_SIZES
        ])

    def forward(self, x: torch.Tensor, lottery_id: int):
        """
        Parameters
        ----------
        x          : (B, SEQ_LEN, INPUT_DIM)
        lottery_id : int  0=LottoMax  1=6/49  2=DailyGrand
                     All samples in the batch must be from the same lottery.

        Returns
        -------
        main_logits  : (B, MAIN_HEAD_SIZES[lottery_id])
        bonus_logits : (B, BONUS_HEAD_SIZES[lottery_id])
        """
        lid_t = torch.tensor(lottery_id, device=x.device)

        h = self.input_proj(x)                                    # (B, T, E)
        h = h + self.lottery_embed(lid_t).view(1, 1, -1)          # broadcast
        h = h + self.pos_embed                                     # positional
        h = self.encoder(h)                                        # (B, T, E)
        h = self.pool(h)                                           # (B, E)

        main_logits  = self.main_heads[lottery_id](h)
        bonus_logits = self.bonus_heads[lottery_id](h)
        return main_logits, bonus_logits


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

class _AttentionPool(nn.Module):
    """Soft attention pooling over the sequence dimension."""

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.attn(x), dim=1)   # (B, T, 1)
        return (x * w).sum(dim=1)                  # (B, E)


def _make_head(embed_dim: int, ff_dim: int, dropout: float, output_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, ff_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(ff_dim, output_size),
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
