"""
Neural network architectures for lottery prediction.

Two models are provided:

LotteryTransformer
    Transformer encoder over the draw-history sequence, followed by two
    independent heads: one for the main numbers (multi-label BCELoss) and
    an optional bonus head (cross-entropy).

LotteryLSTM
    Simpler LSTM baseline for comparison.

Both classes share the same forward / predict interface.
"""

import math
import torch
import torch.nn as nn

import config


MAIN_MAX = config.LOTTERY["main_max"]
BONUS_MAX = config.LOTTERY["bonus_max"]
SEQ_LEN = config.SEQUENCE_LEN
INPUT_DIM = 2 * MAIN_MAX     # multi-hot + rolling-freq (see preprocessing)


# ---------------------------------------------------------------------------
# Transformer model
# ---------------------------------------------------------------------------

class LotteryTransformer(nn.Module):
    """
    Transformer encoder that reads a sequence of augmented multi-hot draw
    vectors and predicts the probability of each number appearing next.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        embed_dim: int = config.EMBED_DIM,
        num_heads: int = config.NUM_HEADS,
        num_layers: int = config.NUM_LAYERS,
        ff_dim: int = config.FF_DIM,
        dropout: float = config.DROPOUT,
        main_max: int = MAIN_MAX,
        bonus_max: int = BONUS_MAX,
        has_bonus: bool = True,
    ):
        super().__init__()
        self.has_bonus = has_bonus

        # Project raw features → embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, SEQ_LEN, embed_dim))
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

        # Aggregate sequence → single vector
        self.pool = _AttentionPool(embed_dim)

        # Output heads
        self.main_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, main_max),
        )
        if has_bonus:
            self.bonus_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, bonus_max),
            )

    def forward(self, x: torch.Tensor):
        """
        x : (B, SEQ_LEN, INPUT_DIM)
        returns:
            main_logits  : (B, main_max)
            bonus_logits : (B, bonus_max) | None
        """
        h = self.input_proj(x) + self.pos_embed          # (B, T, E)
        h = self.encoder(h)                               # (B, T, E)
        h = self.pool(h)                                  # (B, E)

        main_logits = self.main_head(h)
        bonus_logits = self.bonus_head(h) if self.has_bonus else None
        return main_logits, bonus_logits


# ---------------------------------------------------------------------------
# LSTM baseline
# ---------------------------------------------------------------------------

class LotteryLSTM(nn.Module):
    """Bidirectional LSTM baseline."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden: int = 256,
        num_layers: int = 2,
        dropout: float = config.DROPOUT,
        main_max: int = MAIN_MAX,
        bonus_max: int = BONUS_MAX,
        has_bonus: bool = True,
    ):
        super().__init__()
        self.has_bonus = has_bonus

        self.lstm = nn.LSTM(
            input_dim, hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        enc_dim = hidden * 2   # bidirectional

        self.main_head = nn.Sequential(
            nn.LayerNorm(enc_dim),
            nn.Linear(enc_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, main_max),
        )
        if has_bonus:
            self.bonus_head = nn.Sequential(
                nn.LayerNorm(enc_dim),
                nn.Linear(enc_dim, bonus_max),
            )

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x)           # (B, T, 2*H)
        h = out[:, -1, :]               # last step
        main_logits = self.main_head(h)
        bonus_logits = self.bonus_head(h) if self.has_bonus else None
        return main_logits, bonus_logits


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

class _AttentionPool(nn.Module):
    """Soft attention pooling over the sequence dimension."""

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        w = torch.softmax(self.attn(x), dim=1)   # (B, T, 1)
        return (x * w).sum(dim=1)                  # (B, D)


def build_model(arch: str = "transformer", has_bonus: bool = True) -> nn.Module:
    """Factory function – use 'transformer' or 'lstm'."""
    if arch == "transformer":
        return LotteryTransformer(has_bonus=has_bonus)
    elif arch == "lstm":
        return LotteryLSTM(has_bonus=has_bonus)
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose 'transformer' or 'lstm'.")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
