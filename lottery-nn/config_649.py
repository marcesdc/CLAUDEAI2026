"""
Configuration for Lotto 6/49.
6 main numbers from 1-49, bonus ball from 1-49.
Draws every Wednesday and Saturday.
"""

LOTTERY = {
    "name":        "Lotto 6/49",
    "id":          1,
    "main_count":  6,
    "main_max":    49,
    "bonus_max":   49,
}

LINES_PER_PLAY = 3

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_DIR  = "data"
RAW_CSV   = "data/draws_649.csv"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_DIR  = "models"
CHECKPOINT = "models/best_649.pt"

EMBED_DIM  = 64
NUM_HEADS  = 4
NUM_LAYERS = 3
FF_DIM     = 256
DROPOUT    = 0.1

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
SEQUENCE_LEN  = 10
BATCH_SIZE    = 32
EPOCHS        = 100
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 15
VAL_SPLIT     = 0.15
TEST_SPLIT    = 0.05
SEED          = 42

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
NUM_PLAYS   = 5
TEMPERATURE = 1.2
