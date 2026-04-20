"""
Central configuration for lottery-nn.
Adjust LOTTERY settings to match your target game.
"""

# ---------------------------------------------------------------------------
# Lottery definition
# ---------------------------------------------------------------------------
LOTTERY = {
	"name":        "Lotto Max",
	"id":          0,      # swarm lottery index (0=LottoMax, 1=649, 2=DailyGrand)
	"main_count":  7,      # how many numbers to pick
	"main_max":    52,     # highest main number (updated 2026-04-14: range expanded from 50 to 52)
	"has_bonus":   False,  # LottoMax bonus is drawn by the lottery; players don't select it
}

# A single "play" (ticket) contains this many lines of 7 numbers.
LINES_PER_PLAY = 1
# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_DIR = "data"
RAW_CSV = f"{DATA_DIR}/draws.csv"          # columns: date, n1,n2,n3,n4,n5, bonus
PROCESSED_NPY = f"{DATA_DIR}/processed.npy"

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
SEQUENCE_LEN = 10          # how many past draws to use as context (kept small for limited data)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_DIR = "models"
CHECKPOINT = f"{MODEL_DIR}/best.pt"

EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 3
FF_DIM = 256
DROPOUT = 0.1

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15              # early-stopping patience (epochs)
VAL_SPLIT = 0.15
TEST_SPLIT = 0.05
SEED = 42

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
FOCAL_GAMMA = 2.0   # focal loss modulating exponent; 0 = standard BCE; must be >= 0
FOCAL_ALPHA = 0.25  # positive-class weight in (0, 1); set to None to disable alpha weighting

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
NUM_PLAYS = 1              # how many plays (tickets) to generate
TEMPERATURE = 1.2          # >1 -> more diverse, <1 -> more concentrated
