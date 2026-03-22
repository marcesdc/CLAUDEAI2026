"""
Central configuration for lottery-nn.
Adjust LOTTERY settings to match your target game.
"""

# ---------------------------------------------------------------------------
# Lottery definition
# ---------------------------------------------------------------------------
LOTTERY = {
	"name": "Lottery",
	"main_count": 7,       # how many numbers to pick
	"main_max": 50,        # highest main number
	"bonus_max": 50,       # bonus ball range (1 to bonus_max)
}

# A single "play" (ticket) contains this many lines of 7 numbers.
LINES_PER_PLAY = 3
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
# Prediction
# ---------------------------------------------------------------------------
NUM_PLAYS = 5              # how many plays (tickets) to generate
TEMPERATURE = 1.2          # >1 -> more diverse, <1 -> more concentrated
