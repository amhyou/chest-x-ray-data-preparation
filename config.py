import os

# ─── EXPERIMENT CONFIGURATION ─────────────────────────────
NUM_CLASSES = 4   # Change to 4 or 5

# ─── PATHS ────────────────────────────────────────────────
# Change these when switching environments (local / Vast.ai / Kaggle)

# Images the model was TRAINED on (ROI output from B_preprocess.py)
ROI_IMAGE_DIR = f"data_roi_{NUM_CLASSES}class"

# Original raw 384px images — used ONLY for Grad-CAM overlays
RAW_IMAGE_DIR = f"data_384_{NUM_CLASSES}class/images"

# ROI metadata CSV (output of B_preprocess.py)
METADATA_PATH = os.path.join("metadata", f"DATA_ROI_{NUM_CLASSES}CLASS.csv")

# Model checkpoints
CHECKPOINT_DIR = "weights"

# Results / logs
RESULTS_DIR = "results"
LOG_FILE = os.path.join(RESULTS_DIR, "training_log.csv")

# ─── TRAINING HYPERPARAMETERS ─────────────────────────────
BATCH_SIZE = 32
ACCUMULATION_STEPS = 2          # Effective batch = 32 * 2 = 64
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.4
EARLY_STOP_PATIENCE = 7
MAX_GRAD_NORM = 1.0
IMG_SIZE = 384
