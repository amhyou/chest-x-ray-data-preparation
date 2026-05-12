import os

# ─── EXPERIMENT CONFIGURATION ─────────────────────────────
NUM_CLASSES = 2   # Binary: Normal vs Effusion
NUM_CLASSES = 4   # Multi-class: Atelectasis, Cardiomegaly, Effusion, Normal

# ─── PATHS ────────────────────────────────────────────────
# Change these when switching environments (local / Vast.ai / Kaggle)

# Images the model was TRAINED on (ROI output from B_preprocess.py)
ROI_IMAGE_DIR = "data_roi_4class"

# Original raw 384px images — used ONLY for Grad-CAM overlays
RAW_IMAGE_DIR = "data_384_4class/images"

# ROI metadata CSV (output of B_preprocess.py)
METADATA_PATH = os.path.join("metadata", "DATA_ROI_4CLASS.csv")

# Model checkpoints
CHECKPOINT_DIR = "weights"

# Results / logs
RESULTS_DIR = "results"
LOG_FILE = os.path.join(RESULTS_DIR, "training_log.csv")

# ─── TRAINING HYPERPARAMETERS ─────────────────────────────
BATCH_SIZE = 8
ACCUMULATION_STEPS = 8          # Effective batch = 64 (Fits Swin-Base on 12GB VRAM)
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.4
EARLY_STOP_PATIENCE = 7
MAX_GRAD_NORM = 1.0
IMG_SIZE = 384
