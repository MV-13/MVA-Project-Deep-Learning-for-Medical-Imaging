import os

# ─── Chemins ───────────────────────────────────────────────────
TRAIN_H5 = "data/train.h5"
VAL_H5 = "data/val.h5"
TEST_H5 = "data/test.h5"

CHECKPOINT_DIR = "checkpoints"
CACHE_DIR = "cache"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
SWA_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "swa_model.pth")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
OUTPUT_CSV = "submission.csv"

SEED = 42

# ─── Backbone ─────────────────────────────────────────────────
# Phikon-v2 : ViT-L pré-entraîné sur 450M images d'histopathologie
# Produit des features 1024-dim
BACKBONE = "owkin/phikon-v2"
BACKBONE_DIM = 1024
INPUT_SIZE = (224, 224)

# DINOv2 ViT-L comme 2e backbone pour un ensemble
# Phikon capture le domaine histopath, DINOv2 capture la structure générale
USE_DINOV2_ENSEMBLE = True
DINOV2_BACKBONE = "dinov2_vitl14"
DINOV2_DIM = 1024
# Total si ensemble : 1024 + 1024 = 2048

# ─── Feature extraction ───────────────────────────────────────
N_AUG_PASSES = 5
USE_FEATURE_CACHE = True

# ─── Stain augmentation ───────────────────────────────────────
USE_STAIN_AUGMENTATION = True
COLOR_JITTER_BRIGHTNESS = 0.3
COLOR_JITTER_CONTRAST = 0.3
COLOR_JITTER_SATURATION = 0.4
COLOR_JITTER_HUE = 0.15

# ─── Classifieur ──────────────────────────────────────────────
HIDDEN_DIMS = [256]
DROPOUT = 0.3

# ─── Entraînement ─────────────────────────────────────────────
BATCH_SIZE = 128
NUM_WORKERS = 4

LR = 1e-3
WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.05

# Mixup léger
USE_MIXUP = True
MIXUP_ALPHA = 0.2

# Scheduler
USE_SCHEDULER = True
T_MAX = 30

NUM_EPOCHS = 40
PATIENCE = 10

# ─── SWA ──────────────────────────────────────────────────────
USE_SWA = True
SWA_START_EPOCH = 20
SWA_LR = 5e-5

# ─── Prédiction ────────────────────────────────────────────────
THRESHOLD = 0.5
USE_OPTIMAL_THRESHOLD = True
TTA_RUNS = 5
TTA_BATCH_SIZE = 32
