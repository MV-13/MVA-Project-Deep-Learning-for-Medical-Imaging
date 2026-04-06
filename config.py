"""
Configuration v5 — Approche radicalement différente.

Changement de paradigme :
  - AVANT : DINOv2 (pré-entraîné sur images naturelles) + classifieur complexe
  - MAINTENANT : Phikon-v2 (pré-entraîné sur 450M images d'histopathologie)
    + classifieur très simple

Pourquoi c'est mieux :
  DINOv2 a appris à représenter des chiens, des voitures, des paysages.
  Phikon-v2 a appris à représenter des cellules, des tissus, des colorations.
  Ses features sont nativement pertinentes pour la classification
  histopathologique et intrinsèquement plus robustes au staining shift
  car il a vu des centaines de milliers de lames de centres différents.

  Avec des features aussi bonnes, un simple linear probe ou petit MLP
  suffit — la complexité du classifieur v3/v4 n'était pas nécessaire
  et pouvait même nuire (overfitting).

Pipeline :
  1. Phikon-v2 (ViT-L, 1024-dim) comme feature extractor gelé
  2. Multi-pass augmented extraction (stain jitter + géométrie)
  3. Classifieur simple : Linear(1024→1) ou petit MLP
  4. BCEWithLogitsLoss + label smoothing
  5. SWA pour la généralisation
  6. TTA au test
"""

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

# On garde aussi DINOv2 ViT-L comme 2e backbone pour un ensemble
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
# Avec un backbone histopath, un classifieur simple est meilleur.
# Option 1 : linear probe pur  → HIDDEN_DIMS = []
# Option 2 : petit MLP          → HIDDEN_DIMS = [256]
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
T_MAX = 30          # cosine annealing simple, pas de warm restarts

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
