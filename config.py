"""
Configuration centrale du projet.
Hyperparamètres, chemins, et options de stratégie contre le distribution shift.
"""

import os

# ─── Chemins ───────────────────────────────────────────────────
TRAIN_H5 = "data/train.h5"
VAL_H5 = "data/val.h5"
TEST_H5 = "data/test.h5"

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
OUTPUT_CSV = "submission.csv"

# ─── Reproductibilité ─────────────────────────────────────────
SEED = 42

# ─── Feature extractor ────────────────────────────────────────
# DINOv2 ViT‑S/14 – bon compromis qualité/vitesse pour l'histopathologie
BACKBONE = "dinov2_vits14"
INPUT_SIZE = (224, 224)  # taille native du ViT-S/14 pour meilleure qualité

# ─── Stain augmentation ───────────────────────────────────────
# Paramètres pour la color jitter agressive (simule le shift de coloration)
USE_STAIN_AUGMENTATION = True
COLOR_JITTER_BRIGHTNESS = 0.3
COLOR_JITTER_CONTRAST = 0.3
COLOR_JITTER_SATURATION = 0.4
COLOR_JITTER_HUE = 0.15

# ─── Entraînement ─────────────────────────────────────────────
BATCH_SIZE = 64
NUM_WORKERS = 4

# Architecture du classifieur (au‑dessus du backbone gelé)
HIDDEN_DIM = 256
DROPOUT = 0.3

# Optimiseur
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Scheduler cosine annealing
USE_SCHEDULER = True
T_MAX = 30  # période du cosine (= NUM_EPOCHS si une seule période)

NUM_EPOCHS = 30
PATIENCE = 8  # early stopping

# ─── Prédiction ────────────────────────────────────────────────
THRESHOLD = 0.5
# Test‑time augmentation : nombre de vues augmentées par image de test
TTA_RUNS = 5
