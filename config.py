"""
Configuration pour le projet de classification histopathologique.
Tous les hyperparamètres et chemins sont centralisés ici.
"""

# === Chemins des données ===
TRAIN_IMAGES_PATH = "data/train.h5"
VAL_IMAGES_PATH = "data/val.h5"
TEST_IMAGES_PATH = "data/test.h5"

# === Reproductibilité ===
SEED = 0

# === Modèle ===
FEATURE_EXTRACTOR_NAME = "dinov2_vitb14"
PREPROCESSING_SIZE = 224  # DINOv2 ViT-B/14 : patches 14x14, input recommandé 224

# Normalisation ImageNet (requise par DINOv2)
# Les images sont déjà en float [0, 1], pas besoin de /255
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# === Augmentation pour la robustesse inter-centres (domain shift) ===
# ColorJitter fort pour simuler les variations de coloration histologique entre centres
COLOR_JITTER_BRIGHTNESS = 0.4
COLOR_JITTER_CONTRAST   = 0.4
COLOR_JITTER_SATURATION = 0.3
COLOR_JITTER_HUE        = 0.1

# Nombre de passes d'extraction de features pour le train (multi-augmentation)
# Chaque passe utilise des augmentations différentes → dataset enrichi
N_AUG_TRAIN = 5

# Nombre de passes pour le Test-Time Augmentation (TTA) à l'inférence
N_AUG_TEST = 10

# === Entraînement ===
BATCH_SIZE = 256   # Plus grand batch car on travaille sur features pré-calculées
OPTIMIZER = "AdamW"
OPTIMIZER_PARAMS = {"lr": 3e-4, "weight_decay": 1e-4}
LOSS = "BCELoss"
METRIC = "Accuracy"
NUM_EPOCHS = 100
PATIENCE = 15

# === Sauvegardes ===
BEST_MODEL_PATH = "best_model.pth"
OUTPUT_CSV_PATH = "baseline.csv"
