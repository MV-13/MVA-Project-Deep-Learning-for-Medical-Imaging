# Histopathology Classification — Distribution Shift Challenge

## Problème
Classification binaire (tumeur / non-tumeur) de patchs d'images histopathologiques.
Le jeu d'entraînement provient de 3 centres hospitaliers, la validation d'un 4ᵉ centre,
et le test d'un 5ᵉ centre. Les différences de coloration (staining) entre centres créent
un **distribution shift** qui dégrade les performances.

## Stratégies contre le distribution shift

| Technique | Pourquoi ça aide |
|---|---|
| **DINOv2 (backbone gelé)** | Features self-supervised apprises sur des millions d'images, naturellement robustes aux variations visuelles |
| **Color jitter agressive** | Simule les variations de staining pendant l'entraînement pour que le classifieur ne s'appuie pas sur la couleur spécifique d'un centre |
| **Random grayscale (5%)** | Force le modèle à utiliser des indices structurels plutôt que chromatiques |
| **Dropout (0.3)** | Régularisation pour éviter le sur-apprentissage aux centres d'entraînement |
| **MLP à 1 couche cachée** | Plus expressif qu'un simple linear probe, mais assez simple pour bien généraliser |
| **Test-Time Augmentation (TTA)** | Moyenne de N prédictions augmentées par image de test → réduit la variance due au shift de couleur |
| **Cosine annealing LR** | Convergence stable vers un minimum plat (meilleure généralisation) |

## Structure du projet

```
histopath/
├── config.py               # Hyperparamètres et chemins (tout est centralisé ici)
├── utils.py                # Seed, device, sauvegarde/chargement de checkpoints
├── transforms.py           # Augmentations (train, val, TTA)
├── dataset.py              # BaselineDataset (H5) + PrecomputedDataset (features)
├── model.py                # Backbone DINOv2 + ClassificationHead (MLP)
├── feature_extraction.py   # Pré-extraction des embeddings DINOv2
├── train.py                # Boucle d'entraînement + early stopping + checkpoints
├── predict.py              # Prédiction avec TTA
├── main.py                 # Point d'entrée (orchestre tout)
├── requirements.txt
└── data/
    ├── train.h5
    ├── val.h5
    └── test.h5
```

## Usage

```bash
# Installation
pip install -r requirements.txt

# Entraînement complet (extraction → train → prédiction)
python main.py

# Reprendre après un arrêt (utilise le dernier checkpoint)
python main.py --resume

# Prédiction seule (utilise le meilleur modèle sauvegardé)
python main.py --predict-only

# Prédiction sans TTA (plus rapide, un peu moins précis)
python main.py --predict-only --no-tta
```

## Checkpoints

Le dossier `checkpoints/` contient :
- **`best_model.pth`** — meilleur modèle (selon la validation loss). Utilisé pour la prédiction finale.
- **`last_checkpoint.pth`** — état complet (modèle + optimizer + scheduler + époque). Permet de reprendre l'entraînement exactement où il s'est arrêté avec `--resume`.
