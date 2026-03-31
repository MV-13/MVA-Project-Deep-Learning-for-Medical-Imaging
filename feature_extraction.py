"""
Pré-calcul des features avec le feature extractor.
Permet d'éviter de recalculer les embeddings à chaque époque.
"""

import numpy as np
import torch
from tqdm import tqdm


def precompute_features(dataloader, model, device):
    """
    Extrait les features de toutes les images d'un dataloader via le modèle donné.

    Args:
        dataloader: DataLoader qui fournit (images, labels)
        model: feature extractor (en mode eval)
        device: device de calcul

    Returns:
        features: tensor (N, D)
        labels: tensor (N,)
    """
    all_features = []
    all_labels = []

    for x, y in tqdm(dataloader, desc="Extraction des features", leave=False):
        with torch.no_grad():
            feats = model(x.to(device)).detach().cpu().numpy()
        all_features.append(feats)
        all_labels.append(y.numpy())

    features = torch.tensor(np.vstack(all_features))
    labels = torch.tensor(np.hstack(all_labels))
    return features, labels
