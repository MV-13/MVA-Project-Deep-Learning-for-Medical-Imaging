"""
Pré-extraction des features DINOv2 pour tout un DataLoader.
Stocke les résultats en mémoire pour un entraînement ultra-rapide du classifieur.
"""

import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def precompute_features(dataloader, backbone, device):
    """
    Extrait les embeddings DINOv2 de toutes les images du dataloader.

    Args:
        dataloader: fournit (images, labels)
        backbone: feature extractor gelé (eval mode)
        device: 'cuda' ou 'cpu'

    Returns:
        features: Tensor (N, D)
        labels:   Tensor (N,)
    """
    backbone.eval()
    all_feats, all_labels = [], []

    for imgs, labels in tqdm(dataloader, desc="[Features]", leave=False):
        feats = backbone(imgs.to(device)).cpu()
        all_feats.append(feats)
        all_labels.append(labels.squeeze(-1))

    features = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    print(f"  → {features.shape[0]} images, embeddings dim={features.shape[1]}")
    return features, labels
