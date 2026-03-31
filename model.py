"""
Construction des modèles : feature extractor (DINOv2) et têtes de classification.
"""

import torch
import torch.nn as nn


def build_feature_extractor(model_name="dinov2_vitb14", device="cpu"):
    """
    Charge un feature extractor DINOv2 pré-entraîné depuis torch hub.
    Le modèle est mis en mode évaluation (les poids sont gelés).

    Returns:
        model: le feature extractor sur le device spécifié
    """
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def build_mlp_probe(input_dim, hidden_dims=(512, 256), dropout=0.3, device="cpu"):
    """
    Tête de classification MLP avec deux couches cachées.
    Conçu pour être robuste au domain shift grâce au BatchNorm et Dropout.

    Architecture : Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear -> Sigmoid

    Args:
        input_dim:   dimension des features en entrée (768 pour dinov2_vitb14)
        hidden_dims: tuple de dimensions des couches cachées
        dropout:     taux de dropout
        device:      device cible

    Returns:
        model: le classifieur MLP
    """
    layers = []
    prev_dim = input_dim
    for i, h_dim in enumerate(hidden_dims):
        layers += [
            nn.Linear(prev_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout if i == 0 else dropout * 0.5),
        ]
        prev_dim = h_dim
    layers += [
        nn.Linear(prev_dim, 1),
        nn.Sigmoid(),
    ]
    model = nn.Sequential(*layers)
    return model.to(device)


def build_linear_probe(input_dim, device="cpu"):
    """
    Tête de classification linéaire simple (conservée pour référence).
    """
    model = nn.Sequential(
        nn.Linear(input_dim, 1),
        nn.Sigmoid(),
    )
    return model.to(device)
