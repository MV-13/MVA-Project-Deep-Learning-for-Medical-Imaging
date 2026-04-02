"""
Modèles :
  - Backbone DINOv2 gelé (feature extractor)
  - Tête de classification MLP avec dropout (linear probing amélioré)
"""

import torch
import torch.nn as nn


def build_feature_extractor(name: str = "dinov2_vits14", device: str = "cpu"):
    """
    Charge DINOv2 depuis torch.hub, le gèle, et le met en mode eval.
    DINOv2 produit des features robustes aux variations visuelles,
    ce qui est un atout majeur contre le distribution shift.
    """
    model = torch.hub.load("facebookresearch/dinov2", name)
    model.eval()
    # Geler tous les paramètres
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


class ClassificationHead(nn.Module):
    """
    Tête de classification : MLP à une couche cachée avec dropout.

    Le dropout aide à la généralisation (important quand train ≠ val/test).
    Architecture :  Linear → ReLU → Dropout → Linear → Sigmoid

    Args:
        input_dim:  dimension de l'embedding du backbone
        hidden_dim: dimension de la couche cachée
        dropout:    taux de dropout
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def build_classifier(input_dim: int, hidden_dim: int = 256, dropout: float = 0.3,
                     device: str = "cpu"):
    """Construit la tête de classification et la place sur le device."""
    model = ClassificationHead(input_dim, hidden_dim, dropout)
    return model.to(device)
