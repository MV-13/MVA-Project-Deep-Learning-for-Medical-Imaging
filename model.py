"""
Modèles v5 : Phikon-v2 (pathology foundation model) + optional DINOv2 ensemble.

Phikon-v2 (Owkin) :
  - ViT-L pré-entraîné avec DINOv2 sur 450M images d'histopathologie
  - Produit des features 1024-dim via le CLS token
  - Nativement robuste au staining shift car entraîné sur des données
    de centres multiples avec des colorations variées

Ensemble optionnel :
  - Phikon-v2 (1024) + DINOv2 ViT-L (1024) = 2048-dim
  - Phikon capture le domaine histopath spécifiquement
  - DINOv2 capture la structure visuelle générale
  - Les deux se complètent

Classifieur :
  - Volontairement simple (linear probe ou petit MLP)
  - Les features Phikon sont tellement bonnes qu'un classifieur
    complexe ne fait qu'ajouter du risque d'overfitting
"""

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

import config


# ─── Backbone loading ──────────────────────────────────────────

def build_phikon(device: str = "cpu"):
    """
    Charge Phikon-v2 depuis HuggingFace.
    Retourne (model, processor, dim).
    """
    print(f"  Chargement de Phikon-v2 ({config.BACKBONE})...")
    processor = AutoImageProcessor.from_pretrained(config.BACKBONE)
    model = AutoModel.from_pretrained(config.BACKBONE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    print(f"    → dim={config.BACKBONE_DIM}")
    return model, processor


def build_dinov2(device: str = "cpu"):
    """Charge DINOv2 ViT-L/14 depuis torch.hub."""
    print(f"  Chargement de {config.DINOV2_BACKBONE}...")
    model = torch.hub.load("facebookresearch/dinov2", config.DINOV2_BACKBONE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    print(f"    → dim={config.DINOV2_DIM}")
    return model


@torch.no_grad()
def extract_features(phikon_model, dinov2_model, images):
    """
    Extrait les features de Phikon-v2 (+ optionnellement DINOv2).

    Phikon-v2 utilise le CLS token comme feature.
    DINOv2 utilise aussi le CLS token par défaut.

    images: (B, C, H, W) — déjà normalisées avec la normalisation ImageNet
    Returns: (B, total_dim)
    """
    # Phikon-v2 : passe directe, CLS token
    outputs = phikon_model(pixel_values=images)
    phikon_feats = outputs.last_hidden_state[:, 0, :]  # CLS token (B, 1024)

    if dinov2_model is not None and config.USE_DINOV2_ENSEMBLE:
        dino_feats = dinov2_model(images)  # (B, 1024)
        return torch.cat([phikon_feats, dino_feats], dim=-1)  # (B, 2048)

    return phikon_feats


# ─── Classifieur ───────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Classifieur simple. Sortie = logit.

    Si hidden_dims est vide → linear probe pur (1 couche)
    Sinon → petit MLP avec BatchNorm + Dropout
    """

    def __init__(self, input_dim: int, hidden_dims: list = None, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = []

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


def build_classifier(input_dim: int, hidden_dims: list = None, dropout: float = 0.3,
                     device: str = "cpu"):
    model = ClassificationHead(input_dim, hidden_dims, dropout)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Classifieur : {n_params:,} paramètres")
    if not hidden_dims:
        print(f"    Architecture : Linear probe (1024→1)")
    else:
        arch = f"{input_dim}→" + "→".join(str(d) for d in hidden_dims) + "→1"
        print(f"    Architecture : MLP ({arch})")
    return model.to(device)
