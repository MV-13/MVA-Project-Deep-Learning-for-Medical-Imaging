"""
Prédiction sur le jeu de test avec Test-Time Augmentation (TTA).

Stratégie TTA contre le distribution shift :
  On applique plusieurs fois des augmentations de couleur aléatoires
  à chaque image de test, puis on moyenne les probabilités.
  Cela rend la prédiction plus robuste aux variations de coloration.
"""

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
from transforms import get_val_transform, get_tta_transform


@torch.no_grad()
def predict_with_tta(
    test_h5_path: str,
    backbone,
    classifier,
    device: str,
    output_csv: str,
    tta_runs: int = 5,
    threshold: float = 0.5,
):
    """
    Génère les prédictions sur le test set avec TTA.

    Pour chaque image :
      1. On fait 1 prédiction "propre" (val_transform)
      2. On fait (tta_runs - 1) prédictions augmentées (tta_transform)
      3. On moyenne toutes les probabilités
      4. On seuille à `threshold`

    Args:
        test_h5_path: chemin vers test.h5
        backbone: DINOv2 feature extractor (gelé, eval)
        classifier: tête de classification (eval)
        device: cuda ou cpu
        output_csv: chemin du CSV de sortie
        tta_runs: nombre total de passes par image
        threshold: seuil binaire
    """
    backbone.eval()
    classifier.eval()

    val_tf = get_val_transform()
    tta_tf = get_tta_transform()

    results = {"ID": [], "Pred": []}

    with h5py.File(test_h5_path, "r") as f:
        test_ids = list(f.keys())

        for key in tqdm(test_ids, desc="[Predict TTA]"):
            raw = torch.from_numpy(np.array(f[key]["img"]))  # (C, H, W) uint8

            probs = []

            # 1 passe propre (sans augmentation aléatoire)
            img_clean = val_tf(raw).unsqueeze(0).to(device)
            feat = backbone(img_clean)
            probs.append(classifier(feat).cpu().item())

            # (tta_runs - 1) passes augmentées
            for _ in range(tta_runs - 1):
                img_aug = tta_tf(raw).unsqueeze(0).to(device)
                feat = backbone(img_aug)
                probs.append(classifier(feat).cpu().item())

            # Moyenne des probabilités
            avg_prob = np.mean(probs)
            results["ID"].append(int(key))
            results["Pred"].append(int(avg_prob > threshold))

    df = pd.DataFrame(results).set_index("ID")
    df.to_csv(output_csv)
    print(f"[INFO] Soumission sauvegardée : {output_csv}  ({len(df)} images)")
    return df
