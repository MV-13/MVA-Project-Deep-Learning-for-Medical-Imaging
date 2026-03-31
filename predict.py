"""
Prédiction sur le jeu de test avec Test-Time Augmentation (TTA).

Pour chaque image, on extrait les features N fois :
  - 1 passe sans augmentation (val_transform)
  - N-1 passes avec augmentations aléatoires (aug_transform)
On fait la moyenne des probabilités -> prédiction plus robuste au domain shift.
"""

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BaselineDataset


@torch.no_grad()
def predict(
    test_path,
    feature_extractor,
    classifier,
    val_transform,
    aug_transform_fn,
    n_aug,
    output_csv,
    device,
    batch_size=256,
    threshold=0.5,
):
    """
    Génère les prédictions sur le jeu de test avec TTA et écrit un CSV de soumission.

    Args:
        test_path:       chemin vers test.h5
        feature_extractor: modèle DINOv2 (eval mode, frozen)
        classifier:      tête MLP (eval mode)
        val_transform:   transform déterministe (sans augmentation)
        aug_transform_fn: callable qui retourne un nouveau transform aléatoire à chaque appel
        n_aug:           nombre total de passes TTA (1 déterministe + n_aug-1 aléatoires)
        output_csv:      chemin du CSV de sortie
        device:          device de calcul
        batch_size:      taille de batch pour l'extraction de features
        threshold:       seuil de décision binaire
    """
    feature_extractor.eval()
    classifier.eval()

    # Collecte les IDs dans l'ordre
    with h5py.File(test_path, "r") as hdf:
        test_ids = sorted(hdf.keys(), key=lambda x: int(x))

    print(f"TTA : {n_aug} passes sur {len(test_ids)} images test")
    all_probs = []  # liste de tensors (N,) un par passe

    for aug_idx in range(n_aug):
        transform = val_transform if aug_idx == 0 else aug_transform_fn()
        dataset = BaselineDataset(test_path, transform, mode="test")
        loader = DataLoader(
            dataset, shuffle=False, batch_size=batch_size,
            num_workers=0, pin_memory=device.type == "cuda",
        )

        preds = []
        for x, _ in tqdm(loader, desc=f"  TTA {aug_idx + 1}/{n_aug}", leave=False):
            feats = feature_extractor(x.to(device))
            probs = classifier(feats).squeeze(1).cpu()
            preds.append(probs)

        all_probs.append(torch.cat(preds))  # (N,)

    # Moyenne des probabilités sur toutes les passes TTA
    mean_probs = torch.stack(all_probs, dim=0).mean(dim=0)  # (N,)

    results = {
        "ID": [int(tid) for tid in test_ids],
        "Pred": (mean_probs > threshold).int().tolist(),
    }

    df = pd.DataFrame(results).set_index("ID").sort_index()
    df.to_csv(output_csv)
    print(f"Fichier de soumission sauvegardé : {output_csv} ({len(df)} prédictions)")
