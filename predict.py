"""
Prédiction avec TTA Phikon-v2 + DINOv2.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from dataset import BaselineDataset
from model import extract_features
from transforms import get_val_transform, get_tta_transform


@torch.no_grad()
def _predict_pass(h5_path, phikon_model, dinov2_model, classifier,
                  transform, device, batch_size):
    """Une passe complète."""
    phikon_model.eval()
    if dinov2_model is not None:
        dinov2_model.eval()
    classifier.eval()

    ds = BaselineDataset(h5_path, transform, mode="test")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=config.NUM_WORKERS, pin_memory=True)

    all_probs = []
    for imgs, _ in loader:
        feats = extract_features(phikon_model, dinov2_model, imgs.to(device))
        logits = classifier(feats)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.extend(probs.tolist())

    return np.array(all_probs), ds.ids


def predict_with_tta(test_h5_path, phikon_model, dinov2_model, classifier,
                     device, output_csv, tta_runs=10, threshold=0.5):
    """Prédiction TTA : 1 clean + N-1 augmentées."""
    batch_size = config.TTA_BATCH_SIZE

    print(f"  TTA : {tta_runs} passes, seuil={threshold:.3f}")

    print(f"  Pass 1/{tta_runs} (clean)")
    probs_clean, test_ids = _predict_pass(
        test_h5_path, phikon_model, dinov2_model, classifier,
        get_val_transform(), device, batch_size)
    all_probs = [probs_clean]

    for i in range(1, tta_runs):
        print(f"  Pass {i+1}/{tta_runs} (augmented)")
        probs_aug, _ = _predict_pass(
            test_h5_path, phikon_model, dinov2_model, classifier,
            get_tta_transform(), device, batch_size)
        all_probs.append(probs_aug)

    avg_probs = np.mean(all_probs, axis=0)
    ids = [int(k) for k in test_ids]
    preds = (avg_probs > threshold).astype(int)

    df = pd.DataFrame({"ID": ids, "Pred": preds}).set_index("ID")
    df.to_csv(output_csv)
    print(f"\n[INFO] Soumission : {output_csv}  ({len(df)} images)")
    print(f"       {np.sum(preds == 0)} négatifs, {np.sum(preds == 1)} positifs")

    probs_csv = output_csv.replace(".csv", "_probs.csv")
    df_probs = pd.DataFrame({"ID": ids, "Prob": avg_probs}).set_index("ID")
    df_probs.to_csv(probs_csv)
    print(f"       Probabilités : {probs_csv}")

    return df
