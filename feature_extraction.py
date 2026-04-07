"""
Extraction de features : Phikon-v2 + DINOv2 + multi-pass.
"""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import BaselineDataset
from model import extract_features
from transforms import get_train_transform, get_val_transform


@torch.no_grad()
def _extract_one_pass(dataloader, phikon_model, dinov2_model, device, desc=""):
    """Extrait les features pour une passe."""
    phikon_model.eval()
    if dinov2_model is not None:
        dinov2_model.eval()

    all_feats, all_labels = [], []

    for imgs, labels in tqdm(dataloader, desc=desc, leave=False):
        feats = extract_features(phikon_model, dinov2_model, imgs.to(device)).cpu()
        all_feats.append(feats)
        all_labels.append(labels.squeeze(-1))

    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


def _cache_key():
    key = "phikonv2"
    if config.USE_DINOV2_ENSEMBLE:
        key += "+dinov2l14"
    return key


def extract_train_features(phikon_model, dinov2_model, device):
    """Train set : N passes augmentées."""
    key = _cache_key()
    cache_path = os.path.join(config.CACHE_DIR,
                              f"train_feats_{key}_aug{config.N_AUG_PASSES}.pt")

    if config.USE_FEATURE_CACHE and os.path.isfile(cache_path):
        print(f"  [CACHE] {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        print(f"  → {data['features'].shape[0]} vecteurs, dim={data['features'].shape[1]}")
        return data["features"], data["labels"]

    all_feats, all_labels = [], []

    for pass_idx in range(config.N_AUG_PASSES):
        print(f"  Passe {pass_idx + 1}/{config.N_AUG_PASSES}")
        ds = BaselineDataset(config.TRAIN_H5, get_train_transform(), mode="train")
        loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)
        feats, labels = _extract_one_pass(loader, phikon_model, dinov2_model, device,
                                          desc=f"  [Train pass {pass_idx+1}]")
        all_feats.append(feats)
        all_labels.append(labels)

    features = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    print(f"  → Total : {features.shape[0]} vecteurs, dim={features.shape[1]}")

    if config.USE_FEATURE_CACHE:
        torch.save({"features": features, "labels": labels}, cache_path)
        print(f"  [CACHE] Sauvegardé : {cache_path}")

    return features, labels


def extract_val_features(phikon_model, dinov2_model, device):
    """Val set : 1 passe propre."""
    key = _cache_key()
    cache_path = os.path.join(config.CACHE_DIR, f"val_feats_{key}.pt")

    if config.USE_FEATURE_CACHE and os.path.isfile(cache_path):
        print(f"  [CACHE] {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        print(f"  → {data['features'].shape[0]} vecteurs, dim={data['features'].shape[1]}")
        return data["features"], data["labels"]

    ds = BaselineDataset(config.VAL_H5, get_val_transform(), mode="train")
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS, pin_memory=True)
    features, labels = _extract_one_pass(loader, phikon_model, dinov2_model, device,
                                         desc="  [Val]")
    print(f"  → {features.shape[0]} vecteurs, dim={features.shape[1]}")

    if config.USE_FEATURE_CACHE:
        torch.save({"features": features, "labels": labels}, cache_path)
        print(f"  [CACHE] Sauvegardé : {cache_path}")

    return features, labels
