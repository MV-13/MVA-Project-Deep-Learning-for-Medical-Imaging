#!/usr/bin/env python3
"""
Pipeline complet : extraction → entraînement → prédiction.

Fonctionnalités clés contre le distribution shift :
  1. DINOv2 comme backbone (features robustes apprises par self-supervised learning)
  2. Augmentation de couleur agressive (simule les variations de staining)
  3. Dropout dans la tête de classification (régularisation)
  4. Test-Time Augmentation (TTA) : moyenne de N prédictions augmentées

Reprise d'entraînement :
  Le script sauvegarde automatiquement le dernier checkpoint.
  Pour reprendre après un arrêt, relancer simplement `python main.py --resume`.

Usage :
  python main.py                   # entraînement complet
  python main.py --resume          # reprendre depuis le dernier checkpoint
  python main.py --predict-only    # prédiction uniquement (nécessite best_model.pth)
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

import config
from utils import set_seed, get_device, load_checkpoint
from transforms import get_train_transform, get_val_transform
from dataset import BaselineDataset, PrecomputedDataset
from model import build_feature_extractor, build_classifier
from feature_extraction import precompute_features
from train import train
from predict import predict_with_tta


def parse_args():
    parser = argparse.ArgumentParser(description="Histopathology classification")
    parser.add_argument("--resume", action="store_true",
                        help="Reprendre l'entraînement depuis le dernier checkpoint")
    parser.add_argument("--predict-only", action="store_true",
                        help="Seulement prédire (nécessite un best_model.pth)")
    parser.add_argument("--no-tta", action="store_true",
                        help="Désactiver le TTA lors de la prédiction")
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(config.SEED)
    device = get_device()

    # ━━━ 1. Backbone DINOv2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n╔══ Chargement du backbone DINOv2 ══╗")
    backbone = build_feature_extractor(config.BACKBONE, device)
    feat_dim = backbone.num_features
    print(f"  Backbone : {config.BACKBONE}  →  dim={feat_dim}")

    if args.predict_only:
        # ━━━ Prédiction seule ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        print("\n╔══ Mode prédiction uniquement ══╗")
        classifier = build_classifier(feat_dim, config.HIDDEN_DIM, config.DROPOUT, device)
        load_checkpoint(config.BEST_MODEL_PATH, classifier, device=device)
        classifier.eval()

        tta = 1 if args.no_tta else config.TTA_RUNS
        predict_with_tta(
            config.TEST_H5, backbone, classifier, device,
            config.OUTPUT_CSV, tta_runs=tta, threshold=config.THRESHOLD,
        )
        return

    # ━━━ 2. Datasets bruts + extraction de features ━━━━━━━━━━━━
    print("\n╔══ Extraction des features ══╗")

    # Train : avec augmentations de couleur pour que les features capturent
    # la variabilité de staining (le backbone est gelé, mais les augmentations
    # changent les valeurs de pixels en entrée !)
    print("  Train features :")
    train_ds = BaselineDataset(config.TRAIN_H5, get_train_transform(), mode="train")
    train_loader_raw = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                                  shuffle=False, num_workers=config.NUM_WORKERS,
                                  pin_memory=True)
    train_feats, train_labels = precompute_features(train_loader_raw, backbone, device)

    # Validation : sans augmentation
    print("  Val features :")
    val_ds = BaselineDataset(config.VAL_H5, get_val_transform(), mode="train")
    val_loader_raw = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                                shuffle=False, num_workers=config.NUM_WORKERS,
                                pin_memory=True)
    val_feats, val_labels = precompute_features(val_loader_raw, backbone, device)

    # DataLoaders sur features pré-calculées (très rapide)
    train_loader = DataLoader(
        PrecomputedDataset(train_feats, train_labels),
        batch_size=config.BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        PrecomputedDataset(val_feats, val_labels),
        batch_size=config.BATCH_SIZE, shuffle=False,
    )

    # ━━━ 3. Entraînement ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n╔══ Entraînement du classifieur ══╗")
    classifier = build_classifier(feat_dim, config.HIDDEN_DIM, config.DROPOUT, device)

    resume_path = config.LAST_MODEL_PATH if args.resume else None
    train(classifier, train_loader, val_loader, device, resume_path=resume_path)

    # ━━━ 4. Prédiction avec TTA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n╔══ Prédiction sur le test set ══╗")
    # Charger le meilleur modèle
    classifier = build_classifier(feat_dim, config.HIDDEN_DIM, config.DROPOUT, device)
    load_checkpoint(config.BEST_MODEL_PATH, classifier, device=device)
    classifier.eval()

    tta = 1 if args.no_tta else config.TTA_RUNS
    predict_with_tta(
        config.TEST_H5, backbone, classifier, device,
        config.OUTPUT_CSV, tta_runs=tta, threshold=config.THRESHOLD,
    )

    print("\n✓ Terminé !")


if __name__ == "__main__":
    main()
