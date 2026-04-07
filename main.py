#!/usr/bin/env python3
"""
Pipeline v5 — Phikon-v2 (pathology foundation model) + DINOv2 ensemble.

Approche globale :
  - Backbone principal : Phikon-v2, pré-entraîné sur 450M images
    d'histopathologie (au lieu de DINOv2 pré-entraîné sur images naturelles)
  - Classifieur volontairement simple (petit MLP ou linear probe)
  - Ensemble optionnel avec DINOv2 ViT-L

Usage :
  python main.py                   # pipeline complet
  python main.py --resume          # reprendre
  python main.py --predict-only    # prédiction seule
  python main.py --no-tta          # sans TTA
  python main.py --clear-cache     # vider le cache
  python main.py --model swa       # choisir best_ckpt ou swa
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

import config
from utils import set_seed, get_device, load_checkpoint
from dataset import PrecomputedDataset
from model import build_phikon, build_dinov2, build_classifier
from feature_extraction import extract_train_features, extract_val_features
from train import train
from predict import predict_with_tta


def parse_args():
    p = argparse.ArgumentParser(description="Histopathology v5 — Phikon-v2")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--predict-only", action="store_true")
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--clear-cache", action="store_true")
    p.add_argument("--model", type=str, default=None,
                   choices=["best_ckpt", "swa"])
    p.add_argument("--threshold", type=float, default=None)
    return p.parse_args()


def _load_model(classifier, model_name, device):
    path_map = {
        "best_ckpt": config.BEST_MODEL_PATH,
        "swa": config.SWA_MODEL_PATH,
    }
    path = path_map.get(model_name, config.BEST_MODEL_PATH)
    if not os.path.isfile(path):
        print(f"[WARN] {path} introuvable, fallback best_model.pth")
        path = config.BEST_MODEL_PATH
    load_checkpoint(path, classifier, device=device)


def main():
    args = parse_args()

    set_seed(config.SEED)
    device = get_device()

    if args.clear_cache:
        import shutil
        shutil.rmtree(config.CACHE_DIR, ignore_errors=True)
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        print("[INFO] Cache vidé.")

    # ━━━ 1. Backbones ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n╔══ Backbones ══╗")
    phikon_model, phikon_processor = build_phikon(device)

    dinov2_model = None
    if config.USE_DINOV2_ENSEMBLE:
        dinov2_model = build_dinov2(device)

    feat_dim = config.BACKBONE_DIM
    if config.USE_DINOV2_ENSEMBLE:
        feat_dim += config.DINOV2_DIM
    print(f"  → Feature dimension totale : {feat_dim}")

    # ━━━ Mode prédiction seule ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if args.predict_only:
        print("\n╔══ Prédiction ══╗")
        classifier = build_classifier(feat_dim, config.HIDDEN_DIMS,
                                      config.DROPOUT, device)
        model_name = args.model or "best_ckpt"
        _load_model(classifier, model_name, device)
        classifier.eval()

        threshold = args.threshold if args.threshold is not None else config.THRESHOLD
        tta = 1 if args.no_tta else config.TTA_RUNS

        predict_with_tta(config.TEST_H5, phikon_model, dinov2_model, classifier,
                         device, config.OUTPUT_CSV, tta_runs=tta, threshold=threshold)
        return

    # ━━━ 2. Extraction des features ━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n╔══ Features (Phikon-v2"
          + (" + DINOv2" if config.USE_DINOV2_ENSEMBLE else "")
          + f" × {config.N_AUG_PASSES} passes) ══╗")

    print("  ── Train ──")
    train_feats, train_labels = extract_train_features(
        phikon_model, dinov2_model, device)

    print("  ── Validation ──")
    val_feats, val_labels = extract_val_features(
        phikon_model, dinov2_model, device)

    train_loader = DataLoader(
        PrecomputedDataset(train_feats, train_labels),
        batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        PrecomputedDataset(val_feats, val_labels),
        batch_size=config.BATCH_SIZE, shuffle=False)

    # ━━━ 3. Entraînement ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n╔══ Entraînement ══╗")
    classifier = build_classifier(feat_dim, config.HIDDEN_DIMS,
                                  config.DROPOUT, device)

    resume_path = config.LAST_MODEL_PATH if args.resume else None
    best_val_loss, optimal_threshold, best_candidate = train(
        classifier, train_loader, val_loader, device, resume_path=resume_path)

    # ━━━ 4. Prédiction ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n╔══ Prédiction ══╗")
    model_name = args.model or best_candidate
    classifier = build_classifier(feat_dim, config.HIDDEN_DIMS,
                                  config.DROPOUT, device)
    _load_model(classifier, model_name, device)
    classifier.eval()

    threshold = args.threshold if args.threshold is not None else optimal_threshold
    tta = 1 if args.no_tta else config.TTA_RUNS

    predict_with_tta(config.TEST_H5, phikon_model, dinov2_model, classifier,
                     device, config.OUTPUT_CSV, tta_runs=tta, threshold=threshold)

    print("\n✓ Pipeline terminé !")


if __name__ == "__main__":
    main()
