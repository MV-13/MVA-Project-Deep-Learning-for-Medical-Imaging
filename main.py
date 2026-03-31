"""
Point d'entrée principal du projet.
Orchestre : extraction de features -> entraînement -> prédiction.

Stratégie anti-domain-shift :
  - Augmentation de couleur forte (simule les variations de coloration inter-centres)
  - Extraction de features N_AUG_TRAIN fois avec augmentations différentes
  - TTA (Test-Time Augmentation) à l'inférence

Usage :
    python main.py
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import config
from utils import set_seed, get_device
from dataset import BaselineDataset, PrecomputedDataset
from model import build_feature_extractor, build_mlp_probe
from feature_extraction import precompute_features
from train import train
from predict import predict


def build_train_transform():
    """
    Preprocessing pour le train avec augmentation forte anti-domain-shift.
    Images : float16 [0,1] (C,H,W) -> float32 [0,1] normalisé ImageNet.
    ColorJitter fort pour simuler les variations de coloration histologique.
    """
    return transforms.Compose([
        transforms.Lambda(lambda x: x.float()),
        transforms.Resize(config.PREPROCESSING_SIZE, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=config.COLOR_JITTER_BRIGHTNESS,
                contrast=config.COLOR_JITTER_CONTRAST,
                saturation=config.COLOR_JITTER_SATURATION,
                hue=config.COLOR_JITTER_HUE,
            )
        ], p=0.9),
        transforms.RandomGrayscale(p=0.02),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def build_val_transform():
    """
    Preprocessing pour validation/test (sans augmentation aléatoire).
    """
    return transforms.Compose([
        transforms.Lambda(lambda x: x.float()),
        transforms.Resize(config.PREPROCESSING_SIZE, antialias=True),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def main():
    # --- Initialisation ---
    set_seed(config.SEED)
    device = get_device()

    val_transform = build_val_transform()

    # --- 1. Feature extractor ---
    print("\n=== Chargement du feature extractor DINOv2 ===")
    feature_extractor = build_feature_extractor(config.FEATURE_EXTRACTOR_NAME, device)
    print(f"Backbone: {config.FEATURE_EXTRACTOR_NAME} | Features: {feature_extractor.num_features}D")

    # --- 2. Extraction multi-augmentation pour le train ---
    # On extrait N_AUG_TRAIN fois les features d'entraînement avec des augmentations
    # différentes à chaque passe. Cela diversifie les features et rend le MLP robuste
    # aux variations de coloration inter-centres.
    print(f"\n=== Extraction des features train ({config.N_AUG_TRAIN} passes d'augmentation) ===")

    all_train_features = []
    all_train_labels = []

    for aug_idx in range(config.N_AUG_TRAIN):
        print(f"  Passe {aug_idx + 1}/{config.N_AUG_TRAIN}")
        set_seed(config.SEED + aug_idx)  # seed différente -> augmentations différentes
        train_transform = build_train_transform()
        train_dataset = BaselineDataset(config.TRAIN_IMAGES_PATH, train_transform, mode="train")
        train_loader = DataLoader(
            train_dataset, shuffle=False, batch_size=config.BATCH_SIZE,
            num_workers=0, pin_memory=device.type == "cuda",
        )
        feats, labels = precompute_features(train_loader, feature_extractor, device)
        all_train_features.append(feats)
        all_train_labels.append(labels)

    train_features = torch.cat(all_train_features, dim=0)
    train_labels = torch.cat(all_train_labels, dim=0)
    print(f"Dataset train augmenté : {len(train_features)} samples ({config.N_AUG_TRAIN}x)")

    # --- 3. Extraction des features de validation (sans augmentation) ---
    print("\n=== Extraction des features validation ===")
    val_dataset = BaselineDataset(config.VAL_IMAGES_PATH, val_transform, mode="train")
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=config.BATCH_SIZE,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    val_features, val_labels = precompute_features(val_loader, feature_extractor, device)

    # DataLoaders sur features pré-calculées
    precomp_train = PrecomputedDataset(train_features, train_labels)
    precomp_val = PrecomputedDataset(val_features, val_labels)

    precomp_train_loader = DataLoader(precomp_train, shuffle=True, batch_size=config.BATCH_SIZE)
    precomp_val_loader = DataLoader(precomp_val, shuffle=False, batch_size=config.BATCH_SIZE)

    # --- 4. Entraînement du MLP probe ---
    print("\n=== Entraînement du MLP probe ===")
    mlp_probe = build_mlp_probe(feature_extractor.num_features, device=device)
    print(f"Architecture MLP: {feature_extractor.num_features} -> 512 -> 256 -> 1")

    train(
        model=mlp_probe,
        train_dataloader=precomp_train_loader,
        val_dataloader=precomp_val_loader,
        optimizer_name=config.OPTIMIZER,
        optimizer_params=config.OPTIMIZER_PARAMS,
        loss_name=config.LOSS,
        metric_name=config.METRIC,
        num_epochs=config.NUM_EPOCHS,
        patience=config.PATIENCE,
        save_path=config.BEST_MODEL_PATH,
        device=device,
    )

    # --- 5. Prédiction sur le test set avec TTA ---
    print("\n=== Prédiction avec TTA ===")
    mlp_probe.load_state_dict(torch.load(config.BEST_MODEL_PATH, weights_only=True))
    mlp_probe.eval()

    predict(
        test_path=config.TEST_IMAGES_PATH,
        feature_extractor=feature_extractor,
        classifier=mlp_probe,
        val_transform=val_transform,
        aug_transform_fn=build_train_transform,
        n_aug=config.N_AUG_TEST,
        output_csv=config.OUTPUT_CSV_PATH,
        device=device,
        batch_size=config.BATCH_SIZE,
    )

    print("\nTerminé !")


if __name__ == "__main__":
    main()
