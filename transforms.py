"""
Transformations d'augmentation pour lutter contre le distribution shift de coloration.

Stratégie :
  - Train : augmentations agressives de couleur (simule les variations de staining
    entre centres), plus flips/rotations géométriques classiques.
  - Val/Test : seulement resize + normalisation.
  - TTA (test-time augmentation) : mêmes augmentations couleur que le train,
    appliquées plusieurs fois sur chaque image de test, puis on moyenne les prédictions.
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import config


def _normalize():
    """Normalisation ImageNet (adaptée aux backbones pré-entraînés)."""
    return T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])


def get_train_transform():
    """
    Augmentations d'entraînement :
      1) Resize à INPUT_SIZE
      2) Conversion float [0,1]
      3) Flips horizontaux/verticaux aléatoires
      4) Rotations multiples de 90° aléatoires
      5) Color jitter agressive (clé pour le distribution shift !)
      6) Gaussian blur léger (optionnel, simule le flou de mise au point)
      7) Normalisation ImageNet
    """
    transforms_list = [
        T.Resize(config.INPUT_SIZE),
        T.ConvertImageDtype(torch.float32),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([T.RandomRotation((90, 90))], p=0.5),
    ]

    if config.USE_STAIN_AUGMENTATION:
        transforms_list.append(
            T.ColorJitter(
                brightness=config.COLOR_JITTER_BRIGHTNESS,
                contrast=config.COLOR_JITTER_CONTRAST,
                saturation=config.COLOR_JITTER_SATURATION,
                hue=config.COLOR_JITTER_HUE,
            )
        )
        transforms_list.append(
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2)
        )
        # Random grayscale : force le modèle à ne pas se fier uniquement à la couleur
        transforms_list.append(T.RandomGrayscale(p=0.05))

    transforms_list.append(_normalize())
    return T.Compose(transforms_list)


def get_val_transform():
    """Transformation de validation : aucune augmentation."""
    return T.Compose([
        T.Resize(config.INPUT_SIZE),
        T.ConvertImageDtype(torch.float32),
        _normalize(),
    ])


def get_tta_transform():
    """
    Transformation pour le test-time augmentation.
    Même distribution que le train (color jitter + flips) pour capturer
    l'incertitude liée au shift de coloration.
    """
    return get_train_transform()
