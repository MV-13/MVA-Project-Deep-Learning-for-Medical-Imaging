"""
Augmentations v5. Légèrement plus douces qu'avant car Phikon-v2 est
déjà robuste aux variations de staining.
"""

import torch
import torchvision.transforms as T
import config


def _normalize():
    """Normalisation ImageNet (utilisée par Phikon-v2 et DINOv2)."""
    return T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])


def get_train_transform():
    """Augmentations d'entraînement."""
    transforms_list = [
        T.Resize(config.INPUT_SIZE),
        T.ConvertImageDtype(torch.float32),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([T.RandomRotation((90, 90))], p=0.25),
        T.RandomApply([T.RandomRotation((180, 180))], p=0.25),
        T.RandomApply([T.RandomRotation((270, 270))], p=0.25),
    ]

    if config.USE_STAIN_AUGMENTATION:
        transforms_list.extend([
            T.ColorJitter(
                brightness=config.COLOR_JITTER_BRIGHTNESS,
                contrast=config.COLOR_JITTER_CONTRAST,
                saturation=config.COLOR_JITTER_SATURATION,
                hue=config.COLOR_JITTER_HUE,
            ),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.15),
            T.RandomGrayscale(p=0.05),
        ])

    transforms_list.append(_normalize())
    return T.Compose(transforms_list)


def get_val_transform():
    """Validation : pas d'augmentation."""
    return T.Compose([
        T.Resize(config.INPUT_SIZE),
        T.ConvertImageDtype(torch.float32),
        _normalize(),
    ])


def get_tta_transform():
    """TTA : augmentations modérées."""
    transforms_list = [
        T.Resize(config.INPUT_SIZE),
        T.ConvertImageDtype(torch.float32),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([T.RandomRotation((90, 90))], p=0.25),
    ]

    if config.USE_STAIN_AUGMENTATION:
        transforms_list.append(
            T.ColorJitter(
                brightness=config.COLOR_JITTER_BRIGHTNESS * 0.6,
                contrast=config.COLOR_JITTER_CONTRAST * 0.6,
                saturation=config.COLOR_JITTER_SATURATION * 0.6,
                hue=config.COLOR_JITTER_HUE * 0.6,
            )
        )

    transforms_list.append(_normalize())
    return T.Compose(transforms_list)
