"""
Lance uniquement la prédiction avec TTA sur le test set.
Le modèle doit avoir été entraîné au préalable (best_model.pth doit exister).

Usage :
    python run_predict.py
"""

import torch

import config
from utils import set_seed, get_device
from model import build_feature_extractor, build_mlp_probe
from predict import predict
from main import build_train_transform, build_val_transform


def main():
    set_seed(config.SEED)
    device = get_device()

    print("\n=== Chargement du feature extractor ===")
    feature_extractor = build_feature_extractor(config.FEATURE_EXTRACTOR_NAME, device)

    print("\n=== Chargement du MLP probe ===")
    mlp_probe = build_mlp_probe(feature_extractor.num_features, device=device)
    mlp_probe.load_state_dict(torch.load(config.BEST_MODEL_PATH, weights_only=True))
    mlp_probe.eval()
    print(f"Modèle chargé depuis {config.BEST_MODEL_PATH}")

    print("\n=== Prédiction avec TTA ===")
    predict(
        test_path=config.TEST_IMAGES_PATH,
        feature_extractor=feature_extractor,
        classifier=mlp_probe,
        val_transform=build_val_transform(),
        aug_transform_fn=build_train_transform,
        n_aug=config.N_AUG_TEST,
        output_csv=config.OUTPUT_CSV_PATH,
        device=device,
        batch_size=config.BATCH_SIZE,
    )

    print("\nTerminé !")


if __name__ == "__main__":
    main()
