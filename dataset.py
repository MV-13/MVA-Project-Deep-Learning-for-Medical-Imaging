"""
Datasets pour le chargement des données histopathologiques.
- BaselineDataset : charge les images depuis un fichier H5 et applique un prétraitement.
- PrecomputedDataset : charge des features déjà extraites (pour le linear probing).
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class BaselineDataset(Dataset):
    """
    Dataset qui lit les images et labels depuis un fichier H5.

    Args:
        dataset_path: chemin vers le fichier .h5
        preprocessing: transformation torchvision à appliquer aux images
        mode: 'train' pour retourner les labels, 'test' pour retourner None
    """

    def __init__(self, dataset_path, preprocessing, mode="train"):
        super().__init__()
        self.dataset_path = dataset_path
        self.preprocessing = preprocessing
        self.mode = mode

        with h5py.File(self.dataset_path, "r") as hdf:
            self.image_ids = list(hdf.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        with h5py.File(self.dataset_path, "r") as hdf:
            img = torch.tensor(np.array(hdf[img_id]["img"]))
            label = np.array(hdf[img_id]["label"]) if self.mode == "train" else -1
        return self.preprocessing(img).float(), label


class PrecomputedDataset(Dataset):
    """
    Dataset pour features pré-extraites (après passage dans le feature extractor).

    Args:
        features: tensor (N, D) des embeddings
        labels: tensor (N,) des labels
    """

    def __init__(self, features, labels):
        super().__init__()
        self.features = features
        self.labels = labels.unsqueeze(-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].float()
