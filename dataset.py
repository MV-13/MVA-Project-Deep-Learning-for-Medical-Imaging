"""
Datasets.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class BaselineDataset(Dataset):
    """Charge les images/labels depuis un fichier H5."""

    def __init__(self, h5_path: str, transform=None, mode: str = "train"):
        self.h5_path = h5_path
        self.transform = transform
        self.mode = mode
        with h5py.File(self.h5_path, "r") as f:
            self.ids = list(f.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        key = self.ids[idx]
        with h5py.File(self.h5_path, "r") as f:
            img = torch.from_numpy(np.array(f[key]["img"]))
            label = float(np.array(f[key]["label"])) if self.mode == "train" else -1.0
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor([label], dtype=torch.float32)


class PrecomputedDataset(Dataset):
    """Dataset sur features pré-extraites."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels.unsqueeze(-1).float() if labels.dim() == 1 else labels.float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
