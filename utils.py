"""
Fonctions utilitaires : seed, device, etc.
"""

import random
import torch


def set_seed(seed):
    """Fixe les seeds pour la reproductibilité."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Retourne le device disponible (cuda ou cpu)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    return device
