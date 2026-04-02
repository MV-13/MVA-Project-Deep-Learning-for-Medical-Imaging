"""
Fonctions utilitaires : reproductibilité, device, sauvegarde/chargement de checkpoints.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int):
    """Fixe toutes les sources d'aléa pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Déterminisme complet (un peu plus lent)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device : {device}")
    return device


# ─── Checkpointing ─────────────────────────────────────────────

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss):
    """
    Sauvegarde complète de l'état d'entraînement.
    Permet de reprendre exactement où on s'est arrêté.
    """
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Charge un checkpoint.
    Si optimizer/scheduler sont fournis, leur état est aussi restauré (reprise d'entraînement).
    Retourne (epoch, best_val_loss) pour continuer l'entraînement.
    """
    if not os.path.isfile(path):
        return 0, float("inf")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    epoch = ckpt.get("epoch", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    print(f"[INFO] Checkpoint chargé depuis {path} (epoch {epoch}, best_val_loss {best_val_loss:.4f})")
    return epoch, best_val_loss
