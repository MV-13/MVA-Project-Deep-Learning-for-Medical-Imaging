import copy
import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device : {device}")
    if device.type == "cuda":
        print(f"       GPU    : {torch.cuda.get_device_name(0)}")
        print(f"       VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device


def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=0,
                    best_val_loss=float("inf"), best_val_acc=None):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    if not os.path.isfile(path):
        print(f"[WARN] Pas de checkpoint : {path}")
        return 0, float("inf")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    epoch = ckpt.get("epoch", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    print(f"[INFO] Checkpoint chargé : {path}  (epoch={epoch}, loss={best_val_loss:.4f})")
    return epoch, best_val_loss


class ModelEMA:
    """Exponential Moving Average des poids."""

    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
        for ema_b, model_b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data)

    def state_dict(self):
        return self.ema_model.state_dict()


def find_optimal_threshold(labels, probs):
    """Cherche le seuil qui maximise l'accuracy."""
    best_thresh = 0.5
    best_score = 0.0
    for t in np.linspace(0.2, 0.8, 200):
        preds = (probs > t).astype(int)
        score = accuracy_score(labels, preds)
        if score > best_score:
            best_score = score
            best_thresh = t
    return best_thresh, best_score
