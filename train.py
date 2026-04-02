"""
Boucle d'entraînement avec :
  - Early stopping
  - Cosine annealing scheduler
  - Sauvegarde du meilleur modèle ET du dernier checkpoint (reprise possible)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import config
from utils import save_checkpoint, load_checkpoint


def _run_epoch(model, loader, criterion, device, optimizer=None):
    """
    Exécute une passe (train ou val) sur tout le loader.
    Si optimizer est fourni → mode train ; sinon → mode eval.
    Retourne (loss_moyenne, accuracy_moyenne).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.no_grad() if not is_train else torch.enable_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            predicted = (pred > 0.5).float()
            correct += (predicted == y).sum().item()
            total += len(y)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train(model, train_loader, val_loader, device,
          resume_path=None):
    """
    Entraînement complet avec sauvegarde de checkpoints.

    Sauvegarde :
      - BEST_MODEL_PATH : meilleur modèle (selon val loss) → utilisé pour la prédiction finale
      - LAST_MODEL_PATH : dernier état complet (optimizer, scheduler, epoch) → reprise

    Args:
        model:        ClassificationHead
        train_loader: DataLoader sur PrecomputedDataset(train)
        val_loader:   DataLoader sur PrecomputedDataset(val)
        device:       'cuda' ou 'cpu'
        resume_path:  si fourni, reprend l'entraînement depuis ce checkpoint
    """
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR,
                                 weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.T_MAX) if config.USE_SCHEDULER else None

    # ─── Reprise éventuelle ────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")

    if resume_path is not None:
        start_epoch, best_val_loss = load_checkpoint(
            resume_path, model, optimizer, scheduler, device
        )
        start_epoch += 1  # on reprend à l'époque suivante
        print(f"[INFO] Reprise de l'entraînement à l'époque {start_epoch + 1}")

    epochs_no_improve = 0

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        # ─── Train ─────────────────────────────────────────────
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, device, optimizer)

        # ─── Validation ────────────────────────────────────────
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch+1:3d}/{config.NUM_EPOCHS}]  "
              f"lr={lr_now:.1e}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  |  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}", end="")

        # ─── Scheduler step ────────────────────────────────────
        if scheduler is not None:
            scheduler.step()

        # ─── Sauvegarde du dernier checkpoint (reprise) ────────
        save_checkpoint(config.LAST_MODEL_PATH, model, optimizer, scheduler,
                        epoch, best_val_loss)

        # ─── Meilleur modèle ───────────────────────────────────
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(config.BEST_MODEL_PATH, model, optimizer, scheduler,
                            epoch, best_val_loss)
            print(f"  ★ saved (↓{improvement:.4f})")
        else:
            epochs_no_improve += 1
            print()

        # ─── Early stopping ────────────────────────────────────
        if epochs_no_improve >= config.PATIENCE:
            print(f"\n[INFO] Early stopping (patience={config.PATIENCE}). "
                  f"Meilleur modèle à l'époque {epoch + 1 - config.PATIENCE}.")
            break

    print(f"\n[INFO] Entraînement terminé. Meilleure val_loss = {best_val_loss:.4f}")
    return best_val_loss
