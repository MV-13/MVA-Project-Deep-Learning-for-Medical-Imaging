"""
Entraînement v5 — Simple et efficace.

Retour à la simplicité :
  - BCEWithLogitsLoss + label smoothing
  - Cosine annealing
  - SWA en fin d'entraînement
  - Mixup léger
  - Gradient clipping
  - Pas de focal loss, pas de multi-head, pas de CutMix
  
Avec des features Phikon-v2, un classifieur simple + bon training
suffit largement.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import roc_auc_score, accuracy_score

import config
from utils import save_checkpoint, load_checkpoint, find_optimal_threshold


def _mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]


def _smooth_labels(labels, smoothing=0.05):
    return labels * (1.0 - smoothing) + 0.5 * smoothing


def _run_epoch(model, loader, criterion, device, optimizer=None,
               smoothing=0.0, use_mixup=False, mixup_alpha=0.2):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []
    total = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if is_train and use_mixup:
                x, y = _mixup_data(x, y, alpha=mixup_alpha)

            targets = _smooth_labels(y, smoothing) if is_train and smoothing > 0 else y
            logits = model(x)
            loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(y)
            total += len(y)
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            all_preds.extend(probs.tolist())
            all_labels.extend(y.cpu().numpy().flatten().tolist())

    avg_loss = total_loss / total
    arr_p = np.array(all_preds)
    arr_l = np.array(all_labels)
    acc = accuracy_score(arr_l > 0.5, arr_p > 0.5)
    try:
        auc = roc_auc_score(arr_l > 0.5, arr_p)
    except ValueError:
        auc = 0.0
    return avg_loss, acc, auc


@torch.no_grad()
def _validate_with_probs(model, val_loader, criterion, device):
    """Validation + retour des probs pour seuil optimal."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    total = 0

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        total += len(y)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_preds.extend(probs.tolist())
        all_labels.extend(y.cpu().numpy().flatten().tolist())

    avg_loss = total_loss / total
    arr_p = np.array(all_preds)
    arr_l = np.array(all_labels)
    acc = accuracy_score(arr_l > 0.5, arr_p > 0.5)
    try:
        auc = roc_auc_score(arr_l > 0.5, arr_p)
    except ValueError:
        auc = 0.0
    return avg_loss, acc, auc, arr_l, arr_p


def train(model, train_loader, val_loader, device, resume_path=None):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR,
                                  weight_decay=config.WEIGHT_DECAY)

    scheduler = CosineAnnealingLR(optimizer, T_max=config.T_MAX) if config.USE_SCHEDULER else None

    swa_model = AveragedModel(model) if config.USE_SWA else None
    swa_scheduler = SWALR(optimizer, swa_lr=config.SWA_LR) if config.USE_SWA else None

    start_epoch = 0
    best_val_loss = float("inf")
    if resume_path is not None:
        start_epoch, best_val_loss = load_checkpoint(
            resume_path, model, optimizer, scheduler, device)
        start_epoch += 1

    epochs_no_improve = 0
    best_val_acc = 0.0
    swa_active = False

    print(f"\n{'Ep':>4} {'lr':>9} │ {'t_loss':>7} {'t_acc':>6} {'t_auc':>6} │ "
          f"{'v_loss':>7} {'v_acc':>6} {'v_auc':>6} │ Note")
    print("─" * 82)

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        if config.USE_SWA and epoch >= config.SWA_START_EPOCH:
            swa_active = True

        train_loss, train_acc, train_auc = _run_epoch(
            model, train_loader, criterion, device, optimizer,
            smoothing=config.LABEL_SMOOTHING,
            use_mixup=config.USE_MIXUP, mixup_alpha=config.MIXUP_ALPHA)

        if swa_active:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        elif scheduler is not None:
            scheduler.step()

        val_loss, val_acc, val_auc = _run_epoch(model, val_loader, criterion, device)

        lr_now = optimizer.param_groups[0]["lr"]
        note = " SWA" if swa_active else ""

        save_checkpoint(config.LAST_MODEL_PATH, model, optimizer, scheduler,
                        epoch, best_val_loss, best_val_acc)

        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_checkpoint(config.BEST_MODEL_PATH, model, optimizer, scheduler,
                            epoch, best_val_loss, best_val_acc)
            note += f" ★ (↓{improvement:.4f})"
        else:
            epochs_no_improve += 1

        print(f"{epoch+1:4d} {lr_now:9.2e} │ {train_loss:7.4f} {train_acc:6.4f} {train_auc:6.4f} │ "
              f"{val_loss:7.4f} {val_acc:6.4f} {val_auc:6.4f} │{note}")

        if not swa_active and epochs_no_improve >= config.PATIENCE:
            print(f"\n[INFO] Early stopping (patience={config.PATIENCE})")
            break

    # Finaliser SWA
    if config.USE_SWA and swa_model is not None:
        print("\n[INFO] Finalisation SWA (update BN)...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        torch.save({"model_state_dict": swa_model.module.state_dict()},
                    config.SWA_MODEL_PATH)
        print(f"  → {config.SWA_MODEL_PATH}")

    # Seuil optimal
    optimal_threshold = config.THRESHOLD
    best_candidate = "best_ckpt"

    if config.USE_OPTIMAL_THRESHOLD:
        print("\n[INFO] Recherche du seuil optimal...")
        import os

        candidates = [("best_ckpt", config.BEST_MODEL_PATH)]
        if config.USE_SWA:
            candidates.append(("swa", config.SWA_MODEL_PATH))

        best_overall_acc = 0.0

        for name, path in candidates:
            if not os.path.isfile(path):
                continue
            ckpt = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            _, acc, auc, labels_arr, probs_arr = _validate_with_probs(
                model, val_loader, criterion, device)
            thresh, thresh_acc = find_optimal_threshold(labels_arr > 0.5, probs_arr)
            print(f"  {name:12s} : acc={acc:.4f}  auc={auc:.4f}  "
                  f"thresh={thresh:.3f} → acc@thresh={thresh_acc:.4f}")

            if thresh_acc > best_overall_acc:
                best_overall_acc = thresh_acc
                optimal_threshold = thresh
                best_candidate = name

        print(f"\n  → Meilleur : {best_candidate}  seuil={optimal_threshold:.3f}")

    return best_val_loss, optimal_threshold, best_candidate
