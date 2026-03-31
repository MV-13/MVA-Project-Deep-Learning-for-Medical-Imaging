"""
Boucle d'entraînement avec early stopping et cosine LR scheduler.
"""

import numpy as np
import torch
import torchmetrics
from tqdm import tqdm


def train_one_epoch(model, dataloader, optimizer, criterion, metric, device):
    """Entraîne le modèle pendant une époque et retourne la loss et la métrique moyennes."""
    model.train()
    losses, metrics = [], []

    for x, y in tqdm(dataloader, desc="  Train", leave=False):
        optimizer.zero_grad()
        pred = model(x.to(device))
        loss = criterion(pred, y.to(device))
        loss.backward()
        optimizer.step()

        losses.extend([loss.item()] * len(y))
        m = metric(pred.cpu(), y.int().cpu())
        metrics.extend([m.item()] * len(y))

    return np.mean(losses), np.mean(metrics)


@torch.no_grad()
def validate(model, dataloader, criterion, metric, device):
    """Évalue le modèle sur le set de validation."""
    model.eval()
    losses, metrics = [], []

    for x, y in tqdm(dataloader, desc="  Valid", leave=False):
        pred = model(x.to(device))
        loss = criterion(pred, y.to(device))

        losses.extend([loss.item()] * len(y))
        m = metric(pred.cpu(), y.int().cpu())
        metrics.extend([m.item()] * len(y))

    return np.mean(losses), np.mean(metrics)


def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer_name,
    optimizer_params,
    loss_name,
    metric_name,
    num_epochs,
    patience,
    save_path,
    device,
):
    """
    Boucle d'entraînement complète avec early stopping et cosine LR scheduler.

    Args:
        model:                   le classifieur MLP à entraîner
        train_dataloader / val_dataloader: DataLoaders sur features pré-calculées
        optimizer_name:          nom de l'optimiseur (ex: 'AdamW')
        optimizer_params:        dict de paramètres (ex: {'lr': 3e-4, 'weight_decay': 1e-4})
        loss_name:               nom de la loss (ex: 'BCELoss')
        metric_name:             nom de la métrique torchmetrics (ex: 'Accuracy')
        num_epochs:              nombre max d'époques
        patience:                early stopping patience
        save_path:               chemin de sauvegarde du meilleur modèle
        device:                  device de calcul
    """
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **optimizer_params)
    # Cosine annealing : réduit le LR progressivement -> meilleure convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = getattr(torch.nn, loss_name)()
    metric = getattr(torchmetrics, metric_name)("binary")

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(num_epochs):
        train_loss, train_metric = train_one_epoch(
            model, train_dataloader, optimizer, criterion, metric, device
        )
        val_loss, val_metric = validate(model, val_dataloader, criterion, metric, device)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch [{epoch+1:3d}/{num_epochs}] "
            f"train | loss {train_loss:.4f} | {metric_name} {train_metric:.4f}  "
            f"val | loss {val_loss:.4f} | {metric_name} {val_metric:.4f}  "
            f"lr {lr:.2e}"
        )

        if val_loss < best_val_loss:
            print(f"  -> Nouveau meilleur modèle : {best_val_loss:.4f} -> {val_loss:.4f}")
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

        if epoch - best_epoch >= patience:
            print(f"Early stopping à l'époque {epoch+1} (patience={patience})")
            break

    print(f"\nMeilleur modèle : époque {best_epoch+1}, val_loss={best_val_loss:.4f}")
