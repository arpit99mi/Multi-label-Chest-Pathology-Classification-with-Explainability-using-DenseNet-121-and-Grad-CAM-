"""
train.py — Full training pipeline for Lung Disease Classifier.

Key features implemented:
    ✔ EfficientNet-B0 (pretrained ImageNet)
    ✔ WeightedRandomSampler (class imbalance)
    ✔ AdamW optimizer + CosineAnnealingLR scheduler
    ✔ Saves best checkpoint by validation loss
    ✔ Computes AUC-ROC, Sensitivity, Specificity, F1, CM, ROC curve
    ✔ Exports training history as JSON

Usage:
    python train.py
"""

import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm

from config import Config
from model import build_efficientnet_b0
from dataset import get_loaders
from utils import compute_metrics, set_seed


# ═══════════════════════════════════════════════════════════════════════════════
#  Epoch-level helpers
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model    : nn.Module,
    loader   ,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device   : str,
) -> tuple[float, float]:
    """
    One full pass over the training set.

    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    total_loss, correct, n_samples = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  [Train]", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping (stabilises training)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_size  = images.size(0)
        total_loss += loss.item() * batch_size
        correct    += (outputs.argmax(1) == labels).sum().item()
        n_samples  += batch_size

    return total_loss / n_samples, correct / n_samples


@torch.no_grad()
def evaluate(
    model    : nn.Module,
    loader   ,
    criterion: nn.Module,
    device   : str,
) -> tuple[float, float, list, list, list]:
    """
    Evaluate model on val/test set.

    Returns:
        (avg_loss, accuracy, y_true, y_pred, y_prob)
        y_prob is a list of [prob_class0, prob_class1] per sample.
    """
    model.eval()
    total_loss, correct, n_samples = 0.0, 0, 0
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="  [Eval] ", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss    = criterion(outputs, labels)
        probs   = torch.softmax(outputs, dim=1)

        batch_size  = images.size(0)
        total_loss += loss.item() * batch_size
        correct    += (outputs.argmax(1) == labels).sum().item()
        n_samples  += batch_size

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(outputs.argmax(1).cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    return total_loss / n_samples, correct / n_samples, all_labels, all_preds, all_probs


# ═══════════════════════════════════════════════════════════════════════════════
#  Main training loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = Config()
    set_seed(cfg.SEED)

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR,    exist_ok=True)

    print(f"Using device: {cfg.DEVICE}\n")

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, test_loader, classes = get_loaders(
        cfg.TRAIN_DIR, cfg.TEST_DIR, cfg.BATCH_SIZE, cfg.IMG_SIZE
    )

    # ── Model + Optimizer + Scheduler ─────────────────────────────────
    model = build_efficientnet_b0(pretrained=True, num_classes=cfg.NUM_CLASSES)
    model = model.to(cfg.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.LEARNING_RATE,
        weight_decay = cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = cfg.T_MAX,
        eta_min = cfg.ETA_MIN,
    )

    # ── Training loop ─────────────────────────────────────────────────
    best_val_loss = float("inf")
    history       = []

    print("=" * 70)
    print(f"  Starting training for {cfg.NUM_EPOCHS} epochs")
    print("=" * 70)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, cfg.DEVICE
        )
        val_loss, val_acc, y_true, y_pred, y_prob = evaluate(
            model, test_loader, criterion, cfg.DEVICE
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch [{epoch:02d}/{cfg.NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}% | "
            f"LR: {current_lr:.2e}"
        )

        history.append({
            "epoch"      : epoch,
            "train_loss" : round(train_loss, 6),
            "train_acc"  : round(train_acc,  6),
            "val_loss"   : round(val_loss,   6),
            "val_acc"    : round(val_acc,    6),
            "lr"         : round(current_lr, 8),
        })

        # ── Save best model checkpoint by val_loss ─────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch"             : epoch,
                    "model_state_dict"  : model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss"          : val_loss,
                    "val_acc"           : val_acc,
                    "classes"           : classes,
                    "config"            : {
                        "img_size"    : cfg.IMG_SIZE,
                        "num_classes" : cfg.NUM_CLASSES,
                        "batch_size"  : cfg.BATCH_SIZE,
                    },
                },
                cfg.BEST_MODEL_PATH,
            )
            print(f"  ✔ Best model saved  → val_loss: {val_loss:.4f}")

    # ── Final evaluation ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Final evaluation on test set (best checkpoint)")
    print("=" * 70)

    # Reload best weights for final metrics
    checkpoint = torch.load(cfg.BEST_MODEL_PATH, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, _, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion, cfg.DEVICE)
    metrics = compute_metrics(y_true, y_pred, y_prob, classes, cfg.RESULTS_DIR)

    print("\nTest set metrics:")
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))

    # ── Save training history ─────────────────────────────────────────
    with open(os.path.join(cfg.RESULTS_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Done! Artifacts saved to '{cfg.RESULTS_DIR}/' and '{cfg.CHECKPOINT_DIR}/'")
    print(f"   Run the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
