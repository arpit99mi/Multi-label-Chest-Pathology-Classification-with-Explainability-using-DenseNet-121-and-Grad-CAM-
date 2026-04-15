"""
config.py — Central configuration for Lung Disease Classifier
All hyperparameters, paths, and device settings live here.
"""

import os
import torch


class Config:
    # ── Data ──────────────────────────────────────────────────────────
    TRAIN_DIR = os.path.join("data", "train")
    TEST_DIR  = os.path.join("data", "test")
    CLASSES   = ["Normal", "Pneumonia"]   # must match subfolder names
    NUM_CLASSES = 2

    # ── Image ─────────────────────────────────────────────────────────
    IMG_SIZE = 224   # EfficientNet-B0 default

    # ── Training ──────────────────────────────────────────────────────
    BATCH_SIZE    = 32
    NUM_EPOCHS    = 10
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY  = 1e-4

    # CosineAnnealingLR schedule
    T_MAX   = 25       # number of epochs (full period)
    ETA_MIN = 1e-6     # minimum learning rate

    # ── Paths ─────────────────────────────────────────────────────────
    CHECKPOINT_DIR  = "checkpoints"
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    RESULTS_DIR     = "results"

    # ── Device ────────────────────────────────────────────────────────
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Reproducibility ───────────────────────────────────────────────
    SEED = 42
