"""
dataset.py — Dataset pipeline with WeightedRandomSampler for class imbalance.

Chest X-ray data structure expected:
    data/
    ├── train/
    │   ├── Normal/
    │   └── Pneumonia/
    └── test/
        ├── Normal/
        └── Pneumonia/
"""

import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


# ─── Transforms ───────────────────────────────────────────────────────────────

# ImageNet normalization stats (used because we fine-tune from ImageNet weights)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_transforms(img_size: int = 224, split: str = "train") -> transforms.Compose:
    """
    Return augmentation pipeline for train or val/test splits.

    Train augmentations (regularize):
        RandomCrop, HorizontalFlip, Rotation, ColorJitter

    Test (deterministic):
        Resize → ToTensor → Normalize
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    else:   # val / test — no random ops
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])


# ─── Loaders ──────────────────────────────────────────────────────────────────

def get_loaders(
    train_dir : str,
    test_dir  : str,
    batch_size: int = 32,
    img_size  : int = 224,
    num_workers: int = 4,
):
    """
    Build train and test DataLoaders.

    Class imbalance handling:
        WeightedRandomSampler assigns higher sampling probability to the
        minority class so each mini-batch is approximately balanced.

    Args:
        train_dir   : Path to training folder (ImageFolder structure)
        test_dir    : Path to test folder
        batch_size  : Mini-batch size
        img_size    : Spatial resolution for model input
        num_workers : DataLoader worker processes

    Returns:
        train_loader, test_loader, class_names (list[str])
    """
    train_dataset = datasets.ImageFolder(
        train_dir, transform=get_transforms(img_size, "train")
    )
    test_dataset = datasets.ImageFolder(
        test_dir, transform=get_transforms(img_size, "test")
    )

    # ── WeightedRandomSampler ─────────────────────────────────────────
    # class_counts[i] = number of samples in class i
    class_counts  = torch.tensor(
        [train_dataset.targets.count(i) for i in range(len(train_dataset.classes))],
        dtype=torch.float,
    )
    class_weights  = 1.0 / class_counts           # rare class → higher weight
    sample_weights = class_weights[torch.tensor(train_dataset.targets)]
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        sampler     = sampler,          # replaces shuffle=True
        num_workers = num_workers,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    print(f"[Dataset] Classes      : {train_dataset.classes}")
    print(f"[Dataset] Train samples: {len(train_dataset)}")
    print(f"[Dataset]  Test samples: {len(test_dataset)}")
    print(f"[Dataset] Class counts  (train): {dict(zip(train_dataset.classes, class_counts.int().tolist()))}")

    return train_loader, test_loader, train_dataset.classes
