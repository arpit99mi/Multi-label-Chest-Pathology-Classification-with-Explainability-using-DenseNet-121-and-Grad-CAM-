"""
model.py — EfficientNet-B0 with custom classification head
Pretrained on ImageNet; fine-tuned for binary lung disease classification.
"""

import torch.nn as nn
import torchvision.models as models


def build_efficientnet_b0(pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """
    Build EfficientNet-B0 with a custom classification head.

    Architecture changes:
      Original head  → Dropout(0.2) → Linear(1280 → 1000)
      Replaced with  → Dropout(0.3) → Linear(1280 → num_classes)

    Args:
        pretrained  : Load ImageNet-1K weights (IMAGENET1K_V1)
        num_classes : Number of output logits (2 for Normal / Pneumonia)

    Returns:
        nn.Module: Modified EfficientNet-B0 ready for fine-tuning
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.efficientnet_b0(weights=weights)

    # ── Replace classifier head ────────────────────────────────────────
    in_features = model.classifier[1].in_features   # 1280 for B0
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def count_parameters(model: nn.Module) -> dict:
    """Return total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    model = build_efficientnet_b0(pretrained=False)
    params = count_parameters(model)
    print(f"Total params   : {params['total']:,}")
    print(f"Trainable params: {params['trainable']:,}")
    print(model.classifier)
