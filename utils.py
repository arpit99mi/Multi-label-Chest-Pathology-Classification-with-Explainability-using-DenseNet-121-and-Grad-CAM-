"""
utils.py — Grad-CAM explainability + comprehensive evaluation metrics.

Metrics: AUC-ROC, Sensitivity (Recall/TPR), Specificity (TNR), F1, Confusion Matrix
Explainability: Grad-CAM on EfficientNet-B0 features[-1] layer
"""

import os
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix, roc_curve, classification_report,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Grad-CAM
# ═══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for EfficientNet-B0.

    Hooks are registered on `model.features[-1]` (last MBConv block).
    Forward pass stores activations; backward pass stores gradients.
    The CAM = ReLU( Σ_k α_k · A_k )   where α_k = GAP(∂score/∂A_k)

    Usage:
        cam_gen = GradCAM(model)
        heatmap = cam_gen.generate(input_tensor, class_idx=1)   # 0=Normal, 1=Pneumonia
    """

    def __init__(self, model: torch.nn.Module):
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._hooks      = []
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.features[-1]

        self._hooks.append(
            target_layer.register_forward_hook(
                lambda m, inp, out: setattr(self, "activations", out.detach())
            )
        )
        self._hooks.append(
            target_layer.register_full_backward_hook(
                lambda m, grad_in, grad_out: setattr(self, "gradients", grad_out[0].detach())
            )
        )

    def remove_hooks(self):
        """Call this after use to avoid memory leaks in repeated inference."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx   : int = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        Args:
            input_tensor : (1, C, H, W) — preprocessed, on the same device as model
            class_idx    : target class (None → argmax of predicted logits)

        Returns:
            np.ndarray: normalized heatmap in [0, 1], shape = (H, W)
        """
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        # α_k  =  (1/Z) Σ_{i,j} ∂score / ∂A^k_{ij}
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # L_c  =  ReLU( Σ_k α_k · A_k )
        cam = (weights * self.activations).sum(dim=1, keepdim=True)   # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()                              # (H, W)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)      # normalize to [0,1]
        return cam


def overlay_gradcam(
    original_img: np.ndarray,
    cam         : np.ndarray,
    alpha       : float = 0.45,
) -> np.ndarray:
    """
    Blend Grad-CAM heatmap (JET colormap) onto the original RGB image.

    Args:
        original_img : H×W×3 uint8 RGB array
        cam          : H×W float32 heatmap in [0, 1]
        alpha        : blend weight for heatmap (1-alpha for image)

    Returns:
        np.ndarray: blended RGB image (uint8)
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    overlay = np.uint8(alpha * heatmap + (1.0 - alpha) * original_img)
    return overlay


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true   : list,
    y_pred   : list,
    y_prob   : list,
    classes  : list,
    save_dir : str = "results",
) -> dict:
    """
    Compute and persist all evaluation metrics + plots.

    Metrics computed:
        • AUC-ROC          (area under ROC curve)
        • Sensitivity      (Recall / TPR for Pneumonia class)
        • Specificity      (TNR for Normal class)
        • F1-Score         (weighted)
        • Confusion Matrix (saved as PNG)
        • ROC Curve        (saved as PNG)

    Args:
        y_true   : ground-truth labels (list[int])
        y_pred   : predicted labels    (list[int])
        y_prob   : softmax probs       (list[list[float]])  shape: (N, 2)
        classes  : class name list     e.g. ["Normal", "Pneumonia"]
        save_dir : directory to save JSON + PNGs

    Returns:
        dict: metric name → value
    """
    os.makedirs(save_dir, exist_ok=True)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)    # (N, 2)

    # ── Core metrics ──────────────────────────────────────────────────
    auc = roc_auc_score(y_true, y_prob[:, 1])
    f1  = f1_score(y_true, y_pred, average="weighted")
    cm  = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn + 1e-8))   # recall for Pneumonia
    specificity = float(tn / (tn + fp + 1e-8))   # recall for Normal
    accuracy    = float((tp + tn) / (tp + tn + fp + fn))

    metrics = {
        "accuracy"         : round(accuracy,    4),
        "auc_roc"          : round(float(auc),  4),
        "f1_score"         : round(float(f1),   4),
        "sensitivity"      : round(sensitivity, 4),
        "specificity"      : round(specificity, 4),
        "confusion_matrix" : cm.tolist(),
    }

    # ── Save JSON ─────────────────────────────────────────────────────
    with open(os.path.join(save_dir, "eval_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────
    _plot_confusion_matrix(cm, classes, save_dir)
    _plot_roc_curve(y_true, y_prob[:, 1], float(auc), save_dir)

    return metrics


# ─── Private plot helpers ─────────────────────────────────────────────────────

def _plot_confusion_matrix(cm: np.ndarray, classes: list, save_dir: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_roc_curve(y_true: np.ndarray, y_prob_pos: np.ndarray, auc: float, save_dir: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"AUC = {auc:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.15, color="darkorange")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
