# 🫁 LungAI — Pneumonia Detection with EfficientNet-B0

> **Production-grade chest X-ray classifier** · Grad-CAM explainability · AUC-ROC evaluation · Streamlit UI

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## ✨ Features

| Component | Implementation |
|---|---|
| **Architecture** | EfficientNet-B0 (pretrained ImageNet) |
| **Metrics** | AUC-ROC, Sensitivity, Specificity, F1, Confusion Matrix, ROC Curve |
| **Class Imbalance** | `WeightedRandomSampler` |
| **Optimizer** | AdamW + CosineAnnealingLR |
| **Best Model** | Saved by best `val_loss` checkpoint |
| **Explainability** | Grad-CAM on `model.features[-1]` |
| **API Confidence** | Label + probability for **both** classes |
| **Documentation** | Full `MODEL_CARD.md` with limitations & ethics |

---

## 📁 Project Structure

```
lung_disease_classifier/
├── app.py              ← Streamlit application (run this!)
├── train.py            ← Training pipeline
├── model.py            ← EfficientNet-B0 architecture
├── dataset.py          ← DataLoader + WeightedRandomSampler
├── utils.py            ← Grad-CAM + evaluation metrics
├── config.py           ← All hyperparameters & paths
├── requirements.txt
├── README.md
├── MODEL_CARD.md       ← Full model documentation
│
├── data/               ← ⚠️ Populate with your dataset
│   ├── train/
│   │   ├── Normal/
│   │   └── Pneumonia/
│   └── test/
│       ├── Normal/
│       └── Pneumonia/
│
├── checkpoints/        ← Auto-created during training
│   └── best_model.pth
│
└── results/            ← Auto-created during training
    ├── eval_results.json
    ├── history.json
    ├── confusion_matrix.png
    └── roc_curve.png
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate.bat    # Windows

pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Place your chest X-ray images in the following structure:

```
data/
├── train/
│   ├── Normal/      ← Normal chest X-rays (.jpg/.png)
│   └── Pneumonia/   ← Pneumonia X-rays
└── test/
    ├── Normal/
    └── Pneumonia/
```

> **Recommended dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) on Kaggle (5,863 images).

### 3. Train the Model

```bash
python train.py
```

Training will:
- Load EfficientNet-B0 with ImageNet weights
- Apply `WeightedRandomSampler` for class balance
- Optimize with AdamW + CosineAnnealingLR
- Save best checkpoint to `checkpoints/best_model.pth`
- Compute and save all metrics to `results/`

Expected output:
```
Epoch [01/25] Train Loss: 0.4231  Acc: 82.14% | Val Loss: 0.3102  Acc: 87.50% | LR: 9.97e-04
  ✔ Best model saved  → val_loss: 0.3102
...
✅ Done! Artifacts saved to 'results/' and 'checkpoints/'
```

### 4. Launch the App

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## 🎛️ Configuration

All hyperparameters are centralized in `config.py`:

```python
class Config:
    BATCH_SIZE    = 32
    NUM_EPOCHS    = 25
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY  = 1e-4
    T_MAX         = 25      # CosineAnnealingLR period
    IMG_SIZE      = 224
```

---

## 🏗️ Architecture

```
Input (224×224×3)
    │
    ▼
EfficientNet-B0 Feature Extractor
    │  (pretrained ImageNet weights)
    │  features[-1] ← Grad-CAM hook
    ▼
Global Average Pooling (1280-dim)
    │
    ▼
Dropout(p=0.3)
    │
    ▼
Linear(1280 → 2)
    │
    ▼
Softmax → [P(Normal), P(Pneumonia)]
```

---

## 📊 Evaluation Metrics Explained

| Metric | Formula | Why it matters |
|---|---|---|
| **AUC-ROC** | Area under ROC | Threshold-independent discriminability |
| **Sensitivity** | TP/(TP+FN) | Catch all sick patients (↓ false negatives) |
| **Specificity** | TN/(TN+FP) | Avoid over-diagnosis (↓ false positives) |
| **F1-Score** | 2PR/(P+R) | Balance precision & recall |

---

## 🌡️ Grad-CAM Explainability

Grad-CAM highlights **which pixels influenced the prediction** by:

1. Forward pass → store activations at `model.features[-1]`
2. Backward pass → store gradients w.r.t. those activations
3. Compute weights: α_k = GAP(∂score/∂A_k)
4. CAM = ReLU(Σ α_k · A_k), then resize + normalize

The heatmap is blended (α=0.45) over the original image using the JET colormap.
**Red = high activation** → where the model looked to decide.

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**. It is **not** a medical device and should **not** be used as a substitute for professional medical diagnosis. Always consult a qualified healthcare professional.

---

## 📜 License

MIT © 2026 — See [MODEL_CARD.md](MODEL_CARD.md) for full ethical guidelines.
