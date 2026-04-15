# 🫁 LungAI — Model Card

> **Model Card version:** 1.0.0  
> **Last updated:** 2026  
> **Prepared by:** [Arpit Mishra / Data Scientist]

---

## 📋 Model Details

| Field | Value |
|---|---|
| **Model Name** | LungAI — Pneumonia Classifier |
| **Version** | 1.0.0 |
| **Architecture** | EfficientNet-B0 (fine-tuned) |
| **Pre-training** | ImageNet-1K (IMAGENET1K_V1) |
| **Task** | Binary Image Classification |
| **Input** | Chest X-ray image (RGB, 224×224) |
| **Output** | Label + probability for Normal and Pneumonia |
| **Framework** | PyTorch ≥ 2.0 |
| **Optimizer** | AdamW (lr=1e-3, weight_decay=1e-4) |
| **Scheduler** | CosineAnnealingLR (T_max=25, eta_min=1e-6) |
| **Explainability** | Grad-CAM on `model.features[-1]` |

---

## 🎯 Intended Use

### Primary Use Case
Assist radiologists and researchers in **screening** chest X-rays for signs of **pneumonia**.  
The model outputs a class label ("Normal" / "Pneumonia") along with **confidence probabilities for both classes**.

### Intended Users
- Medical researchers conducting retrospective studies
- Data science students learning medical AI
- AI developers building clinical decision support tools (with proper regulatory compliance)

### Out-of-Scope Uses
- ❌ **Not** a replacement for professional radiological diagnosis
- ❌ **Not** validated for paediatric-only or ICU-specific populations
- ❌ **Not** designed to detect disease types other than pneumonia
- ❌ **Not** approved for clinical use without regulatory clearance (e.g. FDA 510(k), CE mark)

---

## 📦 Data

### Dataset Structure
```
data/
├── train/
│   ├── Normal/       ← Normal chest X-rays
│   └── Pneumonia/    ← Pneumonia chest X-rays
└── test/
    ├── Normal/
    └── Pneumonia/
```

### Preprocessing
- Resize to 256×256 → RandomCrop to 224×224 (train only)
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Train augmentations: HorizontalFlip, Rotation(10°), ColorJitter

### Class Imbalance Handling
`WeightedRandomSampler` is used during training to up-sample minority class  
examples so each mini-batch is approximately balanced.

---

## 📊 Performance Metrics

> Metrics are computed on the held-out test set using the **best val_loss checkpoint**.

| Metric | Value |
|---|---|
| **AUC-ROC** | *Run `python train.py` to populate* |
| **Sensitivity (Recall)** | *TP / (TP + FN) — Pneumonia detection rate* |
| **Specificity** | *TN / (TN + FP) — Normal ruling-out rate* |
| **F1-Score** | *Weighted harmonic mean of precision & recall* |
| **Accuracy** | *Overall correct classifications* |

### Why these metrics?
- **AUC-ROC**: Threshold-independent; measures overall discriminability.
- **Sensitivity**: Critical in medicine — missed pneumonia (false negative) is dangerous.
- **Specificity**: Avoids unnecessary follow-up tests (false positives).
- **F1-Score**: Balances precision and recall; robust to class imbalance.
- **Confusion Matrix**: Full breakdown of TP / TN / FP / FN.

---

## 🔍 Explainability

### Grad-CAM
The model uses **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualise which regions of the X-ray influenced the prediction.

- **Target layer**: `model.features[-1]` (last convolutional block of EfficientNet-B0)
- **Heatmap overlay**: JET colormap blended at α=0.45 over the original image
- **Red regions** = high gradient-weighted activation → areas the model "focused on"

> Grad-CAM does **not** replace radiological interpretation but provides a sanity-check for model behaviour.

---

## ⚠️ Limitations

1. **Dataset Bias**: Performance depends heavily on the quality and diversity of training data. X-rays from specific imaging equipment, patient populations, or acquisition protocols may not generalise.

2. **Binary Scope**: The model only distinguishes "Normal" vs "Pneumonia". It cannot detect other pulmonary conditions (e.g., tuberculosis, lung cancer, COVID-19).

3. **Confidence Miscalibration**: Neural networks often produce overconfident predictions. Probability scores should not be treated as calibrated probabilities without temperature scaling or Platt scaling.

4. **Distribution Shift**: Model performance may degrade on X-rays from scanners, hospitals, or patient demographics not well-represented in training data.

5. **Paediatric vs Adult**: If the training set is skewed toward one age group, the model may underperform on the other.

6. **JPEG Compression Artifacts**: Heavy compression can degrade feature quality; standardise image acquisition where possible.

---

## ⚖️ Ethical Considerations

### Patient Privacy
- Training data should be fully de-identified and compliant with applicable regulations (HIPAA, GDPR).
- Model weights should not enable reconstruction of training images.

### Fairness & Bias
- The model has **not** been audited for performance disparities across:
  - Age groups, sex, ethnicity, or socioeconomic groups
  - Different imaging equipment manufacturers
- **Recommendation**: Conduct bias evaluations on stratified sub-populations before deployment.

### Human Oversight
- All predictions **must** be reviewed by a qualified healthcare professional.
- The model should function as a **second opinion** or **triage aid**, not an autonomous decision-maker.

### Transparency
- Model weights, training code, and evaluation results are provided to enable auditability.
- Users should document any fine-tuning performed on top of this base model.

---

## 🧪 Recommendations for Safe Deployment

1. ✅ **Calibrate** model probabilities (temperature scaling) before clinical use.
2. ✅ **Validate** on your local patient population before deployment.
3. ✅ **Monitor** for distribution drift over time (data drift detection).
4. ✅ **Log** all predictions with timestamps for audit trail.
5. ✅ **Display** confidence scores and Grad-CAM to the radiologist, not just the binary label.
6. ✅ **Maintain** a human-in-the-loop pipeline for all borderline predictions (e.g., 40%–60% probability range).

---

## 📎 Citation

If you use this model or codebase in your research, please cite:

```bibtex
@misc{lungai2026,
  title   = {LungAI: EfficientNet-B0 Pneumonia Classifier with Grad-CAM Explainability},
  author  = {[Your Name]},
  year    = {2026},
  url     = {https://github.com/[your-repo]},
}
```

---

## 📜 License

This model card and codebase are released under the **MIT License**.  
Pre-trained EfficientNet-B0 weights are subject to [torchvision's BSD 3-Clause License](https://github.com/pytorch/vision/blob/main/LICENSE).

---

*This model card follows the [Model Cards for Model Reporting (Mitchell et al., 2019)](https://arxiv.org/abs/1810.03993) framework.*
