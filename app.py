"""
app.py — LungAI: Pneumonia Detection Streamlit Application

Features:
    ✔ EfficientNet-B0 inference with probability scores for both classes
    ✔ Grad-CAM heatmap overlay for explainability
    ✔ Evaluation metrics dashboard (AUC-ROC, Sensitivity, Specificity, F1)
    ✔ Confusion matrix + ROC curve display
    ✔ Full Model Card viewer
    ✔ Drag-and-drop chest X-ray upload

Usage:
    streamlit run app.py
"""

import os
import json

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from config import Config
from model import build_efficientnet_b0
from utils import GradCAM, overlay_gradcam


# ═══════════════════════════════════════════════════════════════════════════════
#  Page Configuration
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
    <style>
        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
        }

        /* Optional: make sidebar text dark so it's readable on white */
        [data-testid="stSidebar"] * {
            color: #1a1a2e !important;
        }

        /* Optional: sidebar border for clean separation */
        [data-testid="stSidebar"] {
            border-right: 1px solid #e0e0e0;
        }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title = "LungAI — Pneumonia Detector",
    page_icon  = "🫁",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Custom CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ── Base ──────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

    .block-container { padding-top: 1.5rem; }

    /* ── Hero header ───────────────────────────────────────────────── */
    .hero {
        background: linear-gradient(135deg, #0a0a1a 0%, #111130 40%, #0f2044 100%);
        padding: 2.4rem 2rem;
        border-radius: 18px;
        text-align: center;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(99, 179, 237, 0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: "";
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(ellipse at center, rgba(66,153,225,0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero h1  { color: #ebf8ff; font-size: 2.6rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
    .hero .sub { color: #90cdf4; font-size: 1.05rem; margin: 0.6rem 0 0 0; }
    .hero .tags { margin-top: 0.9rem; }
    .tag {
        display: inline-block;
        background: rgba(66,153,225,0.18);
        color: #90cdf4;
        border: 1px solid rgba(66,153,225,0.3);
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.78rem;
        margin: 0.2rem;
        font-weight: 600;
    }

    /* ── Upload zone ───────────────────────────────────────────────── */
    .upload-zone {
        border: 2px dashed rgba(99,179,237,0.45);
        border-radius: 14px;
        padding: 3.5rem 2rem;
        text-align: center;
        color: #718096;
        background: rgba(66,153,225,0.04);
        transition: border-color 0.2s;
    }
    .upload-zone h3 { color: #a0aec0; margin: 0 0 0.4rem 0; }

    /* ── Result cards ──────────────────────────────────────────────── */
    .result-normal {
        background: linear-gradient(135deg, #1a3a2a, #22543d);
        border: 2px solid #48bb78;
        border-radius: 14px;
        padding: 1.6rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(72,187,120,0.25);
    }
    .result-normal h2 { color: #9ae6b4; font-size: 1.9rem; margin: 0; }
    .result-normal p  { color: #68d391; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    .result-pneumonia {
        background: linear-gradient(135deg, #3d1515, #742a2a);
        border: 2px solid #fc8181;
        border-radius: 14px;
        padding: 1.6rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(252,129,129,0.25);
    }
    .result-pneumonia h2 { color: #feb2b2; font-size: 1.9rem; margin: 0; }
    .result-pneumonia p  { color: #fc8181; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    /* ── Confidence bar labels ─────────────────────────────────────── */
    .conf-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        margin-bottom: 0.2rem;
        font-weight: 600;
    }

    /* ── Metric card ───────────────────────────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, #0d1117, #161b22);
        border: 1px solid rgba(99,179,237,0.25);
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        text-align: center;
    }
    .metric-card .val { font-size: 1.8rem; font-weight: 800; color: #63b3ed; }
    .metric-card .lbl { font-size: 0.78rem; color: #718096; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.15rem; }

    /* ── Sidebar ───────────────────────────────────────────────────── */
    [data-testid="stSidebar"] { background: #0d1117; }

    /* ── Section headers ───────────────────────────────────────────── */
    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #bee3f8;
        margin-bottom: 0.8rem;
        border-left: 3px solid #4299e1;
        padding-left: 0.6rem;
    }

    /* ── Warning banner ────────────────────────────────────────────── */
    .warning-banner {
        background: rgba(237, 137, 54, 0.12);
        border: 1px solid rgba(237, 137, 54, 0.4);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        font-size: 0.82rem;
        color: #fbd38d;
        margin-top: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Cached Resources
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    cfg   = Config()
    model = build_efficientnet_b0(pretrained=False, num_classes=cfg.NUM_CLASSES)

    checkpoint_meta = {}
    if os.path.exists(cfg.BEST_MODEL_PATH):
        ckpt = torch.load(cfg.BEST_MODEL_PATH, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        checkpoint_meta = {
            "epoch"    : ckpt.get("epoch",    "—"),
            "val_loss" : ckpt.get("val_loss", None),
            "val_acc"  : ckpt.get("val_acc",  None),
        }
    else:
        st.warning("⚠️ No trained checkpoint found. Run `python train.py` first.")

    model.to(cfg.DEVICE).eval()
    return model, cfg, checkpoint_meta


@st.cache_data
def load_eval_results():
    path = os.path.join("results", "eval_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_training_history():
    path = os.path.join("results", "history.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Inference Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess(pil_img: Image.Image, img_size: int = 224) -> torch.Tensor:
    """Convert PIL image → normalized (1, 3, H, W) tensor."""
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return tfm(pil_img.convert("RGB")).unsqueeze(0)


def run_inference(model, tensor, device, classes):
    """Return (predicted_label, probs_array, predicted_idx)."""
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx   = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    return pred_label, probs, pred_idx


def generate_gradcam(model, tensor, device, pil_img, pred_idx, img_size):
    """Generate Grad-CAM overlay as np.ndarray (H×W×3 uint8 RGB)."""
    tensor  = tensor.to(device)
    cam_gen = GradCAM(model)
    cam     = cam_gen.generate(tensor, class_idx=pred_idx)
    cam_gen.remove_hooks()

    orig_np = np.array(pil_img.convert("RGB").resize((img_size, img_size)))
    overlay = overlay_gradcam(orig_np, cam, alpha=0.45)
    return overlay


# ═══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar(eval_results, checkpoint_meta):
    with st.sidebar:
        st.markdown("## 🫁 LungAI")
        st.caption("Pneumonia Detection · Grad-CAM XAI")
        st.markdown("---")

        # ── Model info ────────────────────────────────────────────────
        st.markdown("### 🧠 Model")
        st.markdown("""
        | | |
        |---|---|
        | **Arch** | EfficientNet-B0 |
        | **Pretrain** | ImageNet |
        | **Task** | Binary Cls. |
        | **Optimizer** | AdamW |
        | **Scheduler** | CosineAnnealing |
        """)

        # ── Checkpoint info ───────────────────────────────────────────
        if checkpoint_meta.get("val_loss") is not None:
            st.markdown("### 📦 Best Checkpoint")
            col1, col2 = st.columns(2)
            col1.metric("Val Loss", f"{checkpoint_meta['val_loss']:.4f}")
            col2.metric("Val Acc",  f"{checkpoint_meta['val_acc']*100:.1f}%")
            st.caption(f"Saved at epoch {checkpoint_meta['epoch']}")

        # ── Eval metrics ─────────────────────────────────────────────
        if eval_results:
            st.markdown("### 🏆 Test Metrics")
            m1, m2 = st.columns(2)
            m1.metric("AUC-ROC",     f"{eval_results['auc_roc']:.3f}")
            m2.metric("F1-Score",    f"{eval_results['f1_score']:.3f}")
            m1.metric("Sensitivity", f"{eval_results['sensitivity']:.3f}")
            m2.metric("Specificity", f"{eval_results['specificity']:.3f}")

        st.markdown("---")
        st.markdown("### 🗂 Data Structure")
        st.code("""data/
├── train/
│   ├── Normal/
│   └── Pneumonia/
└── test/
    ├── Normal/
    └── Pneumonia/""", language="text")

        st.markdown('<div class="warning-banner">⚕️ <b>Disclaimer:</b> For research & educational purposes only. Not a medical device.</div>',
                    unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 1: Diagnose
# ═══════════════════════════════════════════════════════════════════════════════

def render_diagnose_tab(model, cfg):
    st.markdown('<div class="section-header">Upload a Chest X-Ray Image</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag & drop or click to browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if not uploaded:
        st.markdown("""
        <div class="upload-zone">
            <h3>📤 Drop your chest X-ray here</h3>
            <p>Supports JPG · JPEG · PNG</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Run inference ──────────────────────────────────────────────────
    pil_img    = Image.open(uploaded)
    tensor     = preprocess(pil_img, cfg.IMG_SIZE)
    pred_label, probs, pred_idx = run_inference(model, tensor, cfg.DEVICE, cfg.CLASSES)

    with st.spinner("Generating Grad-CAM heatmap…"):
        overlay = generate_gradcam(model, tensor, cfg.DEVICE, pil_img, pred_idx, cfg.IMG_SIZE)

    # ── Layout: Original | Grad-CAM | Results ─────────────────────────
    col1, col2, col3 = st.columns([1.1, 1.1, 1.0], gap="medium")

    with col1:
        st.markdown('<div class="section-header">🔬 Original X-Ray</div>', unsafe_allow_html=True)
        st.image(
            pil_img.convert("RGB").resize((300, 300)),
            caption="Uploaded chest X-ray",
            use_container_width=True,
        )

    with col2:
        st.markdown('<div class="section-header">🌡️ Grad-CAM Heatmap</div>', unsafe_allow_html=True)
        st.image(
            overlay,
            caption="Red = high activation (model focus area)",
            use_container_width=True,
        )
        st.caption("🔴 Warm regions → where EfficientNet looked to make its decision")

    with col3:
        st.markdown('<div class="section-header">🩺 Prediction</div>', unsafe_allow_html=True)

        is_pneumonia = pred_label == "Pneumonia"
        if is_pneumonia:
            st.markdown("""
            <div class="result-pneumonia">
                <h2>⚠️ Pneumonia</h2>
                <p>Signs detected in X-ray</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-normal">
                <h2>✅ Normal</h2>
                <p>No pneumonia detected</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Confidence Scores</div>', unsafe_allow_html=True)

        # Confidence bars for both classes
        colors = {"Normal": "#48bb78", "Pneumonia": "#fc8181"}
        for i, cls in enumerate(cfg.CLASSES):
            prob_pct = probs[i] * 100
            color    = colors[cls]
            icon     = "🟢" if cls == "Normal" else "🔴"

            st.markdown(
                f'<div class="conf-label">'
                f'<span>{icon} {cls}</span>'
                f'<span style="color:{color};font-size:1rem;">{prob_pct:.2f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(float(probs[i]))

        st.markdown("<br>", unsafe_allow_html=True)

        # Raw probabilities table
        st.markdown("**Raw Probabilities**")
        import pandas as pd
        prob_df = pd.DataFrame({
            "Class"       : cfg.CLASSES,
            "Probability" : [f"{p*100:.4f}%" for p in probs],
            "Logit-prob"  : [f"{p:.6f}"      for p in probs],
        })
        st.dataframe(prob_df, hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 2: Performance Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def render_performance_tab(eval_results):
    st.markdown('<div class="section-header">📊 Test Set Evaluation Metrics</div>', unsafe_allow_html=True)

    if not eval_results:
        st.info("No results found. Train the model first:  `python train.py`")
        return

    # ── Metric cards ──────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "🎯 AUC-ROC",     eval_results["auc_roc"]),
        (c2, "📐 F1-Score",    eval_results["f1_score"]),
        (c3, "🔍 Sensitivity", eval_results["sensitivity"]),
        (c4, "🛡️ Specificity", eval_results["specificity"]),
        (c5, "✅ Accuracy",    eval_results["accuracy"]),
    ]
    for col, label, val in cards:
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="val">{val:.4f}</div>'
            f'<div class="lbl">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Plots ─────────────────────────────────────────────────────────
    pc1, pc2 = st.columns(2, gap="large")

    with pc1:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm_path = os.path.join("results", "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.caption("Run training to generate this plot.")

    with pc2:
        st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
        roc_path = os.path.join("results", "roc_curve.png")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
        else:
            st.caption("Run training to generate this plot.")

    # ── Metric explanations ───────────────────────────────────────────
    with st.expander("📖 What do these metrics mean?"):
        st.markdown("""
| Metric | Formula | Clinical Meaning |
|---|---|---|
| **AUC-ROC** | Area under ROC curve | Overall discriminability (1.0 = perfect) |
| **Sensitivity** | TP / (TP + FN) | How well we *catch* Pneumonia cases (avoid false negatives) |
| **Specificity** | TN / (TN + FP) | How well we *rule out* Pneumonia (avoid false positives) |
| **F1-Score** | 2·P·R / (P+R) | Harmonic mean of precision and recall |
| **AUC-ROC** | Area under curve | Higher = better separation between classes |

> ⚕️ In medical imaging, **Sensitivity** is often prioritised to minimise missed cases.
        """)

    # ── Training curve ────────────────────────────────────────────────
    history = load_training_history()
    if history:
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.DataFrame(history)
        st.markdown('<div class="section-header">Training History</div>', unsafe_allow_html=True)
        tc1, tc2 = st.columns(2)

        with tc1:
            fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0d1117")
            ax.set_facecolor("#0d1117")
            ax.plot(df["epoch"], df["train_loss"], color="#63b3ed", label="Train", lw=2)
            ax.plot(df["epoch"], df["val_loss"],   color="#fc8181", label="Val",   lw=2)
            ax.set_xlabel("Epoch", color="#a0aec0"); ax.set_ylabel("Loss", color="#a0aec0")
            ax.set_title("Loss Curve", color="#e2e8f0", fontweight="bold")
            ax.tick_params(colors="#718096"); ax.legend(facecolor="#161b22", labelcolor="#e2e8f0")
            for spine in ax.spines.values(): spine.set_edgecolor("#2d3748")
            st.pyplot(fig)

        with tc2:
            fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0d1117")
            ax.set_facecolor("#0d1117")
            ax.plot(df["epoch"], df["train_acc"]*100, color="#63b3ed", label="Train", lw=2)
            ax.plot(df["epoch"], df["val_acc"]*100,   color="#fc8181", label="Val",   lw=2)
            ax.set_xlabel("Epoch", color="#a0aec0"); ax.set_ylabel("Accuracy (%)", color="#a0aec0")
            ax.set_title("Accuracy Curve", color="#e2e8f0", fontweight="bold")
            ax.tick_params(colors="#718096"); ax.legend(facecolor="#161b22", labelcolor="#e2e8f0")
            for spine in ax.spines.values(): spine.set_edgecolor("#2d3748")
            st.pyplot(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab 3: Model Card
# ═══════════════════════════════════════════════════════════════════════════════

def render_model_card_tab():
    card_path = "MODEL_CARD.md"
    if os.path.exists(card_path):
        # Uses utf-8, falls back gracefully if still issues
     # ✅ Fix: explicitly set UTF-8 encoding
     with open("MODEL_CARD.md", "r", encoding="utf-8") as f:
      content = f.read()
     st.markdown(content)  # or however you're rendering it
    else:
        st.info("MODEL_CARD.md not found in project root.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    model, cfg, checkpoint_meta = load_model()
    eval_results = load_eval_results()
    render_sidebar(eval_results, checkpoint_meta)

    # ── Hero header ────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>🫁 LungAI — Pneumonia Detector</h1>
        <p class="sub">Deep learning-powered chest X-ray analysis with visual explainability</p>
        <div class="tags">
            <span class="tag">EfficientNet-B0</span>
            <span class="tag">Grad-CAM XAI</span>
            <span class="tag">AUC-ROC Metrics</span>
            <span class="tag">WeightedSampler</span>
            <span class="tag">AdamW + CosineAnnealing</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🔬  Diagnose",
        "📊  Performance",
        "📘  Model Card",
    ])

    with tab1:
        render_diagnose_tab(model, cfg)

    with tab2:
        render_performance_tab(eval_results)

    with tab3:
        render_model_card_tab()


if __name__ == "__main__":
    main()
