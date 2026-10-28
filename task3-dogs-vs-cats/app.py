"""
Interactive Streamlit Web Application for Task 3: Dogs vs. Cats Classification.

Uses the trained HOG + SVM pipeline (StandardScaler + RBF Kernel SVC) to classify
uploaded or sample images, visualizing the 64x64 grayscale normalization and HOG gradient maps.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

# Ensure parent directory is in sys.path
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from src.feature_extractor import HOGConfig, HOGFeatureExtractor, INV_LABEL_MAP, LABEL_MAP

# -----------------------------------------------------------------------------
# Page Configuration & Modern Aesthetics Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dogs vs Cats AI Classifier | SkillCraft ML",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Metric & Card Aesthetics */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    .prediction-badge-cat {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        color: #ffffff !important;
        padding: 12px 24px;
        border-radius: 10px;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.4);
    }
    .prediction-badge-dog {
        background: linear-gradient(135deg, #9a3412, #f97316);
        color: #ffffff !important;
        padding: 12px 24px;
        border-radius: 10px;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 14px rgba(249, 115, 22, 0.4);
    }
    .info-callout {
        background-color: rgba(59, 130, 246, 0.08);
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 4px 8px 8px 4px;
        margin-top: 15px;
        font-size: 0.9rem;
    }
    .warning-callout {
        background-color: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 4px 8px 8px 4px;
        margin-top: 15px;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Cached Resource Loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained HOG + SVM model...")
def load_model_and_config() -> Tuple[Optional[Any], Optional[HOGConfig], Optional[str]]:
    """Safely load serialized pipeline and HOG configuration."""
    model_path = _current_dir / "models" / "best_svm_pipeline.joblib"
    config_path = _current_dir / "models" / "hog_config.json"

    if not model_path.exists():
        return None, None, f"Model artifact not found at: {model_path}"

    if not config_path.exists():
        return None, None, f"HOG configuration not found at: {config_path}"

    try:
        pipeline = joblib.load(model_path)
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        hog_config = HOGConfig.from_dict(config_data)
        return pipeline, hog_config, None
    except Exception as ex:
        return None, None, f"Error loading model: {ex}"


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def validate_uploaded_image(uploaded_file, max_size_mb: int = 10) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Validate uploaded image size and integrity using Pillow."""
    file_bytes = uploaded_file.getvalue()

    if len(file_bytes) > max_size_mb * 1024 * 1024:
        return None, f"File exceeds maximum allowed size ({max_size_mb} MB). Upload a smaller image."

    try:
        # Verify file structure without leaving handle open
        img_check = Image.open(io.BytesIO(file_bytes))
        img_check.verify()

        # Reopen for actual display and processing
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return img, None
    except (UnidentifiedImageError, OSError, SyntaxError) as ex:
        return None, f"Corrupted or unsupported image file: {ex}"


def create_hog_colormap_image(hog_image: np.ndarray) -> Image.Image:
    """Render HOG gradient visualization with an interpretable colormap."""
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.imshow(hog_image, cmap="inferno")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
def main() -> None:
    # Header Banner
    st.title("🐾 Dogs vs. Cats AI Image Classifier")
    st.markdown(
        "**Classical Computer Vision Pipeline** featuring **Histogram of Oriented Gradients (HOG)** "
        "descriptors coupled with a **Support Vector Machine (RBF Kernel SVM)**."
    )
    st.divider()

    # Load Model Pipeline
    pipeline, hog_config, err = load_model_and_config()

    # -------------------------------------------------------------------------
    # Sidebar: Model Architecture & Specifications
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Model Architecture")

        if err:
            st.error(f"⚠️ {err}")
            st.stop()
        else:
            st.success("✅ Model Pipeline Loaded & Active")

        st.markdown(
            """
            - **Classifier**: `StandardScaler + RBF SVC` ($C=1.0$)
            - **Input Grid**: $64 \\times 64$ Grayscale
            - **Orientations**: 9 Gradient Bins ($0^\\circ$–$180^\\circ$)
            - **Cell Geometry**: $8 \\times 8$ Pixels / Cell
            - **Block Normalization**: $2 \\times 2$ Cells (`L2-Hys`)
            - **Feature Dimension**: **1,764 floats** / image
            """
        )

        st.subheader("📊 Benchmark Test Results")
        st.markdown(
            """
            - **Test Accuracy**: **72.50%**
            - **Test Macro F1**: **72.50%**
            - **Test ROC-AUC**: **0.7856**
            """
        )
        st.caption(
            "*Evaluated on held-out benchmark test partition of 600 images (300 cats, 300 dogs). "
            "Reported for transparency; not a performance guarantee for wild photos.*"
        )

        st.divider()
        st.subheader("🖼️ Select Image Source")
        source_mode = st.radio(
            "Input Mode:",
            options=["Upload Custom Image", "Try Benchmark Test Sample"],
            index=0,
        )

        sample_choice = None
        if source_mode == "Try Benchmark Test Sample":
            test_samples = {
                "🐱 Cat Sample A (cat.10074.jpg)": "data/interim/test/cat/cat.10074.jpg",
                "🐱 Cat Sample B (cat.10277.jpg)": "data/interim/test/cat/cat.10277.jpg",
                "🐱 Cat Sample C (cat.8177.jpg)": "data/interim/train/cat/cat.8177.jpg",
                "🐶 Dog Sample A (dog.10017.jpg)": "data/interim/test/dog/dog.10017.jpg",
                "🐶 Dog Sample B (dog.10113.jpg)": "data/interim/test/dog/dog.10113.jpg",
                "🐶 Dog Sample C (dog.10163.jpg)": "data/interim/test/dog/dog.10163.jpg",
            }
            selected_label = st.selectbox("Choose a sample:", list(test_samples.keys()))
            sample_choice = test_samples[selected_label]

    # -------------------------------------------------------------------------
    # Image Acquisition & Validation
    # -------------------------------------------------------------------------
    input_img: Optional[Image.Image] = None
    input_name: str = ""

    if source_mode == "Upload Custom Image":
        uploaded_file = st.file_uploader(
            "Upload an image (JPG, JPEG, or PNG):",
            type=["jpg", "jpeg", "png"],
            help="Maximum supported file size: 10 MB",
        )
        if uploaded_file is not None:
            input_img, val_err = validate_uploaded_image(uploaded_file, max_size_mb=10)
            if val_err:
                st.error(f"❌ {val_err}")
                return
            input_name = uploaded_file.name
    else:
        sample_path = _current_dir / sample_choice
        if sample_path.exists():
            try:
                input_img = Image.open(sample_path).convert("RGB")
                input_name = sample_path.name
            except Exception as ex:
                st.error(f"❌ Failed loading sample: {ex}")
                return
        else:
            st.warning(f"Sample not found at `{sample_path}`. Please select custom upload.")
            return

    # Prompt user if no image selected yet
    if input_img is None:
        st.info("👆 Please upload an image or select a benchmark sample from the sidebar to begin inference.")
        # Display sample gallery preview
        st.markdown("### 🌟 Example Predictions from Test Benchmark")
        sample_cols = st.columns(3)
        sample_cols[0].metric("Benchmark Accuracy", "72.50%", "Classical CV Baseline")
        sample_cols[1].metric("Feature Descriptors", "1,764 floats", "9 Bins, 8x8 Cells")
        sample_cols[2].metric("Inference Latency", "~2.5 ms", "Fast CPU Execution")
        return

    # -------------------------------------------------------------------------
    # Feature Extraction & Inference
    # -------------------------------------------------------------------------
    extractor = HOGFeatureExtractor(hog_config)

    with st.spinner("Extracting HOG features and computing SVM decision boundary..."):
        try:
            feat_vector, hog_vis_img, gray_img = extractor.extract_with_visualization(input_img)
            pred = int(pipeline.predict(feat_vector.reshape(1, -1))[0])
            score = float(pipeline.decision_function(feat_vector.reshape(1, -1))[0])
            label_name = INV_LABEL_MAP.get(pred, "Unknown").capitalize()
        except Exception as ex:
            st.error(f"❌ Inference failed during feature extraction or classification: {ex}")
            return

    # -------------------------------------------------------------------------
    # Results Display Layout
    # -------------------------------------------------------------------------
    col_pred, col_vis = st.columns([1, 1], gap="large")

    # Column 1: Classification Results & Decision Margin
    with col_pred:
        st.subheader("🎯 Classification Decision")

        badge_class = "prediction-badge-cat" if label_name == "Cat" else "prediction-badge-dog"
        emoji = "🐱" if label_name == "Cat" else "🐶"

        st.markdown(
            f'<div class="{badge_class}">{emoji} PREDICTED: {label_name.upper()}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📐 SVM Decision Margin")

        m1, m2 = st.columns(2)
        m1.metric("Signed Margin $f(x)$", f"{score:+.4f}")
        dist = abs(score)
        m2.metric("Distance from Hyperplane $|f(x)|$", f"{dist:.4f}")

        # Visual Margin Gauge
        # Normalize score into a [-3, +3] display window
        norm_score = max(-3.0, min(3.0, score))
        slider_val = float((norm_score + 3.0) / 6.0)

        st.progress(slider_val)
        col_cat_lbl, col_center_lbl, col_dog_lbl = st.columns([1, 1, 1])
        col_cat_lbl.caption("⬅️ Cat Region ($f(x) < 0$)")
        col_center_lbl.markdown("<center><small><b>Boundary (0.0)</b></small></center>", unsafe_allow_html=True)
        col_dog_lbl.markdown("<div style='text-align: right;'><small>Dog Region ($f(x) &ge; 0$) ➡️</small></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="warning-callout">
            ⚠️ <b>Decision Margin Note</b>: The score <code>f(x)</code> is the <b>signed geometric distance</b> 
            from the separating hyperplane in standardized 1,764-dimensional feature space. 
            It is <b>not a calibrated posterior probability</b> and should not be interpreted as a percentage certainty.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="info-callout">
            <b>Image Metadata</b>:
            <br>• Source: <code>{input_name}</code>
            <br>• Original Dimensions: <code>{input_img.width} × {input_img.height} px</code>
            <br>• Normalized Model Input: <code>64 × 64 px Grayscale</code>
            <br>• HOG Feature Vector: <code>1,764 floats</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Column 2: Visualizations (Original, Normalized, HOG Gradients)
    with col_vis:
        st.subheader("🔍 Image & Feature Pipeline Visualizations")

        tab_orig, tab_norm, tab_hog = st.tabs(["1. Original Image", "2. Grayscale 64×64", "3. HOG Gradient Map"])

        with tab_orig:
            st.image(input_img, caption=f"Original Input ({input_img.width}x{input_img.height} px)", use_container_width=True)

        with tab_norm:
            st.image(gray_img, caption="Normalized 64x64 Grayscale Input (Interpolation: Bilinear)", width=280)
            st.caption("All images are converted to 1-channel grayscale and scaled to 64x64 before descriptor extraction.")

        with tab_hog:
            hog_colored = create_hog_colormap_image(hog_vis_img)
            st.image(hog_colored, caption="HOG Gradient Descriptors (Inferno Colormap)", width=280)
            st.caption("Bright vectors correspond to high-magnitude oriented edge gradients across the 49 blocks.")


if __name__ == "__main__":
    main()
