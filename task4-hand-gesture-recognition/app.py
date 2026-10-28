"""
Interactive Streamlit Web Application for Task 4: Hand Gesture Recognition.

Uses the trained HOG + SVM pipeline (StandardScaler + SGD Linear SVM) to classify
near-infrared hand gesture frames, visualizing the 160x60 grayscale representation
and HOG orientation gradient maps.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from skimage.feature import hog

# Ensure task directory is in sys.path
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from src.feature_extractor import (
    CLASS_TO_IDX,
    GESTURE_CLASSES,
    HOGConfig,
    IDX_TO_CLASS,
)

GESTURE_DESCRIPTIONS: Dict[str, str] = {
    "01_palm": "Flat open palm facing sensor",
    "02_l": "L-Shape posture (thumb & index extended)",
    "03_fist": "Closed static fist",
    "04_fist_moved": "Dynamic moving fist",
    "05_thumb": "Thumbs-up gesture",
    "06_index": "Pointing index finger",
    "07_ok": "Circular OK sign (thumb & index touching)",
    "08_palm_moved": "Dynamic moving open palm",
    "09_c": "Curved C-shape hand",
    "10_down": "Downward pointing gesture",
}

GESTURE_ICONS: Dict[str, str] = {
    "01_palm": "✋",
    "02_l": "👆",
    "03_fist": "✊",
    "04_fist_moved": "👊",
    "05_thumb": "👍",
    "06_index": "👉",
    "07_ok": "👌",
    "08_palm_moved": "👋",
    "09_c": "🤌",
    "10_down": "👇",
}

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hand Gesture Recognition AI | SkillCraft ML",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .prediction-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .prediction-badge {
        background: linear-gradient(135deg, #0d9488, #0284c7);
        color: #ffffff !important;
        padding: 14px 28px;
        border-radius: 10px;
        font-size: 1.6rem;
        font-weight: 700;
        text-align: center;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 14px rgba(13, 148, 136, 0.4);
    }
    .info-callout {
        background-color: rgba(14, 165, 233, 0.08);
        border-left: 4px solid #0ea5e9;
        padding: 14px 18px;
        border-radius: 4px 8px 8px 4px;
        margin-top: 15px;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    .warning-callout {
        background-color: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 4px 8px 8px 4px;
        margin-top: 15px;
        font-size: 0.90rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Cached Resource Loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained HOG + SGD SVM model...")
def load_model_and_config() -> Tuple[Optional[Any], Optional[Dict[str, Any]], Optional[str]]:
    """Safely load serialized pipeline and model configuration."""
    model_path = _current_dir / "models" / "best_gesture_model.joblib"
    config_path = _current_dir / "models" / "model_config.json"

    if not model_path.exists():
        return None, None, f"Model artifact not found at: {model_path}"

    if not config_path.exists():
        return None, None, f"Model configuration not found at: {config_path}"

    try:
        pipeline = joblib.load(model_path)
        with open(config_path, mode="r", encoding="utf-8") as f:
            config_data = json.load(f)
        return pipeline, config_data, None
    except Exception as ex:
        return None, None, f"Error loading model: {ex}"


# -----------------------------------------------------------------------------
# Preprocessing & Inference Helpers
# -----------------------------------------------------------------------------
def validate_uploaded_image(
    uploaded_file,
    max_size_mb: int = 10,
) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Validate uploaded image size and integrity using Pillow."""
    file_bytes = uploaded_file.getvalue()

    if len(file_bytes) > max_size_mb * 1024 * 1024:
        return None, f"File exceeds maximum allowed size ({max_size_mb} MB). Upload a smaller image."

    try:
        img_check = Image.open(io.BytesIO(file_bytes))
        img_check.verify()

        # Reopen for display and processing
        img = Image.open(io.BytesIO(file_bytes))
        return img, None
    except (UnidentifiedImageError, OSError, SyntaxError) as ex:
        return None, f"Corrupted or unsupported image file: {ex}"


def preprocess_image_for_hog(
    image: Image.Image,
    target_width: int = 160,
    target_height: int = 60,
) -> Tuple[np.ndarray, Image.Image]:
    """
    Standardize image preserving 2.67:1 aspect ratio:
    1. Convert to single channel grayscale ('L' mode)
    2. Resize to (target_width, target_height) using bilinear interpolation
    3. Normalize pixel intensities to [0.0, 1.0]
    """
    gray_img = image.convert("L")
    resized_img = gray_img.resize((target_width, target_height), Image.Resampling.BILINEAR)
    img_arr = np.array(resized_img, dtype=np.float32) / 255.0
    return img_arr, resized_img


def extract_features_and_visualization(
    img_arr: np.ndarray,
    config: Optional[HOGConfig] = None,
) -> Tuple[np.ndarray, Image.Image]:
    """
    Compute HOG features and 2D gradient visualization image using scikit-image.
    """
    cfg = config or HOGConfig()
    feat_vec, hog_image = hog(
        img_arr,
        orientations=cfg.orientations,
        pixels_per_cell=cfg.pixels_per_cell,
        cells_per_block=cfg.cells_per_block,
        block_norm=cfg.block_norm,
        visualize=True,
        feature_vector=True,
    )

    # Render HOG visualization map
    fig, ax = plt.subplots(figsize=(4, 1.5), dpi=100)
    ax.imshow(hog_image, cmap="inferno")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    hog_vis_img = Image.open(buf)

    return feat_vec.astype(np.float32), hog_vis_img


# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
def main() -> None:
    # Header Banner
    st.title("🖐️ Hand Gesture Recognition AI Classifier")
    st.markdown(
        "**Near-Infrared (NIR) Computer Vision Pipeline** featuring **Histogram of Oriented Gradients (HOG)** "
        "and a **Subject-Independent Linear SVM** (`SGDClassifier`)."
    )
    st.divider()

    # Load Model Pipeline
    pipeline, model_config, err = load_model_and_config()

    # -------------------------------------------------------------------------
    # Sidebar: Architecture & Dataset Specs
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Model Pipeline Specs")

        if err:
            st.error(f"⚠️ {err}")
            st.stop()
        else:
            st.success("✅ Model Pipeline Loaded & Active")

        st.markdown(
            """
            - **Classifier**: `StandardScaler + SGD Linear SVM`
            - **Loss Function**: `hinge` (Maximum Margin Separator)
            - **Input Grid**: $160 \\times 60$ Grayscale (Aspect Ratio 2.67:1)
            - **HOG Orientations**: 9 Gradient Bins ($0^\\circ$–$180^\\circ$)
            - **Cell Geometry**: $10 \\times 10$ Pixels / Cell
            - **Block Normalization**: $2 \\times 2$ Cells (`L2-Hys`)
            - **Feature Dimension**: **2,700 floats** / frame
            """
        )

        st.subheader("📊 Generalization Benchmark")
        st.markdown(
            """
            - **Validation Macro F1** (Subj 07): **79.75%**
            - **Held-Out Test Accuracy** (Subjs 08, 09): **62.20%**
            - **Held-Out Test Macro F1**: **61.25%**
            - **Baseline Guessing**: **10.00%**
            """
        )

        st.subheader("📂 10 Gesture Classes")
        for g, desc in GESTURE_DESCRIPTIONS.items():
            icon = GESTURE_ICONS.get(g, "✋")
            st.markdown(f"**{icon} `{g}`**: {desc}")

    # -------------------------------------------------------------------------
    # Main Content Area
    # -------------------------------------------------------------------------
    tab_upload, tab_sample, tab_about = st.tabs(["📤 Upload Image", "🖼️ Benchmark Sample Gallery", "ℹ️ About & Limitations"])

    # Sample images available from test set
    sample_images = {
        "Palm (01_palm)": _current_dir / "data" / "interim" / "08" / "01_palm" / "frame_08_01_0050.png",
        "L-Shape (02_l)": _current_dir / "data" / "interim" / "08" / "02_l" / "frame_08_02_0050.png",
        "Fist (03_fist)": _current_dir / "data" / "interim" / "08" / "03_fist" / "frame_08_03_0050.png",
        "Thumb Up (05_thumb)": _current_dir / "data" / "interim" / "08" / "05_thumb" / "frame_08_05_0050.png",
        "Index Point (06_index)": _current_dir / "data" / "interim" / "08" / "06_index" / "frame_08_06_0050.png",
        "OK Sign (07_ok)": _current_dir / "data" / "interim" / "08" / "07_ok" / "frame_08_07_0050.png",
        "C-Shape (09_c)": _current_dir / "data" / "interim" / "08" / "09_c" / "frame_08_09_0050.png",
        "Down Point (10_down)": _current_dir / "data" / "interim" / "08" / "10_down" / "frame_08_10_0050.png",
    }

    target_image: Optional[Image.Image] = None
    image_source_label = ""

    with tab_upload:
        st.subheader("Upload a Hand Gesture Frame")
        st.write("Upload a near-infrared frame (or any hand gesture photo in PNG, JPG, or JPEG format, up to 10 MB).")

        uploaded_file = st.file_uploader(
            "Select an image file:",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            val_img, val_err = validate_uploaded_image(uploaded_file, max_size_mb=10)
            if val_err:
                st.error(f"❌ {val_err}")
            else:
                target_image = val_img
                image_source_label = f"Uploaded File: {uploaded_file.name}"

    with tab_sample:
        st.subheader("Select a Benchmark Test Frame")
        st.write("Select a real near-infrared frame from the held-out test participant (Subject 08):")

        selected_sample_key = st.selectbox(
            "Choose a sample test frame:",
            options=list(sample_images.keys()),
            index=0,
        )

        sample_path = sample_images[selected_sample_key]
        if sample_path.exists():
            if st.button("Load This Sample Frame", use_container_width=True):
                with Image.open(sample_path) as s_img:
                    target_image = s_img.copy()
                    image_source_label = f"Benchmark Sample: {selected_sample_key}"
        else:
            st.info("Sample frame not found on disk. Use the upload tab.")

    # -------------------------------------------------------------------------
    # Inference & Diagnostics Display
    # -------------------------------------------------------------------------
    if target_image is not None:
        st.divider()
        st.markdown(f"### 🔍 Analysis: *{image_source_label}*")

        col1, col2, col3 = st.columns([1, 1, 1.2])

        # 1. Preprocess
        img_arr, resized_img = preprocess_image_for_hog(target_image, target_width=160, target_height=60)
        feat_vec, hog_vis = extract_features_and_visualization(img_arr)

        with col1:
            st.markdown("**Original Image Frame**")
            st.image(target_image, use_container_width=True)
            st.caption(f"Original Size: {target_image.size[0]} × {target_image.size[1]} px")

        with col2:
            st.markdown("**Resized ($160 \\times 60$) & HOG Map**")
            st.image(resized_img, caption="160 × 60 Grayscale Input", use_container_width=True)
            st.image(hog_vis, caption="HOG Gradient Orientations (Inferno)", use_container_width=True)

        with col3:
            # Run model prediction
            X_input = feat_vec.reshape(1, -1)
            pred_idx = int(pipeline.predict(X_input)[0])
            pred_class = IDX_TO_CLASS.get(pred_idx, f"Class {pred_idx}")
            pred_icon = GESTURE_ICONS.get(pred_class, "✋")
            pred_desc = GESTURE_DESCRIPTIONS.get(pred_class, "")

            st.markdown(
                f"""
                <div class="prediction-card">
                    <p style="margin: 0; font-size: 0.95rem; color: #94a3b8; font-weight: 500;">PREDICTED GESTURE CLASS</p>
                    <div class="prediction-badge" style="margin-top: 10px;">
                        {pred_icon} {pred_class}
                    </div>
                    <p style="margin-top: 12px; margin-bottom: 0; font-size: 1rem; color: #cbd5e1; text-align: center;">
                        <em>{pred_desc}</em>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Raw Decision Margins (Hinge Loss)
            if hasattr(pipeline, "decision_function"):
                decision_margins = pipeline.decision_function(X_input)[0]

                st.markdown("**Decision Margins by Gesture Class**")
                margin_fig, ax_m = plt.subplots(figsize=(6, 3.5), dpi=100)
                y_positions = np.arange(len(GESTURE_CLASSES))

                colors = ["#0ea5e9" if i == pred_idx else "#475569" for i in range(len(GESTURE_CLASSES))]
                bars = ax_m.barh(y_positions, decision_margins, color=colors, height=0.65)
                ax_m.set_yticks(y_positions)
                ax_m.set_yticklabels([f"{GESTURE_ICONS.get(g, '')} {g}" for g in GESTURE_CLASSES], fontsize=8)
                ax_m.axvline(0, color="#ffffff", linestyle="--", alpha=0.3, linewidth=1)
                ax_m.set_xlabel("SVM Decision Margin (Signed Distance)", fontsize=8)
                ax_m.grid(axis="x", linestyle="--", alpha=0.3)
                margin_fig.tight_layout()

                st.pyplot(margin_fig)
                plt.close(margin_fig)

        # Callout explaining uncalibrated decision scores
        st.markdown(
            """
            <div class="info-callout">
                <strong>ℹ️ Note on Decision Margins:</strong> The active model is a linear Support Vector Machine trained with hinge loss (<code>loss='hinge'</code>).
                The scores shown above are <strong>uncalibrated decision margins</strong> (signed geometric distances from the class hyperplanes in 2,700-dimensional HOG feature space),
                not probabilistic likelihoods. Higher positive margins indicate stronger directional confidence toward that gesture class.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Callout explaining static vs dynamic gesture limitations
        st.markdown(
            """
            <div class="warning-callout">
                <strong>⚠️ Static 2D Limitation on Dynamic Gestures:</strong> The LeapGestRecog dataset contains paired static and dynamic motions
                (e.g., <code>01_palm</code> vs <code>08_palm_moved</code>, and <code>03_fist</code> vs <code>04_fist_moved</code>).
                Because this model evaluates individual static 2D frames without temporal motion sequence tracking (such as optical flow or RNNs),
                hand contour features for these paired variants are virtually identical and may naturally be cross-confused.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_about:
        st.subheader("About the Machine Learning System")
        st.markdown(
            """
            ### Dataset Overview
            - **Name:** LeapGestRecog (GTI-UPM Dataset)
            - **Sensor:** Leap Motion Near-Infrared (NIR) stereo optical sensor
            - **Modality:** 8-Bit single-channel Grayscale
            - **Total Canonical Frames:** 20,000 images evenly balanced across 10 participants and 10 gestures (2,000 frames/class).

            ### Subject-Independent Partitioning
            - **Training Set (70%):** Subjects `00, 01, 02, 03, 04, 05, 06` (14,000 frames)
            - **Validation Set (10%):** Subject `07` (2,000 frames)
            - **Held-Out Test Set (20%):** Subjects `08, 09` (4,000 frames)
            - *Frames from the same subject are never shared across splits, ensuring genuine subject-level generalization.*

            ### Performance Summary
            - **Validation Macro F1 (Subject 07):** **79.75%**
            - **Held-Out Test Macro F1 (Subjects 08, 09):** **61.25%**
            - **Held-Out Test Accuracy:** **62.20%**
            - **Baseline Guessing Accuracy:** **10.00%**
            """
        )


if __name__ == "__main__":
    main()
