"""
Unit tests for Task 4 Interactive Streamlit App (app.py).

Focuses on:
- Preprocessing compatibility with Phase 4 feature extractor
- Exact feature dimension (2,700) matching model pipeline
- Model loading from disk and configuration schema
- Inference on valid synthetic images and decision margin calculation
- Validation of uploaded image byte streams (corrupt, valid, oversized)
- Class name and index mapping integrity
"""

import io
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from app import (
    CLASS_TO_IDX,
    GESTURE_CLASSES,
    IDX_TO_CLASS,
    extract_features_and_visualization,
    load_model_and_config,
    preprocess_image_for_hog,
    validate_uploaded_image,
)


class MockUploadedFile:
    """Mock Streamlit uploaded file object."""

    def __init__(self, data: bytes, name: str = "test.png"):
        self.data = data
        self.name = name

    def getvalue(self) -> bytes:
        return self.data


def _create_mock_image_bytes(format: str = "PNG", width: int = 640, height: int = 240) -> bytes:
    """Helper to generate valid image bytes in memory."""
    img = Image.new("L", (width, height), color=100)
    # Add a synthetic hand shape in the center
    arr = np.array(img)
    arr[60:180, 200:440] = 220
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


# -------------------------------------------------------------------------
# 1. Model Loading & Configuration Tests
# -------------------------------------------------------------------------

def test_app_load_model_and_config():
    """Verify app successfully loads trained pipeline and configuration JSON."""
    pipeline, config, err = load_model_and_config()

    assert err is None, f"Failed loading model: {err}"
    assert pipeline is not None
    assert config is not None
    assert hasattr(pipeline, "predict")
    assert "selected_model" in config
    assert config["selected_model"] == "SGD_LinearSVM"
    assert len(config["classes"]) == 10


# -------------------------------------------------------------------------
# 2. Preprocessing & HOG Compatibility Tests
# -------------------------------------------------------------------------

def test_app_preprocessing_dimensions():
    """Verify preprocessing standardizes to 160x60 grayscale maintaining 2.67:1 aspect ratio."""
    raw_img = Image.new("RGB", (640, 240), color=(50, 50, 50))
    img_arr, resized_img = preprocess_image_for_hog(raw_img, target_width=160, target_height=60)

    assert resized_img.size == (160, 60)
    assert resized_img.mode == "L"
    assert img_arr.shape == (60, 160)
    assert img_arr.dtype == np.float32
    assert 0.0 <= img_arr.min() <= img_arr.max() <= 1.0


def test_app_feature_extraction_dimension_and_vis():
    """Verify feature extractor produces exact 2,700-dim vector matching Phase 4."""
    raw_img = Image.new("L", (640, 240), color=80)
    img_arr, _ = preprocess_image_for_hog(raw_img, target_width=160, target_height=60)
    feat_vec, hog_vis = extract_features_and_visualization(img_arr)

    assert isinstance(feat_vec, np.ndarray)
    assert feat_vec.shape == (2700,)
    assert feat_vec.dtype == np.float32
    assert isinstance(hog_vis, Image.Image)


# -------------------------------------------------------------------------
# 3. Model Inference & Decision Margins Tests
# -------------------------------------------------------------------------

def test_app_inference_on_synthetic_image():
    """Verify end-to-end inference produces a valid class index and uncalibrated decision margins."""
    pipeline, _, _ = load_model_and_config()
    raw_img = Image.new("L", (640, 240), color=80)
    img_arr, _ = preprocess_image_for_hog(raw_img, target_width=160, target_height=60)
    feat_vec, _ = extract_features_and_visualization(img_arr)

    X_in = feat_vec.reshape(1, -1)
    pred_idx = int(pipeline.predict(X_in)[0])

    assert 0 <= pred_idx < 10
    assert pred_idx in IDX_TO_CLASS
    predicted_name = IDX_TO_CLASS[pred_idx]
    assert predicted_name in GESTURE_CLASSES

    # Decision function yields 10 uncalibrated margins
    margins = pipeline.decision_function(X_in)[0]
    assert len(margins) == 10
    # In multiclass linear SVM, predicted class corresponds to maximum decision margin
    assert np.argmax(margins) == pred_idx


# -------------------------------------------------------------------------
# 4. Uploaded Image Validation Tests
# -------------------------------------------------------------------------

def test_validate_uploaded_image_valid_png():
    """Verify valid PNG byte stream passes validation."""
    valid_bytes = _create_mock_image_bytes("PNG")
    mock_file = MockUploadedFile(valid_bytes, "sample.png")
    img, err = validate_uploaded_image(mock_file, max_size_mb=10)

    assert err is None
    assert img is not None
    assert img.size == (640, 240)


def test_validate_uploaded_image_valid_jpeg():
    """Verify valid JPEG byte stream passes validation."""
    valid_bytes = _create_mock_image_bytes("JPEG")
    mock_file = MockUploadedFile(valid_bytes, "sample.jpg")
    img, err = validate_uploaded_image(mock_file, max_size_mb=10)

    assert err is None
    assert img is not None


def test_validate_uploaded_image_corrupted_bytes():
    """Verify corrupted or invalid image stream is caught gracefully."""
    corrupt_bytes = b"CORRUPTED_NON_IMAGE_DATA_BYTES"
    mock_file = MockUploadedFile(corrupt_bytes, "corrupt.png")
    img, err = validate_uploaded_image(mock_file, max_size_mb=10)

    assert img is None
    assert "Corrupted or unsupported image file" in err


def test_validate_uploaded_image_oversized():
    """Verify file exceeding size limit is rejected."""
    # 2 MB file with max_size_mb set to 1 MB
    big_bytes = b"0" * (2 * 1024 * 1024)
    mock_file = MockUploadedFile(big_bytes, "huge.png")
    img, err = validate_uploaded_image(mock_file, max_size_mb=1)

    assert img is None
    assert "File exceeds maximum allowed size" in err


# -------------------------------------------------------------------------
# 5. Class Mapping Integrity Tests
# -------------------------------------------------------------------------

def test_app_class_mappings_integrity():
    """Verify 10 gesture classes match deterministic mapping 0..9."""
    assert len(GESTURE_CLASSES) == 10
    assert len(CLASS_TO_IDX) == 10
    assert len(IDX_TO_CLASS) == 10
    for idx in range(10):
        cls_name = IDX_TO_CLASS[idx]
        assert CLASS_TO_IDX[cls_name] == idx
