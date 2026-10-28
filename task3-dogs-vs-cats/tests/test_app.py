"""
Unit tests for the Task 3 Streamlit application (app.py).
"""

import io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from app import (
    create_hog_colormap_image,
    load_model_and_config,
    validate_uploaded_image,
)
from src.feature_extractor import HOGFeatureExtractor, INV_LABEL_MAP


class MockUploadedFile:
    """Mock Streamlit UploadedFile object."""

    def __init__(self, data: bytes, name: str = "test.jpg"):
        self._data = data
        self.name = name

    def getvalue(self) -> bytes:
        return self._data


def create_in_memory_image(size: tuple = (100, 100), fmt: str = "JPEG", color: tuple = (128, 64, 32)) -> bytes:
    """Create in-memory encoded image bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_load_model_and_config():
    pipeline, config, err = load_model_and_config()
    assert err is None
    assert pipeline is not None
    assert config is not None
    assert config.feature_dim == 1764


def test_validate_uploaded_image_valid_jpeg():
    raw_bytes = create_in_memory_image(size=(120, 90), fmt="JPEG")
    mock_file = MockUploadedFile(raw_bytes, name="sample.jpg")

    img, err = validate_uploaded_image(mock_file, max_size_mb=10)
    assert err is None
    assert img is not None
    assert img.size == (120, 90)
    assert img.mode == "RGB"


def test_validate_uploaded_image_valid_png():
    raw_bytes = create_in_memory_image(size=(80, 80), fmt="PNG")
    mock_file = MockUploadedFile(raw_bytes, name="sample.png")

    img, err = validate_uploaded_image(mock_file, max_size_mb=10)
    assert err is None
    assert img is not None
    assert img.size == (80, 80)


def test_validate_uploaded_image_oversized():
    # Mock file with 11 MB of zero bytes
    raw_bytes = b"0" * (11 * 1024 * 1024)
    mock_file = MockUploadedFile(raw_bytes, name="large.jpg")

    img, err = validate_uploaded_image(mock_file, max_size_mb=10)
    assert img is None
    assert "exceeds maximum allowed size" in err


def test_validate_uploaded_image_corrupt():
    corrupt_bytes = b"CORRUPT_NON_IMAGE_DATA_BYTES"
    mock_file = MockUploadedFile(corrupt_bytes, name="broken.jpg")

    img, err = validate_uploaded_image(mock_file, max_size_mb=10)
    assert img is None
    assert "Corrupted or unsupported" in err


def test_create_hog_colormap_image():
    dummy_hog = np.random.rand(64, 64).astype(np.float32)
    rendered = create_hog_colormap_image(dummy_hog)

    assert isinstance(rendered, Image.Image)
    assert rendered.width > 0
    assert rendered.height > 0


def test_end_to_end_synthetic_inference():
    pipeline, config, err = load_model_and_config()
    assert err is None

    extractor = HOGFeatureExtractor(config)
    synthetic_img = Image.new("RGB", (128, 128), color=(100, 150, 200))

    feat, hog_vis, gray_img = extractor.extract_with_visualization(synthetic_img)
    assert feat.shape == (1764,)
    assert hog_vis.shape == (64, 64)
    assert gray_img.size == (64, 64)

    pred = int(pipeline.predict(feat.reshape(1, -1))[0])
    score = float(pipeline.decision_function(feat.reshape(1, -1))[0])

    assert pred in (0, 1)
    assert isinstance(score, float)
    assert pred in INV_LABEL_MAP


def test_end_to_end_real_benchmark_samples():
    pipeline, config, err = load_model_and_config()
    assert err is None

    extractor = HOGFeatureExtractor(config)
    test_cat_path = Path("data/interim/test/cat/cat.10074.jpg")
    test_dog_path = Path("data/interim/test/dog/dog.10017.jpg")

    if test_cat_path.exists():
        cat_img = Image.open(test_cat_path).convert("RGB")
        feat, _, _ = extractor.extract_with_visualization(cat_img)
        pred = int(pipeline.predict(feat.reshape(1, -1))[0])
        score = float(pipeline.decision_function(feat.reshape(1, -1))[0])
        assert pred == 0  # Cat
        assert score < 0.0

    if test_dog_path.exists():
        dog_img = Image.open(test_dog_path).convert("RGB")
        feat, _, _ = extractor.extract_with_visualization(dog_img)
        pred = int(pipeline.predict(feat.reshape(1, -1))[0])
        score = float(pipeline.decision_function(feat.reshape(1, -1))[0])
        assert pred == 1  # Dog
        assert score >= 0.0
