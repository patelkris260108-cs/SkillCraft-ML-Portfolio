"""
Unit tests for HOGFeatureExtractor (Task 4: Hand Gesture Recognition Phase 4A).

Focuses on:
- HOG feature dimension and configuration
- Deterministic extraction
- Safe handling of missing or corrupted image files
- Split extraction and label-to-feature alignment
- Atomic writes and verification
"""

import csv
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from src.feature_extractor import (
    CLASS_TO_IDX,
    GESTURE_CLASSES,
    HOGConfig,
    HOGFeatureExtractor,
)


def _create_synthetic_png(path: Path, width: int = 640, height: int = 240, color: int = 80) -> None:
    """Helper to create a test grayscale PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((height, width), color, dtype=np.uint8)
    # Add simple foreground pattern
    arr[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 200
    img = Image.fromarray(arr, mode="L")
    img.save(path, format="PNG")


# -------------------------------------------------------------------------
# 1. HOG Configuration Tests
# -------------------------------------------------------------------------

def test_hog_config_defaults():
    """Verify default HOG parameters maintain 2.67:1 aspect ratio."""
    cfg = HOGConfig()
    assert cfg.image_width == 160
    assert cfg.image_height == 60
    assert cfg.orientations == 9
    assert cfg.pixels_per_cell == (10, 10)
    assert cfg.cells_per_block == (2, 2)
    assert cfg.block_norm == "L2-Hys"
    # (6 - 2 + 1) * (16 - 2 + 1) * (2 * 2 * 9) = 5 * 15 * 36 = 2,700
    assert cfg.feature_dim == 2700


# -------------------------------------------------------------------------
# 2. Single Image Feature Extraction Tests
# -------------------------------------------------------------------------

def test_extract_single_image_shape_and_dtype(tmp_path: Path):
    """Verify extraction yields a (2700,) float32 feature vector."""
    img_path = tmp_path / "test_frame.png"
    _create_synthetic_png(img_path)

    extractor = HOGFeatureExtractor(tmp_path / "manifest.csv", tmp_path, tmp_path / "processed")
    feats = extractor.extract_image_features(img_path)

    assert isinstance(feats, np.ndarray)
    assert feats.shape == (2700,)
    assert feats.dtype == np.float32
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()


def test_extract_single_image_deterministic(tmp_path: Path):
    """Verify extraction is completely deterministic across repeated runs."""
    img_path = tmp_path / "deterministic.png"
    _create_synthetic_png(img_path)

    extractor = HOGFeatureExtractor(tmp_path / "manifest.csv", tmp_path, tmp_path / "processed")
    f1 = extractor.extract_image_features(img_path)
    f2 = extractor.extract_image_features(img_path)

    np.testing.assert_array_equal(f1, f2)


def test_extract_single_image_missing_raises(tmp_path: Path):
    """Verify missing image file raises FileNotFoundError."""
    extractor = HOGFeatureExtractor(tmp_path / "manifest.csv", tmp_path, tmp_path / "processed")
    with pytest.raises(FileNotFoundError, match="Image not found"):
        extractor.extract_image_features(tmp_path / "nonexistent.png")


def test_extract_single_image_corrupt_raises(tmp_path: Path):
    """Verify unreadable or corrupted image file raises ValueError."""
    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"NOT_A_VALID_PNG_CONTENT")

    extractor = HOGFeatureExtractor(tmp_path / "manifest.csv", tmp_path, tmp_path / "processed")
    with pytest.raises(ValueError, match="Failed to read/decode image"):
        extractor.extract_image_features(corrupt_file)


# -------------------------------------------------------------------------
# 3. Split-Wise Feature Extraction Tests
# -------------------------------------------------------------------------

def test_extract_split_mock(tmp_path: Path):
    """Verify split extraction correctly maps labels and preserves alignment."""
    data_dir = tmp_path / "data"
    manifest_path = tmp_path / "manifest.csv"
    output_dir = tmp_path / "processed"

    # Create 3 mock frames for train split
    records = []
    gestures = ["01_palm", "02_l", "03_fist"]
    for i, g in enumerate(gestures):
        rel = f"00/{g}/frame_{i}.png"
        _create_synthetic_png(data_dir / rel)
        records.append({
            "relative_path": rel,
            "filename": f"frame_{i}.png",
            "subject_id": "00",
            "gesture_label": g,
            "width": 640,
            "height": 240,
            "split": "train",
        })

    with open(manifest_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    extractor = HOGFeatureExtractor(manifest_path, data_dir, output_dir)
    X, y, meta = extractor.extract_split("train", save_to_disk=True)

    assert X.shape == (3, 2700)
    assert y.shape == (3,)
    assert y.tolist() == [CLASS_TO_IDX["01_palm"], CLASS_TO_IDX["02_l"], CLASS_TO_IDX["03_fist"]]
    assert len(meta) == 3
    assert meta[0]["filename"] == "frame_0.png"
    assert meta[1]["filename"] == "frame_1.png"
    assert meta[2]["filename"] == "frame_2.png"

    # Verify atomic disk files
    assert (output_dir / "X_train.npy").exists()
    assert (output_dir / "y_train.npy").exists()
    assert (output_dir / "train_manifest.json").exists()


def test_verify_extracted_features_missing_files(tmp_path: Path):
    """Verify verify_extracted_features returns False when processed files are absent."""
    extractor = HOGFeatureExtractor(tmp_path / "manifest.csv", tmp_path / "data", tmp_path / "empty_dir")
    ok, report = extractor.verify_extracted_features()
    assert ok is False
    assert "Config not found" in report["error"]


def test_actual_extracted_features_verification():
    """Verify real extracted features and manifest alignment if present on disk."""
    base_dir = Path(__file__).resolve().parent.parent
    manifest_path = base_dir / "reports" / "gesture_manifest.csv"
    data_dir = base_dir / "data" / "interim"
    output_dir = base_dir / "data" / "processed"

    if not (output_dir / "feature_config.json").exists():
        pytest.skip("Processed feature arrays not yet generated.")

    extractor = HOGFeatureExtractor(manifest_path, data_dir, output_dir)
    ok, report = extractor.verify_extracted_features()

    assert ok is True, f"Feature verification failed: {report}"
    assert report["actual_shapes"]["train"]["X_shape"] == [14000, 2700]
    assert report["actual_shapes"]["train"]["y_shape"] == [14000]
    assert report["actual_shapes"]["val"]["X_shape"] == [2000, 2700]
    assert report["actual_shapes"]["val"]["y_shape"] == [2000]
    assert report["actual_shapes"]["test"]["X_shape"] == [4000, 2700]
    assert report["actual_shapes"]["test"]["y_shape"] == [4000]
    assert report["zero_overlap"] is True

