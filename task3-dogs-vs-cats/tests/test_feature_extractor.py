"""
Unit tests for the HOGFeatureExtractor module (Task 3).
"""

import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from src.feature_extractor import (
    HOGConfig,
    HOGFeatureExtractor,
    LABEL_MAP,
    load_partition,
)


def create_dummy_image_file(path: Path, size: tuple = (64, 64), color: tuple = (200, 100, 50)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path, format="JPEG")


def test_hog_config_dimension():
    config = HOGConfig()
    # Programmatic feature length check
    assert config.feature_dim == 1764

    # Serialization test
    d = config.to_dict()
    assert d["feature_dim"] == 1764
    assert d["target_size"] == [64, 64]

    restored = HOGConfig.from_dict(d)
    assert restored.feature_dim == 1764
    assert restored.orientations == 9


def test_extract_from_image_shape_and_dtype():
    extractor = HOGFeatureExtractor()
    img = Image.new("RGB", (100, 80), color=(128, 64, 32))
    feat = extractor.extract_from_image(img)

    assert isinstance(feat, np.ndarray)
    assert feat.dtype == np.float32
    assert feat.shape == (1764,)
    assert np.all(feat >= 0.0)


def test_extract_from_numpy_array():
    extractor = HOGFeatureExtractor()
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    feat = extractor.extract_from_image(arr)

    assert feat.shape == (1764,)


def test_extract_deterministic():
    extractor = HOGFeatureExtractor()
    img = Image.new("RGB", (64, 64), color=(50, 150, 250))
    feat1 = extractor.extract_from_image(img)
    feat2 = extractor.extract_from_image(img)

    np.testing.assert_array_equal(feat1, feat2)


def test_extract_with_visualization():
    extractor = HOGFeatureExtractor()
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    feat, hog_img, pil_gray = extractor.extract_with_visualization(img)

    assert feat.shape == (1764,)
    assert hog_img.shape == (64, 64)
    assert pil_gray.size == (64, 64)
    assert pil_gray.mode == "L"


def test_extract_from_path_missing_raises(tmp_path: Path):
    extractor = HOGFeatureExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.extract_from_path(tmp_path / "nonexistent.jpg")


def test_extract_dataset_mock(tmp_path: Path):
    data_dir = tmp_path / "interim"
    manifest_csv = tmp_path / "manifest.csv"

    # Create dummy images: 2 train (1 cat, 1 dog), 1 val (1 cat)
    create_dummy_image_file(data_dir / "train" / "cat" / "cat.1.jpg", (80, 80))
    create_dummy_image_file(data_dir / "train" / "dog" / "dog.1.jpg", (90, 70))
    create_dummy_image_file(data_dir / "val" / "cat" / "cat.2.jpg", (64, 64))

    manifest_csv.write_text(
        "filename,raw_path,label,split\n"
        "cat.1.jpg,train/cat.1.jpg,cat,train\n"
        "dog.1.jpg,train/dog.1.jpg,dog,train\n"
        "cat.2.jpg,train/cat.2.jpg,cat,val\n",
        encoding="utf-8",
    )

    extractor = HOGFeatureExtractor()
    X_train, y_train, fn_train = extractor.extract_dataset(manifest_csv, data_dir, split="train")

    assert X_train.shape == (2, 1764)
    assert y_train.tolist() == [LABEL_MAP["cat"], LABEL_MAP["dog"]]
    assert fn_train == ["cat.1.jpg", "dog.1.jpg"]

    # Val split
    X_val, y_val, fn_val = extractor.extract_dataset(manifest_csv, data_dir, split="val")
    assert X_val.shape == (1, 1764)
    assert y_val.tolist() == [LABEL_MAP["cat"]]
    assert fn_val == ["cat.2.jpg"]


def test_save_and_load_partition(tmp_path: Path):
    data_dir = tmp_path / "interim"
    manifest_csv = tmp_path / "manifest.csv"
    processed_dir = tmp_path / "processed"

    create_dummy_image_file(data_dir / "train" / "cat" / "cat.1.jpg")
    create_dummy_image_file(data_dir / "train" / "dog" / "dog.1.jpg")
    create_dummy_image_file(data_dir / "val" / "cat" / "cat.2.jpg")
    create_dummy_image_file(data_dir / "val" / "dog" / "dog.2.jpg")
    create_dummy_image_file(data_dir / "test" / "cat" / "cat.3.jpg")
    create_dummy_image_file(data_dir / "test" / "dog" / "dog.3.jpg")

    manifest_csv.write_text(
        "filename,raw_path,label,split\n"
        "cat.1.jpg,train/cat.1.jpg,cat,train\n"
        "dog.1.jpg,train/dog.1.jpg,dog,train\n"
        "cat.2.jpg,train/cat.2.jpg,cat,val\n"
        "dog.2.jpg,train/dog.2.jpg,dog,val\n"
        "cat.3.jpg,train/cat.3.jpg,cat,test\n"
        "dog.3.jpg,train/dog.3.jpg,dog,test\n",
        encoding="utf-8",
    )

    extractor = HOGFeatureExtractor()
    info = extractor.extract_and_save_partitions(manifest_csv, data_dir, processed_dir)

    assert info["train"] == (2, 1764)
    assert info["val"] == (2, 1764)
    assert info["test"] == (2, 1764)

    X_loaded, y_loaded, fn_loaded = load_partition(processed_dir, "train")
    assert X_loaded.shape == (2, 1764)
    assert y_loaded.shape == (2,)
    assert len(fn_loaded) == 2
