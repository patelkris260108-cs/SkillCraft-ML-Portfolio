"""
Unit tests for the BenchmarkEDA module (Task 3).
"""

import io
from pathlib import Path
import pytest
from PIL import Image

from src.eda import BenchmarkEDA, EDAMetrics


def create_test_image(path: Path, size: tuple = (64, 64), color: tuple = (255, 0, 0), fmt: str = "JPEG") -> None:
    """Helper to write a small test image to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path, format=fmt)


def test_scan_dataset_dummy_images(tmp_path: Path):
    data_dir = tmp_path / "interim"

    # 1 landscape cat (100x50), 1 portrait dog (50x100), 1 square cat (60x60)
    create_test_image(data_dir / "train" / "cat" / "cat.0.jpg", (100, 50))
    create_test_image(data_dir / "train" / "dog" / "dog.0.jpg", (50, 100))
    create_test_image(data_dir / "val" / "cat" / "cat.1.jpg", (60, 60))

    eda = BenchmarkEDA(data_dir=data_dir, figures_dir=tmp_path / "figures")
    df, corrupt_files = eda.scan_dataset()

    assert len(df) == 3
    assert len(corrupt_files) == 0

    cat_0 = df[df["filename"] == "cat.0.jpg"].iloc[0]
    assert cat_0["width"] == 100
    assert cat_0["height"] == 50
    assert cat_0["aspect_ratio"] == 2.0
    assert cat_0["orientation"] == "landscape"
    assert cat_0["mode"] == "RGB"

    dog_0 = df[df["filename"] == "dog.0.jpg"].iloc[0]
    assert dog_0["width"] == 50
    assert dog_0["height"] == 100
    assert dog_0["aspect_ratio"] == 0.5
    assert dog_0["orientation"] == "portrait"

    cat_1 = df[df["filename"] == "cat.1.jpg"].iloc[0]
    assert cat_1["orientation"] == "square"


def test_eda_analyze_metrics(tmp_path: Path):
    data_dir = tmp_path / "interim"

    create_test_image(data_dir / "train" / "cat" / "cat.0.jpg", (100, 100))
    create_test_image(data_dir / "train" / "dog" / "dog.0.jpg", (120, 80))
    create_test_image(data_dir / "test" / "dog" / "dog.1.jpg", (80, 120))

    eda = BenchmarkEDA(data_dir=data_dir, figures_dir=tmp_path / "figures")
    df, metrics = eda.analyze()

    assert metrics.total_images == 3
    assert metrics.split_counts["train"]["cat"] == 1
    assert metrics.split_counts["train"]["dog"] == 1
    assert metrics.split_counts["test"]["dog"] == 1
    assert metrics.split_counts["val"]["cat"] == 0

    assert "RGB" in metrics.color_modes
    assert metrics.color_modes["RGB"] == 3

    assert "cat" in metrics.stats_by_class
    assert "dog" in metrics.stats_by_class
    assert metrics.stats_by_class["cat"]["width"]["median"] == 100.0


def test_corrupt_image_handling(tmp_path: Path):
    data_dir = tmp_path / "interim"
    corrupt_file = data_dir / "train" / "cat" / "corrupt.jpg"
    corrupt_file.parent.mkdir(parents=True, exist_ok=True)
    corrupt_file.write_bytes(b"INVALID_HEADER_DATA")

    eda = BenchmarkEDA(data_dir=data_dir, figures_dir=tmp_path / "figures")
    df, corrupt_files = eda.scan_dataset()

    assert len(df) == 0
    assert len(corrupt_files) == 1
    assert "corrupt.jpg" in corrupt_files[0]


def test_generate_figures_and_report(tmp_path: Path):
    data_dir = tmp_path / "interim"
    figures_dir = tmp_path / "figures"
    report_file = tmp_path / "eda_report.md"

    # Create 8 cats and 8 dogs to test montage and distributions
    for i in range(8):
        create_test_image(data_dir / "train" / "cat" / f"cat.{i}.jpg", (50 + i * 5, 60))
        create_test_image(data_dir / "train" / "dog" / f"dog.{i}.jpg", (60, 50 + i * 5))

    eda = BenchmarkEDA(data_dir=data_dir, figures_dir=figures_dir)
    df, metrics = eda.analyze()

    figures = eda.generate_figures(df)
    assert len(figures) == 3
    for f in figures:
        assert f.exists()
        assert f.stat().st_size > 0

    report = eda.generate_markdown_report(metrics, output_file=report_file)
    assert report_file.exists()
    assert "Total Benchmark Images" in report
    assert "Resolution Statistics" in report
    assert "HOG + SVM" in report
