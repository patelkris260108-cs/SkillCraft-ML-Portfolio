"""
Unit tests for GestureEDAAnalyzer (Task 4: Hand Gesture Recognition Phase 3).

Focuses on:
- Manifest loading and schema validation
- Class and subject distribution computations
- Subject split mapping and disjointness validation
- Memory-bounded grayscale pixel statistics computation
- Figure creation (class distribution, subject distribution, heatmap, montage, pixel histogram)
- EDA summary report generation (markdown formatting, metrics, links)
- Real dataset integration verification
"""

import csv
import io
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from src.eda_analyzer import GestureEDAAnalyzer, GESTURE_DISPLAY_NAMES


def _create_mock_png(path: Path, width: int = 640, height: int = 240, mean_val: int = 50) -> None:
    """Create a dummy grayscale PNG file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((height, width), mean_val, dtype=np.uint8)
    # Add some foreground variations
    arr[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 180
    img = Image.fromarray(arr, mode="L")
    img.save(path, format="PNG")


def _create_mock_manifest(manifest_path: Path, records: list) -> Path:
    """Create a mock manifest CSV."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["relative_path", "filename", "subject_id", "gesture_label", "width", "height", "split"]
    with open(manifest_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return manifest_path


# -------------------------------------------------------------------------
# 1. Manifest Loading & Validation Tests
# -------------------------------------------------------------------------

def test_load_manifest_missing_file_raises(tmp_path: Path):
    """Verify loading non-existent manifest raises FileNotFoundError."""
    analyzer = GestureEDAAnalyzer(tmp_path / "missing.csv", tmp_path / "data", tmp_path / "reports")
    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        analyzer.load_manifest()


def test_load_manifest_invalid_columns_raises(tmp_path: Path):
    """Verify manifest with missing required columns raises ValueError."""
    bad_manifest = tmp_path / "bad_manifest.csv"
    bad_manifest.write_text("col_a,col_b\n1,2\n", encoding="utf-8")
    analyzer = GestureEDAAnalyzer(bad_manifest, tmp_path / "data", tmp_path / "reports")
    with pytest.raises(ValueError, match="missing expected columns"):
        analyzer.load_manifest()


# -------------------------------------------------------------------------
# 2. Distribution Statistics Tests
# -------------------------------------------------------------------------

def test_compute_distribution_stats_mock(tmp_path: Path):
    """Verify distribution stats on a controlled mock dataset."""
    manifest_path = tmp_path / "manifest.csv"
    records = []
    # 2 subjects, 2 gestures, 5 frames each = 20 frames total
    for subj, split in [("00", "train"), ("08", "test")]:
        for gest in ["01_palm", "02_l"]:
            for i in range(5):
                records.append({
                    "relative_path": f"{subj}/{gest}/f_{i}.png",
                    "filename": f"f_{i}.png",
                    "subject_id": subj,
                    "gesture_label": gest,
                    "width": 640,
                    "height": 240,
                    "split": split,
                })

    _create_mock_manifest(manifest_path, records)
    analyzer = GestureEDAAnalyzer(manifest_path, tmp_path / "data", tmp_path / "reports")
    stats = analyzer.compute_distribution_stats()

    assert stats["total_images"] == 20
    assert stats["num_subjects"] == 2
    assert stats["num_gestures"] == 2
    assert stats["class_counts"]["01_palm"] == 10
    assert stats["class_counts"]["02_l"] == 10
    assert stats["is_class_balanced"] is True
    assert stats["is_subject_balanced"] is True
    assert stats["zero_overlap"] is True
    assert stats["split_subject_map"]["train"] == ["00"]
    assert stats["split_subject_map"]["test"] == ["08"]


# -------------------------------------------------------------------------
# 3. Grayscale Statistics Tests
# -------------------------------------------------------------------------

def test_compute_grayscale_stats_mock(tmp_path: Path):
    """Verify grayscale pixel statistics computation on sample PNGs."""
    manifest_path = tmp_path / "manifest.csv"
    data_dir = tmp_path / "data"

    records = []
    for gest in ["01_palm", "02_l"]:
        rel = f"00/{gest}/f_0.png"
        _create_mock_png(data_dir / rel, width=640, height=240, mean_val=20)
        records.append({
            "relative_path": rel,
            "filename": "f_0.png",
            "subject_id": "00",
            "gesture_label": gest,
            "width": 640,
            "height": 240,
            "split": "train",
        })

    _create_mock_manifest(manifest_path, records)
    analyzer = GestureEDAAnalyzer(manifest_path, data_dir, tmp_path / "reports")
    stats = analyzer.compute_grayscale_stats(samples_per_class=1)

    assert stats["num_samples_analyzed"] == 2
    assert stats["image_width"] == 640
    assert stats["image_height"] == 240
    assert stats["aspect_ratio"] == 2.67
    assert stats["max_intensity"] == 180
    assert stats["mean_intensity"] > 0
    assert len(stats["histogram_accum"]) == 256


# -------------------------------------------------------------------------
# 4. Visualization & Report Generation Tests
# -------------------------------------------------------------------------

def test_plot_and_report_generation_mock(tmp_path: Path):
    """Verify plot creation and Markdown summary report writing."""
    manifest_path = tmp_path / "manifest.csv"
    data_dir = tmp_path / "data"
    reports_dir = tmp_path / "reports"

    records = []
    for s_idx, subj in enumerate(["00", "01", "07", "08"]):
        split = "train" if s_idx < 2 else ("val" if s_idx == 2 else "test")
        for g_idx in range(1, 11):
            gest = f"{g_idx:02d}_gest"
            rel = f"{subj}/{gest}/frame.png"
            _create_mock_png(data_dir / rel)
            records.append({
                "relative_path": rel,
                "filename": "frame.png",
                "subject_id": subj,
                "gesture_label": gest,
                "width": 640,
                "height": 240,
                "split": split,
            })

    _create_mock_manifest(manifest_path, records)
    analyzer = GestureEDAAnalyzer(manifest_path, data_dir, reports_dir)

    dist_stats = analyzer.compute_distribution_stats()
    gray_stats = analyzer.compute_grayscale_stats(samples_per_class=1)

    # Test plot generation
    p_class = analyzer.plot_class_distribution()
    p_subj = analyzer.plot_subject_distribution()
    p_heat = analyzer.plot_subject_gesture_heatmap()
    p_mont = analyzer.plot_gesture_montage()
    p_hist = analyzer.plot_pixel_intensity_distribution(gray_stats)

    for p in [p_class, p_subj, p_heat, p_mont, p_hist]:
        assert p.exists()
        assert p.stat().st_size > 1000  # Non-trivial image file

    # Test report generation
    report_file = analyzer.generate_eda_summary_report(dist_stats, gray_stats)
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "# Task 4 — Exploratory Data Analysis (EDA) Summary Report" in content
    assert "class_distribution.png" in content
    assert "gesture_montage.png" in content
    assert "pixel_intensity_distribution.png" in content


# -------------------------------------------------------------------------
# 5. Real Dataset Integration Test
# -------------------------------------------------------------------------

def test_real_dataset_distribution_properties():
    """Verify distribution and balance properties against real extracted manifest."""
    base_dir = Path(__file__).resolve().parent.parent
    manifest_path = base_dir / "reports" / "gesture_manifest.csv"
    data_dir = base_dir / "data" / "interim"

    if not manifest_path.exists() or not data_dir.exists():
        pytest.skip("Extracted dataset not present.")

    analyzer = GestureEDAAnalyzer(manifest_path, data_dir, base_dir / "reports")
    stats = analyzer.compute_distribution_stats()

    assert stats["total_images"] == 20000
    assert stats["num_subjects"] == 10
    assert stats["num_gestures"] == 10
    assert stats["is_class_balanced"] is True
    assert stats["is_subject_balanced"] is True
    assert stats["zero_overlap"] is True
    assert stats["split_counts"]["train"] == 14000
    assert stats["split_counts"]["val"] == 2000
    assert stats["split_counts"]["test"] == 4000
