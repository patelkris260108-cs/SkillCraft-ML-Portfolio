"""
Unit tests for BenchmarkExtractor (Task 3).
"""

import io
import zipfile
from pathlib import Path
import pytest
from PIL import Image

from src.extractor import BenchmarkExtractor, ExtractionSummary


def create_dummy_jpeg(width: int = 16, height: int = 16, color: tuple = (0, 128, 255)) -> bytes:
    """Create a minimal valid JPEG image in memory."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_load_manifest_valid(tmp_path: Path):
    manifest_csv = tmp_path / "manifest.csv"
    manifest_csv.write_text(
        "filename,raw_path,label,split\n"
        "cat.0.jpg,train/cat.0.jpg,cat,train\n"
        "dog.0.jpg,train/dog.0.jpg,dog,val\n",
        encoding="utf-8",
    )
    extractor = BenchmarkExtractor("dummy.zip", manifest_csv, tmp_path / "out")
    records = extractor.load_manifest()
    assert len(records) == 2
    assert records[0]["filename"] == "cat.0.jpg"
    assert records[1]["label"] == "dog"


def test_load_manifest_missing_column(tmp_path: Path):
    manifest_csv = tmp_path / "bad_manifest.csv"
    manifest_csv.write_text("filename,label\ncat.0.jpg,cat\n", encoding="utf-8")
    extractor = BenchmarkExtractor("dummy.zip", manifest_csv, tmp_path / "out")
    with pytest.raises(ValueError, match="Manifest missing required columns"):
        extractor.load_manifest()


def test_check_destination_non_empty_raises(tmp_path: Path):
    out_dir = tmp_path / "destination"
    out_dir.mkdir()
    (out_dir / "existing.txt").write_text("already here")

    manifest_csv = tmp_path / "manifest.csv"
    manifest_csv.write_text("filename,raw_path,label,split\n", encoding="utf-8")

    extractor = BenchmarkExtractor("dummy.zip", manifest_csv, out_dir)
    with pytest.raises(FileExistsError, match="already contains 1 files"):
        extractor.check_destination(allow_existing=False)


def test_safe_path_detection(tmp_path: Path):
    out_dir = tmp_path / "interim"
    extractor = BenchmarkExtractor("dummy.zip", "manifest.csv", out_dir)

    safe_target = out_dir / "train" / "cat" / "cat.0.jpg"
    assert extractor.is_safe_path(safe_target) is True

    unsafe_target = out_dir / ".." / "evil.jpg"
    assert extractor.is_safe_path(unsafe_target) is False


def test_extract_mock_benchmark(tmp_path: Path):
    zip_path = tmp_path / "mock_train.zip"
    jpeg_bytes = create_dummy_jpeg(16, 16)

    # 10 images in zip
    with zipfile.ZipFile(zip_path, "w") as zf:
        for i in range(5):
            zf.writestr(f"train/cat.{i}.jpg", jpeg_bytes)
            zf.writestr(f"train/dog.{i}.jpg", jpeg_bytes)
        # Extra image that should NOT be extracted
        zf.writestr("train/cat.999.jpg", jpeg_bytes)

    # Manifest specifying only 6 images
    manifest_csv = tmp_path / "manifest.csv"
    manifest_csv.write_text(
        "filename,raw_path,label,split\n"
        "cat.0.jpg,train/cat.0.jpg,cat,train\n"
        "cat.1.jpg,train/cat.1.jpg,cat,val\n"
        "cat.2.jpg,train/cat.2.jpg,cat,test\n"
        "dog.0.jpg,train/dog.0.jpg,dog,train\n"
        "dog.1.jpg,train/dog.1.jpg,dog,val\n"
        "dog.2.jpg,train/dog.2.jpg,dog,test\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "interim"
    extractor = BenchmarkExtractor(zip_path, manifest_csv, out_dir)
    summary = extractor.extract(verify_after_write=True)

    assert summary.total_manifest_entries == 6
    assert summary.total_extracted == 6
    assert summary.total_verified == 6
    assert summary.failed_extractions == 0
    assert summary.corrupt_images_count == 0

    assert summary.split_counts["train"]["cat"] == 1
    assert summary.split_counts["train"]["dog"] == 1
    assert summary.split_counts["val"]["cat"] == 1
    assert summary.split_counts["val"]["dog"] == 1
    assert summary.split_counts["test"]["cat"] == 1
    assert summary.split_counts["test"]["dog"] == 1

    # Verify cat.999 was NOT extracted
    assert not (out_dir / "train" / "cat" / "cat.999.jpg").exists()

    # Verify dataset structure method
    is_valid, report = extractor.verify_dataset_structure()
    assert is_valid is True
    assert report["actual_total"] == 6
    assert report["corrupt_files_count"] == 0


def test_extract_missing_archive_entry_recorded(tmp_path: Path):
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("train/dummy.txt", "dummy")

    manifest_csv = tmp_path / "manifest.csv"
    manifest_csv.write_text(
        "filename,raw_path,label,split\n"
        "cat.0.jpg,train/cat.0.jpg,cat,train\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "interim"
    extractor = BenchmarkExtractor(zip_path, manifest_csv, out_dir)
    summary = extractor.extract()

    assert summary.failed_extractions == 1
    assert summary.total_extracted == 0
    assert "Archive member not found" in summary.errors[0]


def test_extract_corrupt_image_detected(tmp_path: Path):
    zip_path = tmp_path / "corrupt.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("train/cat.0.jpg", b"NOT_A_VALID_JPEG")

    manifest_csv = tmp_path / "manifest.csv"
    manifest_csv.write_text(
        "filename,raw_path,label,split\n"
        "cat.0.jpg,train/cat.0.jpg,cat,train\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "interim"
    extractor = BenchmarkExtractor(zip_path, manifest_csv, out_dir)
    summary = extractor.extract(verify_after_write=True)

    assert summary.total_extracted == 1
    assert summary.corrupt_images_count == 1
    assert summary.total_verified == 0
