"""
Unit tests for DatasetValidator and benchmark split planner (Task 3).
"""

import io
import os
import zipfile
from pathlib import Path
import pytest
from PIL import Image

from src.dataset_validator import (
    DatasetValidator,
    ImageEntry,
    export_benchmark_manifest_csv,
    generate_benchmark_manifest,
    generate_validation_markdown_report,
    plan_benchmark_splits,
)


def create_dummy_jpeg(width: int = 16, height: int = 16, color: tuple = (255, 0, 0)) -> bytes:
    """Create a minimal valid JPEG image in memory."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_parse_member_name_valid():
    validator = DatasetValidator("dummy.zip")

    label, sample_id, ext, is_valid = validator.parse_member_name("cat.0.jpg")
    assert is_valid is True
    assert label == "cat"
    assert sample_id == 0
    assert ext == "jpg"

    label, sample_id, ext, is_valid = validator.parse_member_name("train/dog.12499.jpeg")
    assert is_valid is True
    assert label == "dog"
    assert sample_id == 12499
    assert ext == "jpeg"

    label, sample_id, ext, is_valid = validator.parse_member_name("cat.50.PNG")
    assert is_valid is True
    assert label == "cat"
    assert sample_id == 50
    assert ext == "png"


def test_parse_member_name_invalid():
    validator = DatasetValidator("dummy.zip")

    invalid_names = [
        "random.jpg",
        "cat.jpg",
        "dog_0.jpg",
        "sampleSubmission.csv",
        "cat.abc.jpg",
        "notes.txt",
        ".DS_Store",
    ]
    for name in invalid_names:
        label, sample_id, ext, is_valid = validator.parse_member_name(name)
        assert is_valid is False, f"Expected {name} to be invalid"
        assert label is None


def test_suspicious_path_detection():
    validator = DatasetValidator("dummy.zip")

    assert validator.is_suspicious_path("../evil.jpg") is True
    assert validator.is_suspicious_path("folder/../../evil.jpg") is True
    assert validator.is_suspicious_path("/etc/passwd") is True
    assert validator.is_suspicious_path("C:\\windows\\test.jpg") is True

    assert validator.is_suspicious_path("train/cat.0.jpg") is False
    assert validator.is_suspicious_path("cat.0.jpg") is False


def test_verify_image_stream_valid():
    validator = DatasetValidator("dummy.zip")
    data = create_dummy_jpeg(24, 32)
    stream = io.BytesIO(data)

    ok, err, w, h, fmt = validator.verify_image_stream(stream)
    assert ok is True
    assert err is None
    assert w == 24
    assert h == 32
    assert fmt == "JPEG"


def test_verify_image_stream_corrupt():
    validator = DatasetValidator("dummy.zip")
    corrupt_data = b"NOT_A_VALID_JPEG_HEADER_RANDOM_GARBAGE"
    stream = io.BytesIO(corrupt_data)

    ok, err, w, h, fmt = validator.verify_image_stream(stream)
    assert ok is False
    assert err is not None
    assert w is None


def test_validate_zip_archive(tmp_path: Path):
    zip_path = tmp_path / "test_archive.zip"
    jpeg_bytes = create_dummy_jpeg(20, 20)
    corrupt_bytes = b"CORRUPTED_BYTES"

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("train/cat.0.jpg", jpeg_bytes)
        zf.writestr("train/cat.1.jpg", jpeg_bytes)
        zf.writestr("train/dog.0.jpg", jpeg_bytes)
        zf.writestr("train/dog.1.jpg", corrupt_bytes)
        zf.writestr("train/unexpected.txt", b"readme")
        zf.writestr("train/cat.0.jpg", jpeg_bytes)  # duplicate entry

    validator = DatasetValidator(zip_path)
    summary, entries = validator.validate(verify_images=True)

    assert summary.source_type == "zip"
    assert summary.total_entries_enumerated == 6
    assert summary.candidate_cats_count == 3  # cat.0, cat.1, duplicate cat.0
    assert summary.candidate_dogs_count == 2  # dog.0, dog.1
    assert summary.unexpected_files_count == 1  # unexpected.txt
    assert summary.duplicate_names_count == 1  # duplicate cat.0.jpg
    assert summary.corrupt_images_count == 1  # dog.1.jpg corrupt
    assert summary.valid_verified_images == 4


def test_validate_directory(tmp_path: Path):
    data_dir = tmp_path / "images"
    data_dir.mkdir()
    jpeg_bytes = create_dummy_jpeg(16, 16)

    (data_dir / "cat.0.jpg").write_bytes(jpeg_bytes)
    (data_dir / "cat.1.jpg").write_bytes(jpeg_bytes)
    (data_dir / "dog.0.jpg").write_bytes(jpeg_bytes)
    (data_dir / "other.csv").write_text("a,b,c")

    validator = DatasetValidator(data_dir)
    summary, entries = validator.validate(verify_images=True)

    assert summary.source_type == "directory"
    assert summary.candidate_cats_count == 2
    assert summary.candidate_dogs_count == 1
    assert summary.unexpected_files_count == 1
    assert summary.corrupt_images_count == 0


def test_missing_path_raises_file_not_found():
    validator = DatasetValidator("nonexistent_path_404.zip")
    with pytest.raises(FileNotFoundError):
        validator.validate()


def test_unsupported_format_raises_value_error(tmp_path: Path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Hello world")

    validator = DatasetValidator(txt_file)
    with pytest.raises(ValueError):
        validator.validate()


def test_plan_benchmark_splits_valid():
    # Construct mock valid entries
    entries = []
    for i in range(2500):
        entries.append(
            ImageEntry(
                raw_path=f"cat.{i}.jpg",
                filename=f"cat.{i}.jpg",
                label="cat",
                sample_id=i,
                extension="jpg",
                is_valid_naming=True,
                is_corrupt=False,
            )
        )
        entries.append(
            ImageEntry(
                raw_path=f"dog.{i}.jpg",
                filename=f"dog.{i}.jpg",
                label="dog",
                sample_id=i,
                extension="jpg",
                is_valid_naming=True,
                is_corrupt=False,
            )
        )

    result = plan_benchmark_splits(
        entries,
        target_per_class=2000,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )

    assert result["total_selected"] == 4000
    counts = result["split_counts"]

    # 70% of 2000 = 1400 per class -> 2800 total
    assert counts["train"]["total"] == 2800
    assert counts["train"]["cats"] == 1400
    assert counts["train"]["dogs"] == 1400

    # 15% of 2000 = 300 per class -> 600 total
    assert counts["val"]["total"] == 600
    assert counts["val"]["cats"] == 300
    assert counts["val"]["dogs"] == 300

    # 15% of 2000 = 300 per class -> 600 total
    assert counts["test"]["total"] == 600
    assert counts["test"]["cats"] == 300
    assert counts["test"]["dogs"] == 300

    # Verify manifest and mutual exclusivity
    manifest = result["manifest"]
    assert len(manifest) == 4000

    train_names = {row["filename"] for row in manifest if row["split"] == "train"}
    val_names = {row["filename"] for row in manifest if row["split"] == "val"}
    test_names = {row["filename"] for row in manifest if row["split"] == "test"}

    assert len(train_names) == 2800
    assert len(val_names) == 600
    assert len(test_names) == 600
    assert len(train_names & val_names) == 0
    assert len(train_names & test_names) == 0
    assert len(val_names & test_names) == 0


def test_plan_benchmark_splits_insufficient_samples_raises_value_error():
    entries = [
        ImageEntry(
            raw_path="cat.0.jpg",
            filename="cat.0.jpg",
            label="cat",
            sample_id=0,
            extension="jpg",
            is_valid_naming=True,
            is_corrupt=False,
        )
    ]
    with pytest.raises(ValueError, match="Insufficient valid images"):
        plan_benchmark_splits(entries, target_per_class=2000)


def test_plan_benchmark_splits_invalid_ratios():
    entries = []
    with pytest.raises(ValueError, match="must sum to 1.0"):
        plan_benchmark_splits(entries, train_ratio=0.5, val_ratio=0.1, test_ratio=0.1)


def test_generate_validation_markdown_report(tmp_path: Path):
    zip_path = tmp_path / "test.zip"
    jpeg_bytes = create_dummy_jpeg(10, 10)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("cat.0.jpg", jpeg_bytes)
        zf.writestr("dog.0.jpg", jpeg_bytes)

    validator = DatasetValidator(zip_path)
    summary, entries = validator.validate(verify_images=True)
    report = generate_validation_markdown_report(summary)

    assert "# Task 3: Dogs vs. Cats — Dataset Validation Audit Report" in report
    assert "Candidate Cat Images" in report
    assert "Candidate Dog Images" in report
    assert "Benchmark Feasibility Assessment" in report


def test_export_benchmark_manifest_csv(tmp_path: Path):
    manifest = [
        {"filename": "cat.0.jpg", "raw_path": "train/cat.0.jpg", "label": "cat", "split": "train"},
        {"filename": "dog.0.jpg", "raw_path": "train/dog.0.jpg", "label": "dog", "split": "val"},
        {"filename": "cat.1.jpg", "raw_path": "train/cat.1.jpg", "label": "cat", "split": "test"},
    ]
    out_csv = tmp_path / "manifest.csv"
    res_path = export_benchmark_manifest_csv(manifest, out_csv)
    assert res_path.exists()

    import csv
    with open(res_path, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 3
    assert reader[0]["filename"] == "cat.0.jpg"
    assert reader[0]["raw_path"] == "train/cat.0.jpg"
    assert reader[0]["label"] == "cat"
    assert reader[0]["split"] == "train"


def test_plan_benchmark_splits_determinism_seed_42():
    entries = []
    for i in range(2100):
        entries.append(ImageEntry(f"cat.{i}.jpg", f"cat.{i}.jpg", "cat", i, "jpg", True, False))
        entries.append(ImageEntry(f"dog.{i}.jpg", f"dog.{i}.jpg", "dog", i, "jpg", True, False))

    run1 = plan_benchmark_splits(entries, target_per_class=2000, random_seed=42)
    run2 = plan_benchmark_splits(entries, target_per_class=2000, random_seed=42)

    assert run1["manifest"] == run2["manifest"]
    assert [row["filename"] for row in run1["manifest"]] == [row["filename"] for row in run2["manifest"]]


def test_plan_benchmark_splits_different_seeds():
    entries = []
    for i in range(2100):
        entries.append(ImageEntry(f"cat.{i}.jpg", f"cat.{i}.jpg", "cat", i, "jpg", True, False))
        entries.append(ImageEntry(f"dog.{i}.jpg", f"dog.{i}.jpg", "dog", i, "jpg", True, False))

    run_seed42 = plan_benchmark_splits(entries, target_per_class=2000, random_seed=42)
    run_seed99 = plan_benchmark_splits(entries, target_per_class=2000, random_seed=99)

    filenames_42 = {row["filename"] for row in run_seed42["manifest"]}
    filenames_99 = {row["filename"] for row in run_seed99["manifest"]}
    # Since pool is 4200 and we sample 4000, there will be non-overlap in sampled subsets
    assert filenames_42 != filenames_99


def test_plan_benchmark_splits_exact_counts_and_balance():
    entries = []
    for i in range(2200):
        entries.append(ImageEntry(f"cat.{i}.jpg", f"cat.{i}.jpg", "cat", i, "jpg", True, False))
        entries.append(ImageEntry(f"dog.{i}.jpg", f"dog.{i}.jpg", "dog", i, "jpg", True, False))

    plan = plan_benchmark_splits(entries, target_per_class=2000, random_seed=42)

    assert plan["total_selected"] == 4000
    manifest = plan["manifest"]
    assert len(manifest) == 4000

    cats_in_manifest = [r for r in manifest if r["label"] == "cat"]
    dogs_in_manifest = [r for r in manifest if r["label"] == "dog"]
    assert len(cats_in_manifest) == 2000
    assert len(dogs_in_manifest) == 2000

    train_cats = [r for r in cats_in_manifest if r["split"] == "train"]
    train_dogs = [r for r in dogs_in_manifest if r["split"] == "train"]
    val_cats = [r for r in cats_in_manifest if r["split"] == "val"]
    val_dogs = [r for r in dogs_in_manifest if r["split"] == "val"]
    test_cats = [r for r in cats_in_manifest if r["split"] == "test"]
    test_dogs = [r for r in dogs_in_manifest if r["split"] == "test"]

    assert len(train_cats) == 1400
    assert len(train_dogs) == 1400
    assert len(val_cats) == 300
    assert len(val_dogs) == 300
    assert len(test_cats) == 300
    assert len(test_dogs) == 300


def test_generate_benchmark_manifest_from_mock_zip(tmp_path: Path):
    mock_zip = tmp_path / "mock_train.zip"
    jpeg_bytes = create_dummy_jpeg(8, 8)
    with zipfile.ZipFile(mock_zip, "w") as zf:
        for i in range(50):
            zf.writestr(f"train/cat.{i}.jpg", jpeg_bytes)
            zf.writestr(f"train/dog.{i}.jpg", jpeg_bytes)

    out_csv = tmp_path / "reports" / "benchmark_manifest.csv"
    plan = generate_benchmark_manifest(mock_zip, out_csv, target_per_class=40, random_seed=42)

    assert out_csv.exists()
    assert plan["total_selected"] == 80
    assert plan["split_counts"]["train"]["total"] == 56  # 70% of 80 = 56 (28 cats, 28 dogs)
    assert plan["split_counts"]["val"]["total"] == 12    # 15% of 80 = 12 (6 cats, 6 dogs)
    assert plan["split_counts"]["test"]["total"] == 12   # 15% of 80 = 12 (6 cats, 6 dogs)

