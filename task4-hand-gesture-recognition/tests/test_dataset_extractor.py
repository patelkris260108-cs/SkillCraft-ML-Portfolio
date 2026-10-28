"""
Unit tests for DatasetExtractor and Manifest Generation (Task 4: Hand Gesture Recognition).

Focuses on:
- Deterministic subject-wise split assignment and isolation (zero subject leakage)
- Path traversal and zip-slip safety checks
- Free disk space validation
- No-overwrite destination protection
- Exclusion of duplicate 'leapgestrecog/' hierarchy
- Deterministic CSV manifest formatting and sorting
- Corrupt image verification
"""

import csv
import io
import zipfile
from pathlib import Path
import pytest
from PIL import Image

from src.dataset_extractor import (
    ALL_SUBJECTS,
    CANONICAL_PREFIX,
    DUPLICATE_PREFIX,
    TEST_SUBJECTS,
    TRAIN_SUBJECTS,
    VAL_SUBJECTS,
    DatasetExtractor,
    GestureManifestRecord,
    assign_split,
)


def _create_test_png_bytes(width: int = 640, height: int = 240) -> bytes:
    """Helper to create valid in-memory grayscale PNG image bytes."""
    img = Image.new("L", (width, height), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# -------------------------------------------------------------------------
# 1. Subject-Wise Split Tests
# -------------------------------------------------------------------------

def test_assign_split_train_val_test():
    """Verify strictly partitioned splits by subject ID."""
    for s in ["00", "01", "02", "03", "04", "05", "06"]:
        assert assign_split(s) == "train"

    assert assign_split("07") == "val"

    for s in ["08", "09"]:
        assert assign_split(s) == "test"


def test_assign_split_unknown_subject_raises():
    """Verify unknown or out-of-range subject IDs raise ValueError."""
    for invalid in ["10", "99", "subject_01", "-1", ""]:
        with pytest.raises(ValueError, match="Unknown subject ID"):
            assign_split(invalid)


def test_split_sets_pairwise_disjoint():
    """Verify zero subject overlap across splits (leak-free evaluation)."""
    assert len(TRAIN_SUBJECTS & VAL_SUBJECTS) == 0
    assert len(TRAIN_SUBJECTS & TEST_SUBJECTS) == 0
    assert len(VAL_SUBJECTS & TEST_SUBJECTS) == 0
    assert (TRAIN_SUBJECTS | VAL_SUBJECTS | TEST_SUBJECTS) == ALL_SUBJECTS
    assert len(ALL_SUBJECTS) == 10


# -------------------------------------------------------------------------
# 2. Path Safety & Disk Space Tests
# -------------------------------------------------------------------------

def test_is_safe_path_valid(tmp_path: Path):
    """Verify safe paths inside output directory are accepted."""
    extractor = DatasetExtractor("archive.zip", tmp_path / "interim", "manifest.csv")
    safe_target = tmp_path / "interim" / "00" / "01_palm" / "frame_0001.png"
    assert extractor.is_safe_path(safe_target) is True


def test_is_safe_path_zip_slip_rejected(tmp_path: Path):
    """Verify directory traversal tokens and external paths are rejected."""
    out_dir = tmp_path / "interim"
    extractor = DatasetExtractor("archive.zip", out_dir, "manifest.csv")

    # Path traversal with ..
    traversal_path = out_dir / ".." / "escaped.png"
    assert extractor.is_safe_path(traversal_path) is False

    # External path completely outside output_dir
    outside_path = tmp_path / "other_dir" / "file.png"
    assert extractor.is_safe_path(outside_path) is False


def test_check_disk_space_sufficient(tmp_path: Path):
    """Verify check_disk_space succeeds when free space is available."""
    extractor = DatasetExtractor("archive.zip", tmp_path, "manifest.csv")
    total, used, free = extractor.check_disk_space(required_bytes=1024)
    assert total > 0
    assert free > 0


def test_check_disk_space_insufficient_raises(tmp_path: Path):
    """Verify check_disk_space raises RuntimeError when required space is unrealistically huge."""
    extractor = DatasetExtractor("archive.zip", tmp_path, "manifest.csv")
    impossible_bytes = 10_000 * 1024 * 1024 * 1024 * 1024  # 10,000 TB
    with pytest.raises(RuntimeError, match="Insufficient disk space"):
        extractor.check_disk_space(required_bytes=impossible_bytes)


# -------------------------------------------------------------------------
# 3. Destination Pre-Flight & Overwrite Protection Tests
# -------------------------------------------------------------------------

def test_check_destination_empty(tmp_path: Path):
    """Verify empty or non-existent destination directory does not raise."""
    extractor = DatasetExtractor("archive.zip", tmp_path / "new_dir", "manifest.csv")
    extractor.check_destination(allow_existing=False)


def test_check_destination_non_empty_raises(tmp_path: Path):
    """Verify non-empty destination directory raises FileExistsError to prevent overwrite."""
    out_dir = tmp_path / "interim"
    out_dir.mkdir(parents=True)
    (out_dir / "existing_file.txt").write_text("existing data", encoding="utf-8")

    extractor = DatasetExtractor("archive.zip", out_dir, "manifest.csv")
    with pytest.raises(FileExistsError, match="already contains 1 files"):
        extractor.check_destination(allow_existing=False)


def test_destination_no_overwrite_during_extraction(tmp_path: Path):
    """Verify individual file overwrite is rejected during extraction."""
    mock_zip = tmp_path / "mock.zip"
    png_bytes = _create_test_png_bytes()
    with zipfile.ZipFile(mock_zip, "w") as zf:
        zf.writestr("leapGestRecog/00/01_palm/frame_0001.png", png_bytes)

    out_dir = tmp_path / "interim"
    target_file = out_dir / "00" / "01_palm" / "frame_0001.png"
    target_file.parent.mkdir(parents=True)
    target_file.write_bytes(b"existing_old_data")

    extractor = DatasetExtractor(mock_zip, out_dir, tmp_path / "manifest.csv")
    with pytest.raises(FileExistsError, match="Destination directory '.*' already contains 1 files"):
        extractor.extract_canonical(allow_existing=False)


# -------------------------------------------------------------------------
# 4. Safe Extraction & Duplicate Hierarchy Exclusion Tests
# -------------------------------------------------------------------------

def test_extract_canonical_excludes_duplicate_hierarchy(tmp_path: Path):
    """
    Critical Test: Verify that files under 'leapgestrecog/' are skipped,
    and only 'leapGestRecog/' files are extracted.
    """
    mock_zip = tmp_path / "mock_archive.zip"
    png_data_1 = _create_test_png_bytes(640, 240)
    png_data_2 = _create_test_png_bytes(640, 240)
    png_data_3 = _create_test_png_bytes(640, 240)

    with zipfile.ZipFile(mock_zip, "w") as zf:
        # Canonical members (Subject 00 = train, Subject 07 = val, Subject 08 = test)
        zf.writestr("leapGestRecog/00/01_palm/frame_00_01_0001.png", png_data_1)
        zf.writestr("leapGestRecog/07/02_l/frame_07_02_0001.png", png_data_2)
        zf.writestr("leapGestRecog/08/03_fist/frame_08_03_0001.png", png_data_3)

        # Duplicate members under duplicate hierarchy (MUST BE SKIPPED)
        zf.writestr("leapgestrecog/leapGestRecog/00/01_palm/frame_00_01_0001.png", png_data_1)
        zf.writestr("leapgestrecog/leapGestRecog/07/02_l/frame_07_02_0001.png", png_data_2)

        # Non-image / irrelevant entry (MUST BE SKIPPED)
        zf.writestr("README.txt", b"Notes")

    out_dir = tmp_path / "interim"
    manifest_csv = tmp_path / "reports" / "manifest.csv"
    extractor = DatasetExtractor(mock_zip, out_dir, manifest_csv)

    summary, records = extractor.extract_canonical(allow_existing=False, verify_after_write=True)

    # 1. Summary checks
    assert summary.total_canonical_files == 3
    assert summary.extracted_files == 3
    assert summary.verified_files == 3
    assert summary.duplicate_hierarchy_skipped == 2
    assert summary.failed_extractions == 0
    assert summary.corrupt_files_count == 0

    # 2. File system checks (Only canonical files extracted)
    f1 = out_dir / "00" / "01_palm" / "frame_00_01_0001.png"
    f2 = out_dir / "07" / "02_l" / "frame_07_02_0001.png"
    f3 = out_dir / "08" / "03_fist" / "frame_08_03_0001.png"

    assert f1.exists() and f1.is_file()
    assert f2.exists() and f2.is_file()
    assert f3.exists() and f3.is_file()

    # Ensure duplicate folder was NEVER created on disk
    dup_folder = out_dir / "leapgestrecog"
    assert not dup_folder.exists()

    # 3. Records check
    assert len(records) == 3
    splits = {r.subject_id: r.split for r in records}
    assert splits["00"] == "train"
    assert splits["07"] == "val"
    assert splits["08"] == "test"


# -------------------------------------------------------------------------
# 5. Manifest CSV Formatting & Determinism Tests
# -------------------------------------------------------------------------

def test_export_manifest_csv_format_and_sort(tmp_path: Path):
    """Verify manifest CSV contains expected headers, data types, and deterministic sort."""
    manifest_csv = tmp_path / "manifest.csv"
    extractor = DatasetExtractor("archive.zip", tmp_path / "data", manifest_csv)

    records = [
        GestureManifestRecord("08/01_palm/f2.png", "f2.png", "08", "01_palm", 640, 240, "test"),
        GestureManifestRecord("00/02_l/f1.png", "f1.png", "00", "02_l", 640, 240, "train"),
        GestureManifestRecord("00/01_palm/f3.png", "f3.png", "00", "01_palm", 640, 240, "train"),
    ]

    extractor.export_manifest_csv(records)
    assert manifest_csv.exists()

    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 3
    # Check headers
    assert list(reader[0].keys()) == [
        "relative_path",
        "filename",
        "subject_id",
        "gesture_label",
        "width",
        "height",
        "split",
    ]

    # Deterministic sort order: (subject_id, gesture_label, filename)
    assert reader[0]["subject_id"] == "00" and reader[0]["gesture_label"] == "01_palm"
    assert reader[1]["subject_id"] == "00" and reader[1]["gesture_label"] == "02_l"
    assert reader[2]["subject_id"] == "08" and reader[2]["gesture_label"] == "01_palm"


# -------------------------------------------------------------------------
# 6. Corrupt Image Detection Test
# -------------------------------------------------------------------------

def test_corrupt_image_detection(tmp_path: Path):
    """Verify corrupt images are flagged by Pillow verification during extraction."""
    mock_zip = tmp_path / "corrupt_mock.zip"
    with zipfile.ZipFile(mock_zip, "w") as zf:
        zf.writestr("leapGestRecog/00/01_palm/bad_frame.png", b"NOT_A_VALID_PNG_CONTENT")

    out_dir = tmp_path / "interim"
    extractor = DatasetExtractor(mock_zip, out_dir, tmp_path / "manifest.csv")

    summary, records = extractor.extract_canonical(allow_existing=False, verify_after_write=True)
    assert summary.extracted_files == 1
    assert summary.verified_files == 0
    assert summary.corrupt_files_count == 1
    assert len(summary.errors) == 1


# -------------------------------------------------------------------------
# 7. Structure Verification Tests
# -------------------------------------------------------------------------

def test_verify_dataset_structure_missing_manifest(tmp_path: Path):
    """Verify verify_dataset_structure handles missing manifest gracefully."""
    extractor = DatasetExtractor("archive.zip", tmp_path / "data", tmp_path / "missing.csv")
    is_valid, report = extractor.verify_dataset_structure()
    assert is_valid is False
    assert "Manifest not found" in report["error"]


def test_actual_manifest_and_structure_validation():
    """Verify the real extracted dataset and generated manifest if present on disk."""
    base_dir = Path(__file__).resolve().parent.parent
    manifest_path = base_dir / "reports" / "gesture_manifest.csv"
    data_dir = base_dir / "data" / "interim"

    if not manifest_path.exists() or not data_dir.exists():
        pytest.skip("Extracted dataset or manifest not yet generated.")

    extractor = DatasetExtractor(base_dir / "archive.zip", data_dir, manifest_path)
    is_valid, report = extractor.verify_dataset_structure()

    assert is_valid is True, f"Structure verification failed: {report}"
    assert report["actual_total"] == 20000
    assert report["expected_total"] == 20000
    assert report["missing_files_count"] == 0
    assert report["no_split_overlap"] is True
    assert report["all_subjects_2000"] is True
    assert report["all_gestures_2000"] is True

    # Check split breakdown
    assert report["split_counts"]["train"] == 14000
    assert report["split_counts"]["val"] == 2000
    assert report["split_counts"]["test"] == 4000

    # Check subject assignment
    assert report["split_subjects"]["train"] == ["00", "01", "02", "03", "04", "05", "06"]
    assert report["split_subjects"]["val"] == ["07"]
    assert report["split_subjects"]["test"] == ["08", "09"]

