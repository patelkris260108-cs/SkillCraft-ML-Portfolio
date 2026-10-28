"""
Unit tests for DatasetInspector (Task 4: Hand Gesture Recognition).
"""

import io
import zipfile
from pathlib import Path
import pytest

from src.dataset_inspector import DatasetInspector


def test_is_safe_zip_path():
    inspector = DatasetInspector("dummy.zip")
    assert inspector.is_safe_zip_path("leapGestRecog/00/01_palm/frame_0001.png") is True
    assert inspector.is_safe_zip_path("sub/folder/file.txt") is True

    # Suspicious paths
    assert inspector.is_safe_zip_path("../evil.png") is False
    assert inspector.is_safe_zip_path("leapGestRecog/../../etc/passwd") is False
    assert inspector.is_safe_zip_path("/absolute/path.png") is False
    assert inspector.is_safe_zip_path("\\windows\\path.png") is False


def test_missing_archive_raises(tmp_path: Path):
    inspector = DatasetInspector(tmp_path / "nonexistent.zip")
    with pytest.raises(FileNotFoundError):
        inspector.inspect_metadata()


def test_invalid_zipfile_raises(tmp_path: Path):
    fake_zip = tmp_path / "fake.zip"
    fake_zip.write_text("NOT A ZIP", encoding="utf-8")
    inspector = DatasetInspector(fake_zip)
    with pytest.raises(ValueError, match="not a valid zip"):
        inspector.inspect_metadata()


def test_inspect_metadata_mock(tmp_path: Path):
    mock_zip = tmp_path / "mock_gestures.zip"
    with zipfile.ZipFile(mock_zip, "w") as zf:
        # Canonical entries for subject 00 (gestures 01_palm and 02_l)
        zf.writestr("leapGestRecog/00/01_palm/frame_00_01_0001.png", b"PNG_DATA_1")
        zf.writestr("leapGestRecog/00/01_palm/frame_00_01_0002.png", b"PNG_DATA_2")
        zf.writestr("leapGestRecog/00/02_l/frame_00_02_0001.png", b"PNG_DATA_3")
        # Canonical entries for subject 01
        zf.writestr("leapGestRecog/01/01_palm/frame_01_01_0001.png", b"PNG_DATA_4")

        # Duplicate entries under leapgestrecog/
        zf.writestr("leapgestrecog/leapGestRecog/00/01_palm/frame_00_01_0001.png", b"PNG_DATA_1")
        zf.writestr("leapgestrecog/leapGestRecog/00/01_palm/frame_00_01_0002.png", b"PNG_DATA_2")

    inspector = DatasetInspector(mock_zip)
    report = inspector.inspect_metadata()

    assert report.is_valid_zip is True
    assert report.total_entries == 6
    assert report.canonical_entries_count == 4
    assert report.duplicate_entries_count == 2
    assert report.has_nested_duplicate is True
    assert report.suspicious_paths_count == 0

    assert report.subjects == ["00", "01"]
    assert report.gestures == ["01_palm", "02_l"]
    assert report.subject_counts["00"] == 3
    assert report.subject_counts["01"] == 1
    assert report.gesture_counts["01_palm"] == 3
    assert report.gesture_counts["02_l"] == 1


def test_verify_integrity_mock(tmp_path: Path):
    mock_zip = tmp_path / "valid.zip"
    with zipfile.ZipFile(mock_zip, "w") as zf:
        zf.writestr("test.txt", b"hello world")

    inspector = DatasetInspector(mock_zip)
    ok, err = inspector.verify_integrity()
    assert ok is True
    assert err is None
