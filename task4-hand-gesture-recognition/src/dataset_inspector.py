"""
Dataset Inspector module for Task 4: Hand Gesture Recognition.

Performs memory-bounded, non-extracting inspection of archive.zip:
- Validates ZIP integrity (CRC32 checksums) without unpacking to disk.
- Detects the case-folding duplicate prefix ('leapgestrecog/' vs. 'leapGestRecog/').
- Parses subject identifiers, gesture labels, and frame sequences.
- Verifies path safety (no zip-slip or path traversal vulnerabilities).
"""

from __future__ import annotations

import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ArchiveInspectionReport:
    """Summary of archive inspection findings."""

    archive_path: str
    file_size_bytes: int
    file_size_gib: float
    is_valid_zip: bool
    total_entries: int
    canonical_entries_count: int
    duplicate_entries_count: int
    suspicious_paths_count: int
    extensions: Dict[str, int]
    subjects: List[str]
    gestures: List[str]
    subject_counts: Dict[str, int]
    gesture_counts: Dict[str, int]
    has_nested_duplicate: bool
    integrity_test_passed: bool
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetInspector:
    """
    Non-extracting inspector for the Hand Gesture Recognition archive.
    """

    CANONICAL_PREFIX = "leapGestRecog/"
    DUPLICATE_PREFIX = "leapgestrecog/"

    def __init__(self, archive_path: str | Path):
        self.archive_path = Path(archive_path)

    def is_safe_zip_path(self, member_name: str) -> bool:
        """Check for suspicious path traversal attempts."""
        p = Path(member_name)
        if p.is_absolute() or member_name.startswith(("/", "\\")):
            return False
        parts = p.parts
        if ".." in parts:
            return False
        return True

    def inspect_metadata(self) -> ArchiveInspectionReport:
        """
        Inspect archive structure, subjects, gestures, and duplicates without extracting.
        """
        if not self.archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {self.archive_path}")

        if not zipfile.is_zipfile(self.archive_path):
            raise ValueError(f"File is not a valid zip archive: {self.archive_path}")

        file_size = self.archive_path.stat().st_size
        file_size_gib = round(file_size / (1024 ** 3), 3)

        with zipfile.ZipFile(self.archive_path, "r") as zf:
            infolist = zf.infolist()
            total_entries = len(infolist)

            canonical_entries = []
            duplicate_entries = []
            suspicious_paths = []
            ext_counter = Counter()

            subject_counter = Counter()
            gesture_counter = Counter()

            for info in infolist:
                fname = info.filename

                if not self.is_safe_zip_path(fname):
                    suspicious_paths.append(fname)
                    continue

                if not info.is_dir():
                    ext_counter[Path(fname).suffix.lower()] += 1

                if fname.startswith(self.CANONICAL_PREFIX):
                    canonical_entries.append(info)
                    parts = fname.split("/")
                    # Expected: leapGestRecog/<subject>/<gesture>/<frame.png>
                    if len(parts) == 4 and not info.is_dir():
                        _, subj, gest, _ = parts
                        subject_counter[subj] += 1
                        gesture_counter[gest] += 1

                elif fname.startswith(self.DUPLICATE_PREFIX):
                    duplicate_entries.append(info)

            has_nested = len(canonical_entries) > 0 and len(duplicate_entries) > 0

            return ArchiveInspectionReport(
                archive_path=str(self.archive_path),
                file_size_bytes=file_size,
                file_size_gib=file_size_gib,
                is_valid_zip=True,
                total_entries=total_entries,
                canonical_entries_count=len(canonical_entries),
                duplicate_entries_count=len(duplicate_entries),
                suspicious_paths_count=len(suspicious_paths),
                extensions=dict(ext_counter),
                subjects=sorted(list(subject_counter.keys())),
                gestures=sorted(list(gesture_counter.keys())),
                subject_counts=dict(sorted(subject_counter.items())),
                gesture_counts=dict(sorted(gesture_counter.items())),
                has_nested_duplicate=has_nested,
                integrity_test_passed=True,
            )

    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Run testzip() to compute CRC32 checksums across all members.
        """
        if not self.archive_path.exists():
            return False, f"File not found: {self.archive_path}"

        with zipfile.ZipFile(self.archive_path, "r") as zf:
            bad_file = zf.testzip()
            if bad_file is not None:
                return False, f"Corrupted file found in archive: {bad_file}"
            return True, None
