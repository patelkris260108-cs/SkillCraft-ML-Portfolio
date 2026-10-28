"""
Dataset Extractor and Manifest Generator for Task 4: Hand Gesture Recognition.

Safely extracts the 20,000 canonical LeapGestRecog image frames from archive.zip
into data/interim/<subject>/<gesture>/<filename>, strictly excluding the redundant
duplicate 'leapgestrecog/leapGestRecog/' hierarchy.

Generates a deterministic, subject-independent CSV manifest ensuring zero subject/temporal leakage:
- Train: Subjects 00, 01, 02, 03, 04, 05, 06 (14,000 images, 70%)
- Validation: Subject 07 (2,000 images, 10%)
- Held-Out Test: Subjects 08, 09 (4,000 images, 20%)
"""

from __future__ import annotations

import csv
import os
import shutil
import time
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image, UnidentifiedImageError

TRAIN_SUBJECTS: Set[str] = {"00", "01", "02", "03", "04", "05", "06"}
VAL_SUBJECTS: Set[str] = {"07"}
TEST_SUBJECTS: Set[str] = {"08", "09"}
ALL_SUBJECTS: Set[str] = TRAIN_SUBJECTS | VAL_SUBJECTS | TEST_SUBJECTS

CANONICAL_PREFIX = "leapGestRecog/"
DUPLICATE_PREFIX = "leapgestrecog/"


def assign_split(subject_id: str) -> str:
    """Deterministically assign split based strictly on subject ID."""
    if subject_id in TRAIN_SUBJECTS:
        return "train"
    elif subject_id in VAL_SUBJECTS:
        return "val"
    elif subject_id in TEST_SUBJECTS:
        return "test"
    else:
        raise ValueError(f"Unknown subject ID '{subject_id}' not in expected subjects {ALL_SUBJECTS}")


@dataclass
class GestureManifestRecord:
    """Record describing a single benchmark gesture frame."""

    relative_path: str
    filename: str
    subject_id: str
    gesture_label: str
    width: int
    height: int
    split: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionSummary:
    """Summary of extraction and manifest generation execution."""

    total_canonical_files: int = 0
    extracted_files: int = 0
    verified_files: int = 0
    failed_extractions: int = 0
    corrupt_files_count: int = 0
    duplicate_hierarchy_skipped: int = 0
    subject_counts: Dict[str, int] = field(default_factory=dict)
    gesture_counts: Dict[str, int] = field(default_factory=dict)
    split_counts: Dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    disk_free_before_bytes: int = 0
    disk_free_after_bytes: int = 0
    errors: List[str] = field(default_factory=list)


class DatasetExtractor:
    """
    Memory-efficient, path-safe extractor and manifest generator for LeapGestRecog.
    """

    def __init__(
        self,
        archive_path: str | Path,
        output_dir: str | Path,
        manifest_path: str | Path,
    ):
        self.archive_path = Path(archive_path)
        self.output_dir = Path(output_dir)
        self.manifest_path = Path(manifest_path)

    def check_disk_space(self, required_bytes: int = 1_500_000_000) -> Tuple[int, int, int]:
        """
        Check that sufficient disk space exists before extracting.

        Returns:
            Tuple of (total_bytes, used_bytes, free_bytes)
        Raises:
            RuntimeError if free_bytes < required_bytes.
        """
        check_path = self.output_dir.anchor or "."
        total, used, free = shutil.disk_usage(check_path)
        if free < required_bytes:
            raise RuntimeError(
                f"Insufficient disk space for extraction. Required: {required_bytes / (1024**3):.2f} GB, "
                f"Available: {free / (1024**3):.2f} GB on {check_path}."
            )
        return total, used, free

    def is_safe_path(self, target_file: Path) -> bool:
        """Ensure destination path stays strictly inside output_dir (prevent zip-slip)."""
        try:
            if ".." in target_file.parts:
                return False
            target_resolved = target_file.resolve()
            output_resolved = self.output_dir.resolve()
            target_resolved.relative_to(output_resolved)
            return True
        except ValueError:
            return False

    def check_destination(self, allow_existing: bool = False) -> None:
        """Verify destination directory state to prevent accidental silent overwrites."""
        if not self.output_dir.exists():
            return

        existing_files = [f for f in self.output_dir.rglob("*") if f.is_file()]
        if existing_files and not allow_existing:
            raise FileExistsError(
                f"Destination directory '{self.output_dir}' already contains {len(existing_files)} files. "
                "Aborting extraction to prevent overwriting existing data."
            )

    def extract_canonical(
        self,
        allow_existing: bool = False,
        verify_after_write: bool = True,
        chunk_size: int = 64 * 1024,
        required_space_bytes: int = 1_500_000_000,
    ) -> Tuple[ExtractionSummary, List[GestureManifestRecord]]:
        """
        Stream and extract strictly canonical leapGestRecog/ members into data/interim/<subj>/<gest>/<filename>.
        
        Args:
            allow_existing: If True, allows writing to non-empty directory.
            verify_after_write: If True, validates image integrity with Pillow after writing.
            chunk_size: Chunk size in bytes for streaming copyfileobj.
            required_space_bytes: Minimum free disk space in bytes before extracting.
            
        Returns:
            Tuple of (ExtractionSummary, List[GestureManifestRecord])
        """
        if not self.archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {self.archive_path}")

        if not zipfile.is_zipfile(self.archive_path):
            raise ValueError(f"File is not a valid zip archive: {self.archive_path}")

        # Check free disk space before extracting
        total_b, used_b, free_before = self.check_disk_space(required_space_bytes)

        # Check destination directory
        self.check_destination(allow_existing=allow_existing)

        summary = ExtractionSummary()
        summary.disk_free_before_bytes = free_before
        records: List[GestureManifestRecord] = []
        t0 = time.time()

        with zipfile.ZipFile(self.archive_path, "r") as zf:
            infolist = zf.infolist()

            for info in infolist:
                fname = info.filename

                # Explicitly skip duplicate hierarchy and non-canonical members
                if fname.startswith(DUPLICATE_PREFIX):
                    summary.duplicate_hierarchy_skipped += 1
                    continue

                if not fname.startswith(CANONICAL_PREFIX) or info.is_dir():
                    continue

                # Expected canonical path: leapGestRecog/<subject>/<gesture>/<filename>
                parts = fname.split("/")
                if len(parts) != 4:
                    continue

                _, subject_id, gesture_label, filename = parts
                summary.total_canonical_files += 1

                # Destination: output_dir/<subject>/<gesture>/<filename>
                dest_dir = self.output_dir / subject_id / gesture_label
                dest_file = dest_dir / filename

                if not self.is_safe_path(dest_file):
                    summary.failed_extractions += 1
                    summary.errors.append(f"Unsafe destination path rejected: {dest_file}")
                    continue

                if dest_file.exists() and not allow_existing:
                    summary.failed_extractions += 1
                    summary.errors.append(f"Destination file already exists: {dest_file}")
                    raise FileExistsError(
                        f"Destination file already exists: '{dest_file}'. Extraction aborted to prevent overwriting."
                    )

                dest_dir.mkdir(parents=True, exist_ok=True)

                # Memory-bounded streaming extraction
                try:
                    with zf.open(fname, "r") as src, open(dest_file, "wb") as dst:
                        shutil.copyfileobj(src, dst, length=chunk_size)

                    summary.extracted_files += 1

                    img_w, img_h = 640, 240
                    if verify_after_write:
                        try:
                            with Image.open(dest_file) as img:
                                img_w, img_h = img.size
                                img.verify()
                            summary.verified_files += 1
                        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as err:
                            summary.corrupt_files_count += 1
                            summary.errors.append(f"Corrupt extracted image '{dest_file}': {err}")

                    # Determine subject-wise split
                    split_name = assign_split(subject_id)
                    rel_path = f"{subject_id}/{gesture_label}/{filename}"

                    records.append(
                        GestureManifestRecord(
                            relative_path=rel_path,
                            filename=filename,
                            subject_id=subject_id,
                            gesture_label=gesture_label,
                            width=img_w,
                            height=img_h,
                            split=split_name,
                        )
                    )

                    # Update aggregations
                    summary.subject_counts[subject_id] = summary.subject_counts.get(subject_id, 0) + 1
                    summary.gesture_counts[gesture_label] = summary.gesture_counts.get(gesture_label, 0) + 1
                    summary.split_counts[split_name] = summary.split_counts.get(split_name, 0) + 1

                except Exception as ex:
                    summary.failed_extractions += 1
                    summary.errors.append(f"Failed extracting '{fname}' to '{dest_file}': {ex}")

        # Post-extraction disk check
        _, _, free_after = shutil.disk_usage(self.output_dir.anchor or ".")
        summary.disk_free_after_bytes = free_after
        summary.elapsed_seconds = round(time.time() - t0, 2)
        return summary, records

    def export_manifest_csv(self, records: List[GestureManifestRecord]) -> Path:
        """
        Write deterministic manifest records to CSV file.
        """
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "relative_path",
            "filename",
            "subject_id",
            "gesture_label",
            "width",
            "height",
            "split",
        ]

        # Sort deterministically by subject, gesture, and filename
        sorted_records = sorted(
            records,
            key=lambda r: (r.subject_id, r.gesture_label, r.filename),
        )

        with open(self.manifest_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in sorted_records:
                writer.writerow(r.to_dict())

        return self.manifest_path

    def verify_dataset_structure(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify that extracted files on disk match the manifest records and strict count constraints.
        - Exactly 20,000 files in manifest and present on disk
        - Exactly 10 subjects ('00' through '09'), 2,000 images each
        - Exactly 10 gestures, 2,000 images each
        - Zero subject overlap across train, val, and test splits
        """
        if not self.manifest_path.exists():
            return False, {"error": f"Manifest not found: {self.manifest_path}"}

        records: List[Dict[str, str]] = []
        with open(self.manifest_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)

        expected_total = len(records)
        actual_total = 0
        missing_files = []
        subject_counts: Counter[str] = Counter()
        gesture_counts: Counter[str] = Counter()
        split_counts: Counter[str] = Counter()
        split_subjects: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}

        for row in records:
            file_path = self.output_dir / row["relative_path"]
            if file_path.exists() and file_path.is_file():
                actual_total += 1
                subj = row["subject_id"]
                gest = row["gesture_label"]
                splt = row["split"]
                subject_counts[subj] += 1
                gesture_counts[gest] += 1
                split_counts[splt] += 1
                if splt in split_subjects:
                    split_subjects[splt].add(subj)
            else:
                missing_files.append(str(file_path))

        # Check split subject isolation
        train_val_overlap = split_subjects["train"] & split_subjects["val"]
        train_test_overlap = split_subjects["train"] & split_subjects["test"]
        val_test_overlap = split_subjects["val"] & split_subjects["test"]
        no_split_overlap = (
            len(train_val_overlap) == 0
            and len(train_test_overlap) == 0
            and len(val_test_overlap) == 0
        )

        # Subject and gesture balance checks
        all_subjects_2000 = len(subject_counts) == 10 and all(c == 2000 for c in subject_counts.values())
        all_gestures_2000 = len(gesture_counts) == 10 and all(c == 2000 for c in gesture_counts.values())

        is_valid = (
            actual_total == expected_total
            and len(missing_files) == 0
            and actual_total == 20000
            and no_split_overlap
            and all_subjects_2000
            and all_gestures_2000
        )

        return is_valid, {
            "expected_total": expected_total,
            "actual_total": actual_total,
            "missing_files_count": len(missing_files),
            "missing_sample": missing_files[:5] if missing_files else [],
            "subject_counts": dict(sorted(subject_counts.items())),
            "gesture_counts": dict(sorted(gesture_counts.items())),
            "split_counts": dict(sorted(split_counts.items())),
            "split_subjects": {k: sorted(list(v)) for k, v in split_subjects.items()},
            "no_split_overlap": no_split_overlap,
            "all_subjects_2000": all_subjects_2000,
            "all_gestures_2000": all_gestures_2000,
        }


def main() -> int:
    """CLI runner for Task 4 canonical dataset extraction and manifest generation."""
    if Path.cwd().name == "task4-hand-gesture-recognition":
        archive_path = Path("archive.zip")
        output_dir = Path("data/interim")
        manifest_path = Path("reports/gesture_manifest.csv")
    else:
        archive_path = Path("task4-hand-gesture-recognition/archive.zip")
        output_dir = Path("task4-hand-gesture-recognition/data/interim")
        manifest_path = Path("task4-hand-gesture-recognition/reports/gesture_manifest.csv")

    print("=" * 70)
    print("SkillCraft ML Portfolio — Task 4: Canonical Dataset Extractor")
    print("=" * 70)
    print(f"Archive:     {archive_path}")
    print(f"Destination: {output_dir}")
    print(f"Manifest:    {manifest_path}\n")

    extractor = DatasetExtractor(archive_path, output_dir, manifest_path)

    # 1. Check disk space before extraction
    try:
        total, used, free = extractor.check_disk_space(required_bytes=1_500_000_000)
        print(f"Disk Space Check (Drive {extractor.output_dir.anchor or '.'}):")
        print(f"  Total: {total / (1024**3):.2f} GB | Used: {used / (1024**3):.2f} GB | Free: {free / (1024**3):.2f} GB")
        print(f"  Estimated extraction requirement: ~1.1 to 1.5 GB [OK]\n")
    except RuntimeError as err:
        print(f"[FATAL ERROR] {err}")
        return 1

    # 2. Check destination directory
    try:
        extractor.check_destination(allow_existing=False)
    except FileExistsError as err:
        print(f"[WARNING] {err}")
        print("Checking existing dataset structure...")
        is_valid, report = extractor.verify_dataset_structure()
        if is_valid:
            print(f"Dataset already fully extracted and valid ({report['actual_total']:,} images).")
            return 0
        else:
            print("[ERROR] Existing dataset structure is incomplete or corrupt.")
            return 1

    print("Extracting 20,000 canonical LeapGestRecog images (streaming with Pillow verification)...")
    summary, records = extractor.extract_canonical(allow_existing=False, verify_after_write=True)

    print(f"\nExtraction completed in {summary.elapsed_seconds:.2f} seconds!")
    print(f"  Canonical entries processed:   {summary.total_canonical_files:,}")
    print(f"  Extracted to disk:             {summary.extracted_files:,}")
    print(f"  Verified valid images:         {summary.verified_files:,}")
    print(f"  Duplicate entries skipped:     {summary.duplicate_hierarchy_skipped:,}")
    print(f"  Failed extractions:            {summary.failed_extractions}")
    print(f"  Corrupt images count:          {summary.corrupt_files_count}")

    print("\nExporting deterministic subject-wise manifest CSV...")
    extractor.export_manifest_csv(records)
    print(f"  Manifest written to: {manifest_path} ({len(records):,} records)")

    print("\nSplit Distribution:")
    for split_name, cnt in sorted(summary.split_counts.items()):
        prop = cnt / len(records) * 100
        print(f"  - {split_name.upper():5s}: {cnt:,} images ({prop:.1f}%)")

    print("\nRunning post-extraction verification audit...")
    is_valid, report = extractor.verify_dataset_structure()
    print(f"  Integrity match:       {is_valid}")
    print(f"  Actual total on disk:  {report['actual_total']:,}")
    print(f"  Missing files on disk: {report['missing_files_count']}")
    print(f"  Zero split overlap:    {report.get('no_split_overlap')}")
    print(f"  All subjects 2,000:    {report.get('all_subjects_2000')}")
    print(f"  All gestures 2,000:    {report.get('all_gestures_2000')}")

    if not is_valid:
        print("[ERROR] Post-extraction verification failed!")
        return 1

    # Report disk usage
    disk_used_by_data = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"\nDisk Storage Summary:")
    print(f"  Extracted data size:   {disk_used_by_data / (1024**2):.2f} MB ({disk_used_by_data / (1024**3):.3f} GB)")
    print(f"  Free space remaining:  {summary.disk_free_after_bytes / (1024**3):.2f} GB")

    print("\n" + "=" * 70)
    print("PHASE 2 EXTRACTION & MANIFEST GENERATION: 100% SUCCESS!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

