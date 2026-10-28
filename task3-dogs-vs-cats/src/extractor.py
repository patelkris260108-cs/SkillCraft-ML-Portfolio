"""
Benchmark extraction utility for Task 3: Dogs vs Cats Classification.

Safely extracts only the 4,000 benchmark images specified in benchmark_manifest.csv
from the raw train.zip archive into organized split/class subdirectories in data/interim/.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image, UnidentifiedImageError


@dataclass
class ExtractionSummary:
    """Summary of benchmark image extraction results."""

    total_manifest_entries: int = 0
    total_extracted: int = 0
    total_verified: int = 0
    failed_extractions: int = 0
    corrupt_images_count: int = 0
    split_counts: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            "train": {"cat": 0, "dog": 0},
            "val": {"cat": 0, "dog": 0},
            "test": {"cat": 0, "dog": 0},
        }
    )
    errors: List[str] = field(default_factory=list)


class BenchmarkExtractor:
    """
    Memory-efficient, path-safe extractor for benchmark dataset partitions.
    """

    def __init__(
        self,
        archive_path: str | Path,
        manifest_path: str | Path,
        output_dir: str | Path,
    ):
        self.archive_path = Path(archive_path)
        self.manifest_path = Path(manifest_path)
        self.output_dir = Path(output_dir)

    def is_safe_path(self, target_file: Path) -> bool:
        """Ensure destination path stays strictly inside output_dir (prevent zip-slip)."""
        try:
            target_file.resolve().relative_to(self.output_dir.resolve())
            return True
        except ValueError:
            return False

    def load_manifest(self) -> List[Dict[str, str]]:
        """Load and validate manifest records from CSV."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        records: List[Dict[str, str]] = []
        with open(self.manifest_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required_cols = {"filename", "raw_path", "label", "split"}
            if not required_cols.issubset(reader.fieldnames or set()):
                raise ValueError(
                    f"Manifest missing required columns: {required_cols - set(reader.fieldnames or set())}"
                )
            for row in reader:
                records.append(row)

        return records

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

    def extract(
        self,
        allow_existing: bool = False,
        verify_after_write: bool = True,
        chunk_size: int = 64 * 1024,
    ) -> ExtractionSummary:
        """
        Extract manifest items from archive into split/class directory structure.
        
        Args:
            allow_existing: If True, permit writing into non-empty output directory.
            verify_after_write: If True, test image validity with Pillow after write.
            chunk_size: Streaming buffer size in bytes.
            
        Returns:
            ExtractionSummary object.
        """
        if not self.archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {self.archive_path}")

        if not zipfile.is_zipfile(self.archive_path):
            raise ValueError(f"File is not a valid zip archive: {self.archive_path}")

        records = self.load_manifest()
        self.check_destination(allow_existing=allow_existing)

        summary = ExtractionSummary(total_manifest_entries=len(records))

        # Create target split directories
        for split in ("train", "val", "test"):
            for label in ("cat", "dog"):
                (self.output_dir / split / label).mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(self.archive_path, "r") as zf:
            available_names = set(zf.namelist())

            for row in records:
                raw_path = row["raw_path"]
                filename = row["filename"]
                label = row["label"].lower()
                split = row["split"].lower()

                if raw_path not in available_names:
                    summary.failed_extractions += 1
                    err = f"Archive member not found: '{raw_path}'"
                    summary.errors.append(err)
                    continue

                dest_file = self.output_dir / split / label / filename
                if not self.is_safe_path(dest_file):
                    summary.failed_extractions += 1
                    err = f"Unsafe target path detected: '{dest_file}'"
                    summary.errors.append(err)
                    continue

                # Stream extraction using memory-bounded chunks
                try:
                    with zf.open(raw_path, "r") as src, open(dest_file, "wb") as dst:
                        shutil.copyfileobj(src, dst, length=chunk_size)
                    summary.total_extracted += 1
                    summary.split_counts[split][label] += 1

                    if verify_after_write:
                        try:
                            with Image.open(dest_file) as img:
                                img.verify()
                            summary.total_verified += 1
                        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as err:
                            summary.corrupt_images_count += 1
                            summary.errors.append(f"Corrupt extracted image '{dest_file}': {err}")

                except Exception as ex:
                    summary.failed_extractions += 1
                    summary.errors.append(f"Failed extracting '{raw_path}' to '{dest_file}': {ex}")

        return summary

    def verify_dataset_structure(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify that extracted dataset on disk matches manifest expectations.
        """
        records = self.load_manifest()
        expected_total = len(records)
        expected_counts: Dict[str, Dict[str, int]] = {
            "train": {"cat": 0, "dog": 0},
            "val": {"cat": 0, "dog": 0},
            "test": {"cat": 0, "dog": 0},
        }
        for r in records:
            expected_counts[r["split"]][r["label"]] += 1

        actual_counts: Dict[str, Dict[str, int]] = {
            "train": {"cat": 0, "dog": 0},
            "val": {"cat": 0, "dog": 0},
            "test": {"cat": 0, "dog": 0},
        }

        corrupt_files = []
        actual_total = 0

        for split in ("train", "val", "test"):
            for label in ("cat", "dog"):
                folder = self.output_dir / split / label
                if not folder.exists():
                    continue
                files = [p for p in folder.iterdir() if p.is_file()]
                actual_counts[split][label] = len(files)
                actual_total += len(files)

                for p in files:
                    try:
                        with Image.open(p) as img:
                            img.verify()
                    except Exception as ex:
                        corrupt_files.append((str(p), str(ex)))

        is_match = (
            actual_total == expected_total
            and actual_counts == expected_counts
            and len(corrupt_files) == 0
        )

        return is_match, {
            "expected_total": expected_total,
            "actual_total": actual_total,
            "expected_counts": expected_counts,
            "actual_counts": actual_counts,
            "corrupt_files_count": len(corrupt_files),
            "corrupt_files": corrupt_files,
        }


def main() -> int:
    """CLI runner for benchmark dataset extraction."""
    import time

    if Path.cwd().name == "task3-dogs-vs-cats":
        archive_path = Path("data/raw/train.zip")
        manifest_path = Path("reports/benchmark_manifest.csv")
        output_dir = Path("data/interim")
    else:
        archive_path = Path("task3-dogs-vs-cats/data/raw/train.zip")
        manifest_path = Path("task3-dogs-vs-cats/reports/benchmark_manifest.csv")
        output_dir = Path("task3-dogs-vs-cats/data/interim")

    print("=" * 70)
    print("SkillCraft ML Portfolio — Task 3: Benchmark Image Extractor")
    print("=" * 70)
    print(f"Archive:     {archive_path}")
    print(f"Manifest:    {manifest_path}")
    print(f"Destination: {output_dir}\n")

    if not archive_path.exists():
        print(f"[ERROR] Source archive not found: {archive_path}")
        return 1

    if not manifest_path.exists():
        print(f"[ERROR] Benchmark manifest not found: {manifest_path}")
        return 1

    start_time = time.time()
    extractor = BenchmarkExtractor(archive_path, manifest_path, output_dir)

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

    print("Extracting 4,000 benchmark images (streaming with Pillow verification)...")
    summary = extractor.extract(allow_existing=False, verify_after_write=True)
    elapsed = time.time() - start_time

    print(f"\nExtraction completed in {elapsed:.2f} seconds!")
    print(f"  Total manifest entries: {summary.total_manifest_entries:,}")
    print(f"  Total extracted:        {summary.total_extracted:,}")
    print(f"  Total verified valid:   {summary.total_verified:,}")
    print(f"  Failed extractions:     {summary.failed_extractions}")
    print(f"  Corrupt images count:   {summary.corrupt_images_count}")
    print("\nSplit counts:")
    for split_name in ("train", "val", "test"):
        counts = summary.split_counts[split_name]
        c_cat = counts["cat"]
        c_dog = counts["dog"]
        tot = c_cat + c_dog
        print(f"  - {split_name.upper():5s}: {c_cat} cats + {c_dog} dogs = {tot:,} total")

    print("\nRunning post-extraction disk structure audit...")
    is_valid, report = extractor.verify_dataset_structure()
    print(f"  Integrity match:       {is_valid}")
    print(f"  Actual total on disk:  {report['actual_total']:,}")
    print(f"  Corrupt files on disk: {report['corrupt_files_count']}")

    if not is_valid:
        print("[ERROR] Structure verification failed!")
        return 1

    print("\nBenchmark extraction and integrity verification: 100% SUCCESS!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

