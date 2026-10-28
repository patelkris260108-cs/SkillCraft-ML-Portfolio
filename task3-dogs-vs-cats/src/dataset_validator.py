"""
Dataset validation and benchmark planning utility for Task 3: Dogs vs Cats Classification.

Provides memory-conscious inspection, archive validation, naming convention auditing,
and reproducible 4,000-image benchmark split generation.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import zipfile

from PIL import Image, UnidentifiedImageError


# Canonical Kaggle Dogs vs Cats naming pattern: cat.0.jpg or dog.12499.jpg
FILENAME_PATTERN = re.compile(
    r"^(?:.*[/\\])?(cat|dog)\.(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE
)


@dataclass
class ImageEntry:
    """Represents an enumerated candidate image entry."""

    raw_path: str
    filename: str
    label: Optional[str]  # 'cat', 'dog', or None
    sample_id: Optional[int]
    extension: str
    is_valid_naming: bool
    is_verified: bool = False
    is_corrupt: bool = False
    error_message: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None


@dataclass
class ValidationSummary:
    """Aggregated dataset validation statistics."""

    source_path: str
    source_type: str  # 'zip' or 'directory'
    total_entries_enumerated: int = 0
    total_candidate_images: int = 0
    candidate_cats_count: int = 0
    candidate_dogs_count: int = 0
    unexpected_files_count: int = 0
    duplicate_names_count: int = 0
    suspicious_paths_count: int = 0
    total_verified_images: int = 0
    valid_verified_images: int = 0
    corrupt_images_count: int = 0
    unexpected_files: List[str] = field(default_factory=list)
    duplicate_names: List[str] = field(default_factory=list)
    suspicious_paths: List[str] = field(default_factory=list)
    corrupt_files: List[Dict[str, str]] = field(default_factory=list)


class DatasetValidator:
    """
    Memory-bounded dataset validator for the Dogs vs Cats image collection.
    
    Inspects ZIP archives without extracting to disk and validates individual
    JPEG images via Pillow header decoding without loading pixel rasters into memory.
    """

    def __init__(self, source_path: str | Path):
        self.source_path = Path(source_path)

    def parse_member_name(self, member_name: str) -> Tuple[Optional[str], Optional[int], str, bool]:
        """
        Parse and validate filename structure against canonical cat.<id>.jpg / dog.<id>.jpg format.
        
        Returns:
            Tuple of (label, sample_id, extension, is_valid)
        """
        basename = os.path.basename(member_name)
        match = FILENAME_PATTERN.match(basename)
        if match:
            label = match.group(1).lower()
            sample_id = int(match.group(2))
            ext = match.group(3).lower()
            return label, sample_id, ext, True
        
        _, ext = os.path.splitext(basename)
        return None, None, ext.lstrip(".").lower(), False

    def is_suspicious_path(self, path_str: str) -> bool:
        """Detect directory traversal sequences or dangerous root paths."""
        parts = path_str.replace("\\", "/").split("/")
        if ".." in parts or path_str.startswith("/") or re.match(r"^[a-zA-Z]:", path_str):
            return True
        return False

    def verify_image_stream(self, stream: io.BytesIO) -> Tuple[bool, Optional[str], Optional[int], Optional[int], Optional[str]]:
        """
        Verify an image using Pillow without decoding the full pixel array into RAM.
        
        Returns:
            Tuple of (is_valid, error_msg, width, height, format)
        """
        try:
            with Image.open(stream) as img:
                img.verify()  # Verifies structural integrity without pixel decoding
                width, height = img.size
                fmt = img.format
                return True, None, width, height, fmt
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as err:
            return False, str(err), None, None, None

    def validate(
        self,
        verify_images: bool = True,
        max_verify: Optional[int] = None,
    ) -> Tuple[ValidationSummary, List[ImageEntry]]:
        """
        Execute comprehensive dataset validation.
        
        Args:
            verify_images: Whether to decode headers of images to ensure no corruption.
            max_verify: Optional cap on verified images (useful for quick audit passes).
            
        Returns:
            Tuple of (ValidationSummary, List of ImageEntry objects)
        """
        if not self.source_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.source_path}")

        is_zip = zipfile.is_zipfile(self.source_path) if self.source_path.is_file() else False
        is_dir = self.source_path.is_dir()

        if not (is_zip or is_dir):
            raise ValueError(
                f"Unsupported source format: {self.source_path}. Must be a .zip file or directory."
            )

        summary = ValidationSummary(
            source_path=str(self.source_path),
            source_type="zip" if is_zip else "directory",
        )
        entries: List[ImageEntry] = []
        seen_names: Set[str] = set()

        if is_zip:
            self._validate_zip(summary, entries, seen_names, verify_images, max_verify)
        else:
            self._validate_dir(summary, entries, seen_names, verify_images, max_verify)

        return summary, entries

    def _validate_zip(
        self,
        summary: ValidationSummary,
        entries: List[ImageEntry],
        seen_names: Set[str],
        verify_images: bool,
        max_verify: Optional[int],
    ) -> None:
        with zipfile.ZipFile(self.source_path, "r") as zf:
            infolist = zf.infolist()
            summary.total_entries_enumerated = len(infolist)
            verify_count = 0

            for info in infolist:
                # Skip directories and macOS metadata
                if info.is_dir() or "__MACOSX" in info.filename or info.filename.endswith("/"):
                    continue

                if self.is_suspicious_path(info.filename):
                    summary.suspicious_paths_count += 1
                    summary.suspicious_paths.append(info.filename)
                    continue

                basename = os.path.basename(info.filename)
                if basename in seen_names:
                    summary.duplicate_names_count += 1
                    summary.duplicate_names.append(info.filename)
                seen_names.add(basename)

                label, sample_id, ext, is_valid_name = self.parse_member_name(info.filename)
                if not is_valid_name:
                    summary.unexpected_files_count += 1
                    summary.unexpected_files.append(info.filename)
                    continue

                summary.total_candidate_images += 1
                if label == "cat":
                    summary.candidate_cats_count += 1
                elif label == "dog":
                    summary.candidate_dogs_count += 1

                entry = ImageEntry(
                    raw_path=info.filename,
                    filename=basename,
                    label=label,
                    sample_id=sample_id,
                    extension=ext,
                    is_valid_naming=True,
                )

                # Memory-bounded image verification
                if verify_images and (max_verify is None or verify_count < max_verify):
                    verify_count += 1
                    summary.total_verified_images += 1
                    with zf.open(info) as stream:
                        data_bytes = io.BytesIO(stream.read())
                        ok, err_msg, w, h, fmt = self.verify_image_stream(data_bytes)
                        entry.is_verified = True
                        if ok:
                            summary.valid_verified_images += 1
                            entry.width = w
                            entry.height = h
                            entry.format = fmt
                        else:
                            summary.corrupt_images_count += 1
                            entry.is_corrupt = True
                            entry.error_message = err_msg
                            summary.corrupt_files.append({"path": info.filename, "error": str(err_msg)})

                entries.append(entry)

    def _validate_dir(
        self,
        summary: ValidationSummary,
        entries: List[ImageEntry],
        seen_names: Set[str],
        verify_images: bool,
        max_verify: Optional[int],
    ) -> None:
        all_files = [p for p in self.source_path.rglob("*") if p.is_file()]
        summary.total_entries_enumerated = len(all_files)
        verify_count = 0

        for file_path in all_files:
            rel_str = str(file_path.relative_to(self.source_path)).replace("\\", "/")
            if "__MACOSX" in rel_str or ".DS_Store" in rel_str:
                continue

            basename = file_path.name
            if basename in seen_names:
                summary.duplicate_names_count += 1
                summary.duplicate_names.append(rel_str)
            seen_names.add(basename)

            label, sample_id, ext, is_valid_name = self.parse_member_name(rel_str)
            if not is_valid_name:
                summary.unexpected_files_count += 1
                summary.unexpected_files.append(rel_str)
                continue

            summary.total_candidate_images += 1
            if label == "cat":
                summary.candidate_cats_count += 1
            elif label == "dog":
                summary.candidate_dogs_count += 1

            entry = ImageEntry(
                raw_path=rel_str,
                filename=basename,
                label=label,
                sample_id=sample_id,
                extension=ext,
                is_valid_naming=True,
            )

            if verify_images and (max_verify is None or verify_count < max_verify):
                verify_count += 1
                summary.total_verified_images += 1
                try:
                    with open(file_path, "rb") as f:
                        data_bytes = io.BytesIO(f.read())
                        ok, err_msg, w, h, fmt = self.verify_image_stream(data_bytes)
                        entry.is_verified = True
                        if ok:
                            summary.valid_verified_images += 1
                            entry.width = w
                            entry.height = h
                            entry.format = fmt
                        else:
                            summary.corrupt_images_count += 1
                            entry.is_corrupt = True
                            entry.error_message = err_msg
                            summary.corrupt_files.append({"path": rel_str, "error": str(err_msg)})
                except Exception as ex:
                    summary.corrupt_images_count += 1
                    entry.is_corrupt = True
                    entry.error_message = str(ex)
                    summary.corrupt_files.append({"path": rel_str, "error": str(ex)})

            entries.append(entry)


def plan_benchmark_splits(
    entries: List[ImageEntry],
    target_per_class: int = 2000,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Plan a balanced, reproducible benchmark subset with mutually exclusive splits.
    
    Args:
        entries: List of enumerated ImageEntry records.
        target_per_class: Number of samples per class (default: 2,000 cats, 2,000 dogs).
        train_ratio: Proportion of target allocated to training (default: 0.70).
        val_ratio: Proportion of target allocated to validation (default: 0.15).
        test_ratio: Proportion of target allocated to test (default: 0.15).
        random_seed: Deterministic seed for reproducible selection.
        
    Returns:
        Dictionary containing manifest metadata, per-split lists, and verification flags.
    """
    import random

    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0")

    # Filter to valid, non-corrupt candidates
    valid_entries = [e for e in entries if e.is_valid_naming and not e.is_corrupt]
    cat_entries = sorted([e for e in valid_entries if e.label == "cat"], key=lambda e: e.filename)
    dog_entries = sorted([e for e in valid_entries if e.label == "dog"], key=lambda e: e.filename)

    available_cats = len(cat_entries)
    available_dogs = len(dog_entries)

    if available_cats < target_per_class or available_dogs < target_per_class:
        raise ValueError(
            f"Insufficient valid images to build {target_per_class}-per-class benchmark. "
            f"Available: cats={available_cats}, dogs={available_dogs}. "
            f"Minimum required per class is {target_per_class}."
        )

    # Deterministic sampling
    rng = random.Random(random_seed)
    sampled_cats = rng.sample(cat_entries, target_per_class)
    sampled_dogs = rng.sample(dog_entries, target_per_class)

    n_train_class = int(round(target_per_class * train_ratio))
    n_val_class = int(round(target_per_class * val_ratio))
    n_test_class = target_per_class - n_train_class - n_val_class

    # Partition cats
    cats_train = sampled_cats[:n_train_class]
    cats_val = sampled_cats[n_train_class : n_train_class + n_val_class]
    cats_test = sampled_cats[n_train_class + n_val_class :]

    # Partition dogs
    dogs_train = sampled_dogs[:n_train_class]
    dogs_val = sampled_dogs[n_train_class : n_train_class + n_val_class]
    dogs_test = sampled_dogs[n_train_class + n_val_class :]

    train_set = cats_train + dogs_train
    val_set = cats_val + dogs_val
    test_set = cats_test + dogs_test

    # Verify mutual exclusivity
    train_files = {e.filename for e in train_set}
    val_files = {e.filename for e in val_set}
    test_files = {e.filename for e in test_set}

    assert len(train_files & val_files) == 0, "Data leakage detected: train and val overlap!"
    assert len(train_files & test_files) == 0, "Data leakage detected: train and test overlap!"
    assert len(val_files & test_files) == 0, "Data leakage detected: val and test overlap!"

    return {
        "random_seed": random_seed,
        "target_per_class": target_per_class,
        "total_selected": len(train_set) + len(val_set) + len(test_set),
        "split_counts": {
            "train": {"total": len(train_set), "cats": len(cats_train), "dogs": len(dogs_train)},
            "val": {"total": len(val_set), "cats": len(cats_val), "dogs": len(dogs_val)},
            "test": {"total": len(test_set), "cats": len(cats_test), "dogs": len(dogs_test)},
        },
        "manifest": [
            {"filename": e.filename, "raw_path": e.raw_path, "label": e.label, "split": "train"}
            for e in train_set
        ]
        + [
            {"filename": e.filename, "raw_path": e.raw_path, "label": e.label, "split": "val"}
            for e in val_set
        ]
        + [
            {"filename": e.filename, "raw_path": e.raw_path, "label": e.label, "split": "test"}
            for e in test_set
        ],
    }


def generate_validation_markdown_report(summary: ValidationSummary) -> str:
    """Generate a clean, professional markdown summary report."""
    md = [
        "# Task 3: Dogs vs. Cats — Dataset Validation Audit Report",
        "",
        f"- **Audited Source**: `{summary.source_path}`",
        f"- **Source Type**: `{summary.source_type.upper()}`",
        f"- **Total Enumerated Entries**: `{summary.total_entries_enumerated:,}`",
        "",
        "## 1. Class Population & Enumeration Summary",
        "",
        "| Metric / Attribute | Count | Percentage of Candidates |",
        "| :--- | :--- | :--- |",
        f"| **Candidate Cat Images** | {summary.candidate_cats_count:,} | {summary.candidate_cats_count / max(1, summary.total_candidate_images) * 100:.1f}% |",
        f"| **Candidate Dog Images** | {summary.candidate_dogs_count:,} | {summary.candidate_dogs_count / max(1, summary.total_candidate_images) * 100:.1f}% |",
        f"| **Total Candidate Images** | {summary.total_candidate_images:,} | 100.0% |",
        f"| **Unexpected / Malformed Names** | {summary.unexpected_files_count} | - |",
        f"| **Duplicate Names Detected** | {summary.duplicate_names_count} | - |",
        f"| **Suspicious Path Traversals** | {summary.suspicious_paths_count} | - |",
        "",
        "## 2. Image Header Integrity & Verification",
        "",
        f"- **Images Header-Verified**: `{summary.total_verified_images:,}`",
        f"- **Valid Images (Verified)**: `{summary.valid_verified_images:,}`",
        f"- **Corrupt Images Identified**: `{summary.corrupt_images_count}`",
        "",
    ]

    if summary.corrupt_files:
        md.append("### Corrupt Files Detail:")
        for cf in summary.corrupt_files[:10]:
            md.append(f"- `{cf['path']}`: {cf['error']}")
        if len(summary.corrupt_files) > 10:
            md.append(f"- *... and {len(summary.corrupt_files) - 10} more corrupt files.*")
        md.append("")

    md.extend(
        [
            "## 3. Benchmark Feasibility Assessment",
            "",
            f"- **Target Benchmark**: 4,000 images (2,000 cats, 2,000 dogs).",
            f"- **Status**: {'FEASIBLE' if summary.candidate_cats_count >= 2000 and summary.candidate_dogs_count >= 2000 else 'INSUFFICIENT DATA'}.",
            "",
            "---",
            "*Validation audit performed via memory-bounded Pillow header inspection without decoding pixel arrays into RAM.*",
        ]
    )

    return "\n".join(md)


def export_benchmark_manifest_csv(
    manifest: List[Dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """
    Export benchmark split manifest to CSV.
    
    Args:
        manifest: List of dictionary entries with filename, raw_path, label, split.
        output_path: Target CSV destination path.
        
    Returns:
        Path to the created CSV file.
    """
    import csv

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["filename", "raw_path", "label", "split"]
    with open(out_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    return out_p


def generate_benchmark_manifest(
    source_path: str | Path,
    output_csv: str | Path,
    target_per_class: int = 2000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Enumerate source archive/directory and generate a reproducible benchmark split manifest CSV.
    
    Reads only metadata; does not extract image rasters to disk.
    
    Args:
        source_path: Path to dataset archive (train.zip) or directory.
        output_csv: Destination path for benchmark manifest CSV.
        target_per_class: Balanced samples per class (default: 2,000).
        random_seed: Deterministic random seed (default: 42).
        
    Returns:
        Split plan summary dictionary.
    """
    validator = DatasetValidator(source_path)
    # Enumerate members without verifying all 25,000 rasters (speed and memory efficiency)
    summary, entries = validator.validate(verify_images=False)

    plan = plan_benchmark_splits(
        entries=entries,
        target_per_class=target_per_class,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=random_seed,
    )

    export_benchmark_manifest_csv(plan["manifest"], output_csv)
    return plan


def main() -> int:
    """CLI runner for dataset validation and benchmark manifest generation."""
    default_zip = Path("task3-dogs-vs-cats/data/raw/train.zip")
    alt_zip = Path("data/raw/train.zip")

    target_path = default_zip if default_zip.exists() else alt_zip

    print("=" * 70)
    print("SkillCraft ML Portfolio — Task 3: Dogs vs Cats Dataset Validator")
    print("=" * 70)

    if not target_path.exists():
        print(f"\n[NOTICE] Raw dataset archive was NOT found at expected path:")
        print(f"         '{target_path}'")
        print("\nNext steps:")
        print("  1. Please download 'train.zip' from: https://www.kaggle.com/c/dogs-vs-cats/data")
        print(f"  2. Place the file at: {target_path}")
        print("  3. Refer to 'task3-dogs-vs-cats/DATASET_SETUP.md' for detailed instructions.")
        print("\nValidator exiting cleanly without errors (dataset waiting on manual placement).")
        return 0

    print(f"\nValidating dataset at: {target_path} ...")
    validator = DatasetValidator(target_path)
    summary, entries = validator.validate(verify_images=True, max_verify=500)

    if Path.cwd().name == "task3-dogs-vs-cats":
        report_file = Path("reports/dataset_validation_report.md")
        manifest_file = Path("reports/benchmark_manifest.csv")
    else:
        report_file = Path("task3-dogs-vs-cats/reports/dataset_validation_report.md")
        manifest_file = Path("task3-dogs-vs-cats/reports/benchmark_manifest.csv")

    report_md = generate_validation_markdown_report(summary)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(report_md, encoding="utf-8")

    print("\nValidation complete!")
    print(f"  Total candidate images: {summary.total_candidate_images:,}")
    print(f"  Cats: {summary.candidate_cats_count:,} | Dogs: {summary.candidate_dogs_count:,}")
    print(f"  Corrupt images: {summary.corrupt_images_count}")
    print(f"  Report generated: {report_file}")

    if summary.candidate_cats_count >= 2000 and summary.candidate_dogs_count >= 2000:
        print("\nGenerating reproducible 4,000-image benchmark manifest (seed=42)...")
        plan = plan_benchmark_splits(
            entries=entries,
            target_per_class=2000,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )
        export_benchmark_manifest_csv(plan["manifest"], manifest_file)
        counts = plan["split_counts"]
        print(f"  Manifest written to: {manifest_file}")
        print(f"  Total selected:      {plan['total_selected']:,} images")
        print(f"  - Training (70%):   {counts['train']['total']:,} ({counts['train']['cats']} cats, {counts['train']['dogs']} dogs)")
        print(f"  - Validation (15%): {counts['val']['total']:,} ({counts['val']['cats']} cats, {counts['val']['dogs']} dogs)")
        print(f"  - Test (15%):       {counts['test']['total']:,} ({counts['test']['cats']} cats, {counts['test']['dogs']} dogs)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

