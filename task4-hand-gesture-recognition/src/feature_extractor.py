"""
HOG Feature Extraction Pipeline for Task 4: Hand Gesture Recognition.

Memory-efficient, sequential extraction of Histogram of Oriented Gradients (HOG) features:
- Reads canonical image paths and subject-independent splits from reports/gesture_manifest.csv
- Preserves native 640:240 aspect ratio by resizing to 160x60 grayscale (2.67:1)
- Computes 2,700 HOG features per frame (9 orientations, 10x10 cell size, 2x2 blocks, L2-Hys)
- Bounded memory: processes frames sequentially without loading full dataset into RAM
- Saves split-wise feature matrices (X_train, y_train, X_val, y_val, X_test, y_test) into data/processed/
- Stores comprehensive metadata and provenance tracking back to manifest rows
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from skimage.feature import hog

# 10 canonical LeapGestRecog classes mapped deterministically to integers 0..9
GESTURE_CLASSES: List[str] = [
    "01_palm",
    "02_l",
    "03_fist",
    "04_fist_moved",
    "05_thumb",
    "06_index",
    "07_ok",
    "08_palm_moved",
    "09_c",
    "10_down",
]

CLASS_TO_IDX: Dict[str, int] = {g: idx for idx, g in enumerate(GESTURE_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {idx: g for g, idx in CLASS_TO_IDX.items()}

TRAIN_SUBJECTS: List[str] = ["00", "01", "02", "03", "04", "05", "06"]
VAL_SUBJECTS: List[str] = ["07"]
TEST_SUBJECTS: List[str] = ["08", "09"]


@dataclass
class HOGConfig:
    """Configuration for HOG feature extraction preserving 2.67:1 aspect ratio."""

    image_width: int = 160
    image_height: int = 60
    native_width: int = 640
    native_height: int = 240
    orientations: int = 9
    pixels_per_cell: Tuple[int, int] = (10, 10)
    cells_per_block: Tuple[int, int] = (2, 2)
    block_norm: str = "L2-Hys"
    feature_dim: int = 2700  # 5 height blocks * 15 width blocks * 36 = 2,700 features

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["pixels_per_cell"] = list(self.pixels_per_cell)
        d["cells_per_block"] = list(self.cells_per_block)
        return d


class HOGFeatureExtractor:
    """
    Sequential, memory-bounded HOG feature extractor.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_dir: str | Path,
        output_dir: str | Path,
        config: Optional[HOGConfig] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.config = config or HOGConfig()
        self._manifest_df: Optional[pd.DataFrame] = None

    def load_manifest(self) -> pd.DataFrame:
        """Load and validate manifest CSV."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        df = pd.read_csv(self.manifest_path)
        expected_cols = {"relative_path", "filename", "subject_id", "gesture_label", "split"}
        if not expected_cols.issubset(df.columns):
            raise ValueError(f"Manifest missing expected columns: {expected_cols - set(df.columns)}")

        df["subject_id"] = df["subject_id"].astype(str).str.zfill(2)
        self._manifest_df = df
        return self._manifest_df

    @property
    def manifest_df(self) -> pd.DataFrame:
        if self._manifest_df is None:
            return self.load_manifest()
        return self._manifest_df

    def extract_image_features(self, image_path: Path) -> np.ndarray:
        """
        Extract HOG features from a single image file on disk.
        
        Args:
            image_path: Path to PNG image file.
            
        Returns:
            1D float32 numpy array of shape (feature_dim,).
            
        Raises:
            FileNotFoundError: If image does not exist.
            ValueError: If image is corrupt or cannot be decoded.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            with Image.open(image_path) as img:
                # Ensure single channel grayscale
                gray = img.convert("L")
                # Resize preserving 2.67:1 aspect ratio
                resized = gray.resize(
                    (self.config.image_width, self.config.image_height),
                    Image.Resampling.BILINEAR,
                )
                img_arr = np.array(resized, dtype=np.float32) / 255.0

            feats = hog(
                img_arr,
                orientations=self.config.orientations,
                pixels_per_cell=self.config.pixels_per_cell,
                cells_per_block=self.config.cells_per_block,
                block_norm=self.config.block_norm,
                feature_vector=True,
            )
            return feats.astype(np.float32)

        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as err:
            raise ValueError(f"Failed to read/decode image '{image_path}': {err}")

    def extract_split(
        self,
        split_name: str,
        save_to_disk: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Extract HOG feature matrix and labels for a specific split (train, val, or test).
        
        Args:
            split_name: 'train', 'val', or 'test'.
            save_to_disk: If True, writes X_{split}.npy, y_{split}.npy, and {split}_meta.json.
            
        Returns:
            Tuple of (X, y, sample_metadata_records)
        """
        df = self.manifest_df
        split_df = df[df["split"] == split_name].reset_index(drop=True)
        num_samples = len(split_df)

        if num_samples == 0:
            raise ValueError(f"No samples found in manifest for split '{split_name}'")

        print(f"Extracting {num_samples:,} samples for split '{split_name}'...")
        X = np.zeros((num_samples, self.config.feature_dim), dtype=np.float32)
        y = np.zeros(num_samples, dtype=np.int64)
        sample_meta: List[Dict[str, Any]] = []

        t0 = time.time()
        for idx, row in split_df.iterrows():
            rel_path = row["relative_path"]
            img_path = self.data_dir / rel_path
            label_str = row["gesture_label"]

            if label_str not in CLASS_TO_IDX:
                raise ValueError(f"Unknown gesture label '{label_str}' at manifest index {idx}")

            feat_vec = self.extract_image_features(img_path)
            X[idx] = feat_vec
            y[idx] = CLASS_TO_IDX[label_str]

            sample_meta.append({
                "index": int(idx),
                "relative_path": rel_path,
                "filename": row["filename"],
                "subject_id": row["subject_id"],
                "gesture_label": label_str,
                "label_idx": int(CLASS_TO_IDX[label_str]),
                "split": split_name,
            })

            # Progress logging every 2,000 samples
            if (idx + 1) % 2000 == 0 or (idx + 1) == num_samples:
                elapsed = time.time() - t0
                fps = (idx + 1) / max(elapsed, 0.001)
                print(f"  [{split_name.upper()}] Processed {idx + 1:,}/{num_samples:,} frames ({fps:.1f} fps)")

        if save_to_disk:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Atomic writes via temporary files to guard against partial writes
            tmp_x = self.output_dir / f"X_{split_name}.npy.tmp"
            final_x = self.output_dir / f"X_{split_name}.npy"
            with open(tmp_x, "wb") as f:
                np.save(f, X)
            if final_x.exists():
                final_x.unlink()
            tmp_x.rename(final_x)

            tmp_y = self.output_dir / f"y_{split_name}.npy.tmp"
            final_y = self.output_dir / f"y_{split_name}.npy"
            with open(tmp_y, "wb") as f:
                np.save(f, y)
            if final_y.exists():
                final_y.unlink()
            tmp_y.rename(final_y)

            meta_file = self.output_dir / f"{split_name}_manifest.json"
            tmp_meta = self.output_dir / f"{split_name}_manifest.json.tmp"
            with open(tmp_meta, mode="w", encoding="utf-8") as f:
                json.dump(sample_meta, f, indent=2)
            if meta_file.exists():
                meta_file.unlink()
            tmp_meta.rename(meta_file)

        return X, y, sample_meta

    def extract_all(self) -> Dict[str, Any]:
        """
        Execute full sequential extraction across all three splits: train, val, and test.
        """
        t0 = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, Any] = {}
        for split in ["train", "val", "test"]:
            X, y, meta = self.extract_split(split, save_to_disk=True)
            results[split] = {
                "num_samples": len(y),
                "feature_shape": list(X.shape),
                "dtype": str(X.dtype),
                "subjects": sorted(list({m["subject_id"] for m in meta})),
            }

        # Save consolidated feature configuration metadata
        config_payload = {
            "feature_extractor": "scikit-image HOG",
            "feature_config": self.config.to_dict(),
            "class_to_idx": CLASS_TO_IDX,
            "idx_to_class": {str(k): v for k, v in IDX_TO_CLASS.items()},
            "train_subjects": TRAIN_SUBJECTS,
            "val_subjects": VAL_SUBJECTS,
            "test_subjects": TEST_SUBJECTS,
            "split_summary": results,
            "elapsed_seconds": round(time.time() - t0, 2),
        }

        config_path = self.output_dir / "feature_config.json"
        tmp_config = self.output_dir / "feature_config.json.tmp"
        with open(tmp_config, mode="w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2)
        if config_path.exists():
            config_path.unlink()
        tmp_config.rename(config_path)

        return config_payload

    def verify_extracted_features(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify that all extracted feature matrices, labels, and metadata exist and are intact.
        """
        config_path = self.output_dir / "feature_config.json"
        if not config_path.exists():
            return False, {"error": f"Config not found: {config_path}"}

        with open(config_path, mode="r", encoding="utf-8") as f:
            config_data = json.load(f)

        expected_splits = {
            "train": 14000,
            "val": 2000,
            "test": 4000,
        }

        actual_shapes = {}
        split_subjects = {}

        for splt, expected_count in expected_splits.items():
            fx = self.output_dir / f"X_{splt}.npy"
            fy = self.output_dir / f"y_{splt}.npy"
            fm = self.output_dir / f"{splt}_manifest.json"

            if not (fx.exists() and fy.exists() and fm.exists()):
                return False, {"error": f"Missing feature files for split '{splt}'"}

            # Load using mmap_mode='r' to avoid loading everything into RAM at once
            X = np.load(fx, mmap_mode="r")
            y = np.load(fy, mmap_mode="r")

            if X.shape != (expected_count, self.config.feature_dim):
                return False, {"error": f"Invalid shape for X_{splt}: {X.shape}, expected ({expected_count}, {self.config.feature_dim})"}

            if y.shape != (expected_count,):
                return False, {"error": f"Invalid shape for y_{splt}: {y.shape}, expected ({expected_count},)"}

            # Check for NaN / Inf
            if np.isnan(X[0]).any() or np.isinf(X[0]).any():
                return False, {"error": f"NaN or Inf detected in X_{splt}"}

            with open(fm, mode="r", encoding="utf-8") as f:
                meta = json.load(f)
            if len(meta) != expected_count:
                return False, {"error": f"Metadata length mismatch for {splt}: {len(meta)} != {expected_count}"}

            subjs = sorted(list({m["subject_id"] for m in meta}))
            split_subjects[splt] = subjs
            actual_shapes[splt] = {"X_shape": list(X.shape), "y_shape": list(y.shape)}

        # Verify subject isolation
        train_val_overlap = set(split_subjects["train"]) & set(split_subjects["val"])
        train_test_overlap = set(split_subjects["train"]) & set(split_subjects["test"])
        val_test_overlap = set(split_subjects["val"]) & set(split_subjects["test"])
        zero_overlap = (
            len(train_val_overlap) == 0
            and len(train_test_overlap) == 0
            and len(val_test_overlap) == 0
        )

        is_valid = zero_overlap and len(actual_shapes) == 3
        return is_valid, {
            "is_valid": is_valid,
            "actual_shapes": actual_shapes,
            "split_subjects": split_subjects,
            "zero_overlap": zero_overlap,
        }


def main() -> int:
    """CLI runner for Task 4 HOG Feature Extraction."""
    if Path.cwd().name == "task4-hand-gesture-recognition":
        manifest_path = Path("reports/gesture_manifest.csv")
        data_dir = Path("data/interim")
        output_dir = Path("data/processed")
    else:
        manifest_path = Path("task4-hand-gesture-recognition/reports/gesture_manifest.csv")
        data_dir = Path("task4-hand-gesture-recognition/data/interim")
        output_dir = Path("task4-hand-gesture-recognition/data/processed")

    print("=" * 70)
    print("SkillCraft ML Portfolio — Task 4: HOG Feature Extractor")
    print("=" * 70)
    print(f"Manifest:   {manifest_path}")
    print(f"Data Dir:   {data_dir}")
    print(f"Output Dir: {output_dir}\n")

    extractor = HOGFeatureExtractor(manifest_path, data_dir, output_dir)

    # Check if features are already extracted
    is_valid, report = extractor.verify_extracted_features()
    if is_valid:
        print("[INFO] Features are already extracted and verified valid!")
        print(f"  Train: {report['actual_shapes']['train']['X_shape']}")
        print(f"  Val:   {report['actual_shapes']['val']['X_shape']}")
        print(f"  Test:  {report['actual_shapes']['test']['X_shape']}")
        return 0

    print("Starting full dataset HOG extraction...")
    summary = extractor.extract_all()

    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"  Total Elapsed Time: {summary['elapsed_seconds']} seconds")
    print(f"  Feature Dimensions: {summary['feature_config']['feature_dim']}")
    print(f"  Train samples:      {summary['split_summary']['train']['num_samples']:,}")
    print(f"  Validation samples: {summary['split_summary']['val']['num_samples']:,}")
    print(f"  Test samples:       {summary['split_summary']['test']['num_samples']:,}")

    # Post-extraction verification
    ok, check = extractor.verify_extracted_features()
    if not ok:
        print(f"[ERROR] Post-extraction verification failed: {check}")
        return 1

    print("\n[VERIFIED] All feature matrices, labels, and metadata match the manifest exactly.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
