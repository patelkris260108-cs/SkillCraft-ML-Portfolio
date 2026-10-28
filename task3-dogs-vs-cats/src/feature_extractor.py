"""
HOG Feature Extraction module for Task 3: Dogs vs. Cats Benchmark.

Extracts memory-efficient Histogram of Oriented Gradients (HOG) descriptors
from 64x64 grayscale-normalized benchmark images.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from skimage.feature import hog


@dataclass
class HOGConfig:
    """Configuration parameters for HOG feature extraction."""

    target_size: Tuple[int, int] = (64, 64)
    orientations: int = 9
    pixels_per_cell: Tuple[int, int] = (8, 8)
    cells_per_block: Tuple[int, int] = (2, 2)
    block_norm: str = "L2-Hys"
    transform_sqrt: bool = True

    @property
    def feature_dim(self) -> int:
        """Calculate and programmatically verify feature vector dimension."""
        dummy = np.zeros(self.target_size, dtype=np.float32)
        features = hog(
            dummy,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            transform_sqrt=self.transform_sqrt,
        )
        return int(features.shape[0])

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to serializable dictionary."""
        d = asdict(self)
        d["feature_dim"] = self.feature_dim
        d["target_size"] = list(self.target_size)
        d["pixels_per_cell"] = list(self.pixels_per_cell)
        d["cells_per_block"] = list(self.cells_per_block)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HOGConfig:
        """Create configuration from dictionary."""
        return cls(
            target_size=tuple(data.get("target_size", (64, 64))),
            orientations=int(data.get("orientations", 9)),
            pixels_per_cell=tuple(data.get("pixels_per_cell", (8, 8))),
            cells_per_block=tuple(data.get("cells_per_block", (2, 2))),
            block_norm=str(data.get("block_norm", "L2-Hys")),
            transform_sqrt=bool(data.get("transform_sqrt", True)),
        )


LABEL_MAP: Dict[str, int] = {"cat": 0, "dog": 1}
INV_LABEL_MAP: Dict[int, str] = {0: "cat", 1: "dog"}


class HOGFeatureExtractor:
    """
    Modular, memory-bounded HOG feature extractor for image classification.
    """

    def __init__(self, config: Optional[HOGConfig] = None):
        self.config = config or HOGConfig()
        self._expected_dim = self.config.feature_dim

    def extract_from_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Extract HOG features from a PIL Image or NumPy array.
        
        Args:
            image: Input image (RGB or grayscale).
            
        Returns:
            1D float32 NumPy array of shape (feature_dim,).
        """
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                # Convert RGB array to PIL Image
                pil_img = Image.fromarray(image.astype(np.uint8)).convert("L")
            elif image.ndim == 2:
                pil_img = Image.fromarray(image.astype(np.uint8))
            else:
                raise ValueError(f"Unsupported array shape: {image.shape}")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("L")
        else:
            raise TypeError(f"Expected PIL Image or np.ndarray, got {type(image)}")

        # Consistent resize
        if pil_img.size != self.config.target_size:
            pil_img = pil_img.resize(self.config.target_size, Image.Resampling.BILINEAR)

        # Convert to float32 normalized [0, 1]
        img_array = np.asarray(pil_img, dtype=np.float32) / 255.0

        features = hog(
            img_array,
            orientations=self.config.orientations,
            pixels_per_cell=self.config.pixels_per_cell,
            cells_per_block=self.config.cells_per_block,
            block_norm=self.config.block_norm,
            transform_sqrt=self.config.transform_sqrt,
        )

        feat_vector = np.asarray(features, dtype=np.float32)
        if feat_vector.shape[0] != self._expected_dim:
            raise ValueError(
                f"Extracted feature length {feat_vector.shape[0]} does not match expected {self._expected_dim}"
            )

        return feat_vector

    def extract_with_visualization(
        self, image: Union[Image.Image, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Image.Image]:
        """
        Extract HOG features and return feature vector alongside the 2D gradient visualization.
        
        Returns:
            Tuple of (feat_vector, hog_image, normalized_grayscale_pil_img)
        """
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                pil_img = Image.fromarray(image.astype(np.uint8)).convert("L")
            elif image.ndim == 2:
                pil_img = Image.fromarray(image.astype(np.uint8))
            else:
                raise ValueError(f"Unsupported array shape: {image.shape}")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("L")
        else:
            raise TypeError(f"Expected PIL Image or np.ndarray, got {type(image)}")

        if pil_img.size != self.config.target_size:
            pil_img = pil_img.resize(self.config.target_size, Image.Resampling.BILINEAR)

        img_array = np.asarray(pil_img, dtype=np.float32) / 255.0

        features, hog_image = hog(
            img_array,
            orientations=self.config.orientations,
            pixels_per_cell=self.config.pixels_per_cell,
            cells_per_block=self.config.cells_per_block,
            block_norm=self.config.block_norm,
            transform_sqrt=self.config.transform_sqrt,
            visualize=True,
        )

        feat_vector = np.asarray(features, dtype=np.float32)
        return feat_vector, hog_image.astype(np.float32), pil_img

    def extract_from_path(self, path: Union[str, Path]) -> np.ndarray:
        """
        Stream and extract HOG features from a single image file on disk.
        """
        img_path = Path(path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        try:
            with Image.open(img_path) as img:
                return self.extract_from_image(img)
        except Exception as ex:
            raise RuntimeError(f"Failed extracting HOG from '{img_path}': {ex}") from ex

    def extract_dataset(
        self,
        manifest_path: Union[str, Path],
        data_dir: Union[str, Path],
        split: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract features for all samples in a manifest partition.
        
        Args:
            manifest_path: Path to benchmark_manifest.csv.
            data_dir: Root directory of extracted interim images (containing train/, val/, test/).
            split: Optional filter ('train', 'val', 'test'). If None, extracts all.
            
        Returns:
            Tuple of:
                X: 2D float32 array of shape (N, feature_dim)
                y: 1D int64 array of shape (N,)
                filenames: List of original filenames aligned with rows
        """
        manifest_file = Path(manifest_path)
        data_root = Path(data_dir)

        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")

        records: List[Dict[str, str]] = []
        with open(manifest_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split is None or row["split"].lower() == split.lower():
                    records.append(row)

        n_samples = len(records)
        if n_samples == 0:
            raise ValueError(f"No records found for split '{split}' in {manifest_file}")

        # Preallocate memory for exact bounding
        X = np.empty((n_samples, self._expected_dim), dtype=np.float32)
        y = np.empty(n_samples, dtype=np.int64)
        filenames: List[str] = []

        for idx, row in enumerate(records):
            filename = row["filename"]
            label_str = row["label"].lower()
            split_str = row["split"].lower()

            if label_str not in LABEL_MAP:
                raise ValueError(f"Unknown label '{label_str}' in row {idx}: {row}")

            img_path = data_root / split_str / label_str / filename
            if not img_path.exists():
                raise FileNotFoundError(f"Image missing at expected location: {img_path}")

            feat = self.extract_from_path(img_path)
            X[idx] = feat
            y[idx] = LABEL_MAP[label_str]
            filenames.append(filename)

        return X, y, filenames

    def extract_and_save_partitions(
        self,
        manifest_path: Union[str, Path],
        data_dir: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> Dict[str, Tuple[int, int]]:
        """
        Extract and save precomputed feature arrays for train, val, and test partitions.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        partition_info: Dict[str, Tuple[int, int]] = {}

        for s in ("train", "val", "test"):
            X, y, filenames = self.extract_dataset(manifest_path, data_dir, split=s)
            np.save(out_path / f"X_{s}.npy", X)
            np.save(out_path / f"y_{s}.npy", y)
            (out_path / f"filenames_{s}.json").write_text(json.dumps(filenames), encoding="utf-8")
            partition_info[s] = (X.shape[0], X.shape[1])

        # Save config
        config_file = out_path / "hog_config.json"
        config_file.write_text(json.dumps(self.config.to_dict(), indent=2), encoding="utf-8")

        return partition_info


def load_partition(
    processed_dir: Union[str, Path],
    split: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load precomputed features, labels, and filenames for a given partition.
    """
    p_dir = Path(processed_dir)
    x_file = p_dir / f"X_{split}.npy"
    y_file = p_dir / f"y_{split}.npy"
    fn_file = p_dir / f"filenames_{split}.json"

    if not x_file.exists() or not y_file.exists():
        raise FileNotFoundError(f"Precomputed features not found for split '{split}' in {p_dir}")

    X = np.load(x_file)
    y = np.load(y_file)
    filenames = json.loads(fn_file.read_text(encoding="utf-8")) if fn_file.exists() else []

    return X, y, filenames
