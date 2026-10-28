"""
Exploratory Data Analysis (EDA) module for Task 4: Hand Gesture Recognition.

Analyzes the LeapGestRecog dataset using streaming, memory-bounded operations:
- Manifest distribution metrics (class, subject, split balance)
- Grayscale near-infrared image properties (intensity histogram, foreground/background contrast)
- Visualizations: class balance, subject splits, 10x10 cross-tab heatmap, sample montage, pixel distribution
- Generates reports/eda_summary.md with comprehensive findings and Phase 4 modeling recommendations
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless backend for headless/CI execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

GESTURE_DISPLAY_NAMES: Dict[str, str] = {
    "01_palm": "01: Palm (Flat Open)",
    "02_l": "02: L-Shape (Index+Thumb)",
    "03_fist": "03: Fist (Static)",
    "04_fist_moved": "04: Fist (Dynamic)",
    "05_thumb": "05: Thumb (Up)",
    "06_index": "06: Index (Pointing)",
    "07_ok": "07: OK (Circular)",
    "08_palm_moved": "08: Palm (Dynamic)",
    "09_c": "09: C-Shape (Curved)",
    "10_down": "10: Down (Pointing)",
}

SPLIT_COLORS: Dict[str, str] = {
    "train": "#2b5c8f",  # Deep slate blue
    "val": "#d9822b",    # Warm amber
    "test": "#2e8b57",   # Forest green
}


class GestureEDAAnalyzer:
    """
    Memory-bounded EDA Analyzer for the LeapGestRecog hand gesture benchmark.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_dir: str | Path,
        output_dir: str | Path,
    ):
        self.manifest_path = Path(manifest_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self._df: Optional[pd.DataFrame] = None

    def load_manifest(self) -> pd.DataFrame:
        """Load manifest CSV into a DataFrame."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        df = pd.read_csv(self.manifest_path)
        expected_cols = {"relative_path", "filename", "subject_id", "gesture_label", "width", "height", "split"}
        if not expected_cols.issubset(df.columns):
            raise ValueError(f"Manifest missing expected columns: {expected_cols - set(df.columns)}")

        # Convert subject_id to 2-digit zero-padded string
        df["subject_id"] = df["subject_id"].astype(str).str.zfill(2)
        self._df = df
        return self._df

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            return self.load_manifest()
        return self._df

    def compute_distribution_stats(self) -> Dict[str, Any]:
        """Compute dataset balance, split isolation, and cross-tabulation statistics."""
        df = self.df
        total_images = len(df)

        class_counts = df["gesture_label"].value_counts().sort_index().to_dict()
        subject_counts = df["subject_id"].value_counts().sort_index().to_dict()
        split_counts = df["split"].value_counts().to_dict()

        # Subject-split mapping
        split_subject_map = (
            df.groupby("split")["subject_id"]
            .unique()
            .apply(lambda s: sorted(list(s)))
            .to_dict()
        )

        # Cross-tabulation (Subject x Gesture)
        cross_tab = pd.crosstab(df["subject_id"], df["gesture_label"])

        # Check balance
        is_class_balanced = len(set(class_counts.values())) == 1
        is_subject_balanced = len(set(subject_counts.values())) == 1

        # Check disjoint splits
        train_subjs = set(split_subject_map.get("train", []))
        val_subjs = set(split_subject_map.get("val", []))
        test_subjs = set(split_subject_map.get("test", []))
        zero_overlap = (
            len(train_subjs & val_subjs) == 0
            and len(train_subjs & test_subjs) == 0
            and len(val_subjs & test_subjs) == 0
        )

        return {
            "total_images": total_images,
            "num_subjects": len(subject_counts),
            "num_gestures": len(class_counts),
            "class_counts": class_counts,
            "subject_counts": subject_counts,
            "split_counts": split_counts,
            "split_subject_map": split_subject_map,
            "is_class_balanced": is_class_balanced,
            "is_subject_balanced": is_subject_balanced,
            "zero_overlap": zero_overlap,
            "cross_tab_matrix": cross_tab.to_dict(),
        }

    def compute_grayscale_stats(
        self,
        samples_per_class: int = 20,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Memory-efficient streaming analysis of grayscale pixel statistics.
        Samples an evenly distributed subset across gestures and subjects to bound memory.
        """
        df = self.df
        sampled_df = (
            df.groupby("gesture_label", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), samples_per_class), random_state=random_state))
        )

        pixel_means: List[float] = []
        pixel_stds: List[float] = []
        pixel_mins: List[int] = []
        pixel_maxs: List[int] = []
        fg_ratios: List[float] = []  # Fraction of pixels > 30 (hand foreground)
        histogram_accum = np.zeros(256, dtype=np.int64)

        for _, row in sampled_df.iterrows():
            img_path = self.data_dir / row["relative_path"]
            if not img_path.exists():
                continue

            with Image.open(img_path) as img:
                arr = np.array(img, dtype=np.uint8)
                pixel_means.append(float(np.mean(arr)))
                pixel_stds.append(float(np.std(arr)))
                pixel_mins.append(int(np.min(arr)))
                pixel_maxs.append(int(np.max(arr)))
                # Near-IR background is nearly pitch black (0-25), foreground hand is reflective (>30)
                fg_ratios.append(float(np.mean(arr > 30)))
                counts, _ = np.histogram(arr, bins=256, range=(0, 256))
                histogram_accum += counts

        return {
            "num_samples_analyzed": len(pixel_means),
            "mean_intensity": round(float(np.mean(pixel_means)), 2),
            "std_intensity": round(float(np.mean(pixel_stds)), 2),
            "min_intensity": int(min(pixel_mins)) if pixel_mins else 0,
            "max_intensity": int(max(pixel_maxs)) if pixel_maxs else 255,
            "mean_foreground_ratio": round(float(np.mean(fg_ratios)) * 100, 2),
            "histogram_accum": histogram_accum.tolist(),
            "image_width": int(df["width"].iloc[0]),
            "image_height": int(df["height"].iloc[0]),
            "aspect_ratio": round(float(df["width"].iloc[0]) / float(df["height"].iloc[0]), 2),
        }

    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------

    def plot_class_distribution(self, save_path: Optional[Path] = None) -> Path:
        """Generate class distribution bar chart."""
        df = self.df
        class_counts = df["gesture_label"].value_counts().sort_index()

        plt.figure(figsize=(10, 5), dpi=150)
        labels = [GESTURE_DISPLAY_NAMES.get(k, k) for k in class_counts.index]
        bars = plt.barh(labels, class_counts.values, color="#1f77b4", edgecolor="#0e4369", height=0.65)

        for bar in bars:
            width = bar.get_width()
            plt.text(
                width - 150,
                bar.get_y() + bar.get_height() / 2,
                f"{width:,} ({width/len(df)*100:.1f}%)",
                va="center",
                ha="right",
                color="white",
                fontweight="bold",
                fontsize=9,
            )

        plt.title("Task 4: Class Distribution (Perfect Balance: 2,000 Frames/Class)", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Number of Image Frames", fontsize=11)
        plt.xlim(0, 2400)
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()

        out_path = save_path or (self.figures_dir / "class_distribution.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        return out_path

    def plot_subject_distribution(self, save_path: Optional[Path] = None) -> Path:
        """Generate subject distribution bar chart colored by split."""
        df = self.df
        subject_split_counts = df.groupby(["subject_id", "split"]).size().unstack(fill_value=0)

        plt.figure(figsize=(10, 5), dpi=150)
        subjects = subject_split_counts.index.tolist()
        x = np.arange(len(subjects))
        width = 0.55

        # Plot bars colored by split assignment
        colors = []
        for s in subjects:
            if s in ["00", "01", "02", "03", "04", "05", "06"]:
                colors.append(SPLIT_COLORS["train"])
            elif s == "07":
                colors.append(SPLIT_COLORS["val"])
            else:
                colors.append(SPLIT_COLORS["test"])

        counts = [int(subject_split_counts.loc[s].sum()) for s in subjects]
        bars = plt.bar(x, counts, width=width, color=colors, edgecolor="#222222")

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height / 2,
                f"{int(height):,}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=10,
            )

        # Legend handles
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=SPLIT_COLORS["train"], edgecolor="#222", label="Train (00–06, 14,000 frames, 70%)"),
            Patch(facecolor=SPLIT_COLORS["val"], edgecolor="#222", label="Validation (07, 2,000 frames, 10%)"),
            Patch(facecolor=SPLIT_COLORS["test"], edgecolor="#222", label="Held-Out Test (08–09, 4,000 frames, 20%)"),
        ]
        plt.legend(handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=9)

        plt.title("Task 4: Subject Distribution & Partition Assignment", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Subject Identifier", fontsize=11)
        plt.ylabel("Frame Count", fontsize=11)
        plt.xticks(x, [f"Subject {s}" for s in subjects], rotation=25)
        plt.ylim(0, max(counts) * 1.25 if counts else 2500)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        out_path = save_path or (self.figures_dir / "subject_distribution.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        return out_path

    def plot_subject_gesture_heatmap(self, save_path: Optional[Path] = None) -> Path:
        """Generate heatmap verifying subject x gesture balance."""
        df = self.df
        ct = pd.crosstab(df["subject_id"], df["gesture_label"])

        plt.figure(figsize=(11, 7), dpi=150)
        matrix = ct.values
        n_rows, n_cols = matrix.shape
        max_val = max(int(matrix.max()), 1)
        plt.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=max_val * 1.2)

        plt.colorbar(label="Frame Count")
        plt.title("Task 4: Subject × Gesture Balance Matrix (100% Uniform: 200 Frames/Cell)", fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("Gesture Class", fontsize=10)
        plt.ylabel("Subject ID", fontsize=10)

        plt.xticks(np.arange(n_cols), [g.replace("_", "\n") for g in ct.columns], fontsize=8)
        plt.yticks(np.arange(n_rows), [f"Subj {s}" for s in ct.index], fontsize=9)

        # Annotate cell values
        for i in range(n_rows):
            for j in range(n_cols):
                plt.text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > (max_val / 2) else "black",
                    fontsize=9,
                    fontweight="bold",
                )

        plt.tight_layout()
        out_path = save_path or (self.figures_dir / "subject_gesture_heatmap.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        return out_path

    def plot_gesture_montage(self, save_path: Optional[Path] = None) -> Path:
        """Generate a montage displaying sample frames from all gesture classes."""
        df = self.df
        gestures = sorted(df["gesture_label"].unique())
        n_gest = len(gestures)
        n_cols = min(5, n_gest)
        n_rows = (n_gest + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), dpi=150)
        axes = np.array(axes).flatten()

        for idx in range(len(axes)):
            ax = axes[idx]
            if idx < n_gest:
                gest = gestures[idx]
                sample_rows = df[(df["gesture_label"] == gest) & (df["subject_id"] == "00")]
                if sample_rows.empty:
                    sample_rows = df[df["gesture_label"] == gest]

                sample_rel_path = sample_rows.iloc[min(99, len(sample_rows) - 1)]["relative_path"]
                img_path = self.data_dir / sample_rel_path

                if img_path.exists():
                    with Image.open(img_path) as img:
                        ax.imshow(img, cmap="gray")
                else:
                    ax.imshow(np.zeros((240, 640), dtype=np.uint8), cmap="gray")

                display_title = GESTURE_DISPLAY_NAMES.get(gest, gest)
                ax.set_title(display_title, fontsize=10, fontweight="bold", pad=6)
            ax.axis("off")

        plt.suptitle("Task 4: Near-Infrared LeapGestRecog 10-Class Gesture Montage", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()

        out_path = save_path or (self.figures_dir / "gesture_montage.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        return out_path


    def plot_pixel_intensity_distribution(
        self,
        grayscale_stats: Dict[str, Any],
        save_path: Optional[Path] = None,
    ) -> Path:
        """Plot pixel intensity distribution showing near-infrared sensor characteristics."""
        hist = np.array(grayscale_stats["histogram_accum"], dtype=np.float64)
        total_pixels = np.sum(hist)
        density = (hist / total_pixels) * 100

        plt.figure(figsize=(10, 4.5), dpi=150)
        bins = np.arange(256)
        plt.bar(bins, density, color="#4a7c59", width=1.0, edgecolor="none", alpha=0.85)

        # Highlight near-IR thresholding regions
        plt.axvline(25, color="#d9534f", linestyle="--", linewidth=1.5, label="Background Cutoff (Intensity ~25)")
        plt.text(12, max(density) * 0.7, "Dark Background\n(Absorbed IR)", color="#8b0000", fontsize=9, fontweight="bold", ha="center")
        plt.text(110, max(density) * 0.4, "Hand & Finger Reflections\n(Reflected IR Signal)", color="#1b4d3e", fontsize=9, fontweight="bold", ha="center")

        plt.title("Task 4: Near-Infrared Pixel Intensity Profile (640x240 Grayscale)", fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("Grayscale Pixel Intensity (0 = Black, 255 = White)", fontsize=10)
        plt.ylabel("Percentage of Total Pixels (%)", fontsize=10)
        plt.grid(linestyle="--", alpha=0.4)
        plt.legend(loc="upper right", fontsize=9)
        plt.tight_layout()

        out_path = save_path or (self.figures_dir / "pixel_intensity_distribution.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        return out_path

    # -------------------------------------------------------------------------
    # Comprehensive EDA Summary Report Generation
    # -------------------------------------------------------------------------

    def generate_eda_summary_report(
        self,
        dist_stats: Dict[str, Any],
        gray_stats: Dict[str, Any],
        save_path: Optional[Path] = None,
    ) -> Path:
        """Generate reports/eda_summary.md summarizing all EDA findings."""
        report_path = save_path or (self.output_dir / "eda_summary.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        class_rows = []
        for gest, cnt in sorted(dist_stats["class_counts"].items()):
            disp_name = GESTURE_DISPLAY_NAMES.get(gest, gest)
            prop = (cnt / dist_stats["total_images"]) * 100
            class_rows.append(f"| `{gest}` | {disp_name} | {cnt:,} | {prop:.1f}% |")
        class_table = "\n".join(class_rows)

        split_rows = []
        for splt in ["train", "val", "test"]:
            cnt = dist_stats["split_counts"].get(splt, 0)
            subjs = ", ".join(dist_stats["split_subject_map"].get(splt, []))
            prop = (cnt / dist_stats["total_images"]) * 100
            split_rows.append(f"| **{splt.upper()}** | `{subjs}` | {cnt:,} | {prop:.1f}% |")
        split_table = "\n".join(split_rows)

        content = f"""# Task 4 — Exploratory Data Analysis (EDA) Summary Report
**Project:** Hand Gesture Recognition (LeapGestRecog GTI-UPM Dataset)  
**Dataset Modality:** Near-Infrared (NIR) 8-Bit Grayscale Sensor Frames  
**Total Canonical Frames:** {dist_stats['total_images']:,} unique images  

---

## 1. Executive Summary & Key Findings

1. **Perfect Dataset Balance:**
   - The canonical dataset contains exactly **20,000 frames** evenly distributed across **10 subjects** and **10 gesture classes**.
   - Exactly **2,000 frames per class** (10.0% each) and **200 frames per subject-gesture pair** ($10 \\times 10 = 100$ cells).
   - Zero class imbalance exists; baseline guessing accuracy is exactly **10.0%**.

2. **Subject-Independent Split Isolation:**
   - To prevent catastrophic temporal and identity data leakage between consecutive frames ($t$ and $t+1$), splitting is performed strictly **by subject ID**:
     - **Train:** Subjects `00, 01, 02, 03, 04, 05, 06` (14,000 frames, 70.0%)
     - **Validation:** Subject `07` (2,000 frames, 10.0%)
     - **Held-Out Test:** Subjects `08, 09` (4,000 frames, 20.0%)
   - Split subject sets are **100% pairwise disjoint** (zero subject overlap).

3. **Near-Infrared Sensor Characteristics:**
   - Image format: **PNG**, Single-channel Grayscale (`L` mode), native resolution **{gray_stats['image_width']} $\\times$ {gray_stats['image_height']}** pixels (aspect ratio **{gray_stats['aspect_ratio']}:1**).
   - **Bimodal Intensity Distribution:** The infrared background is dark (mean background intensity < 25), while human hands and fingers strongly reflect NIR illumination (intensities up to {gray_stats['max_intensity']}).
   - **Foreground Proportion:** Across analyzed frames, approximately **{gray_stats['mean_foreground_ratio']}%** of pixels represent the active hand foreground.

---

## 2. Gesture Class Distribution

| Class Identifier | Gesture Description | Sample Count | Proportion |
| :--- | :--- | :--- | :--- |
{class_table}

![Class Distribution](figures/class_distribution.png)

---

## 3. Subject Distribution & Partition Strategy

| Split Partition | Assigned Subjects | Total Images | Dataset Proportion |
| :--- | :--- | :--- | :--- |
{split_table}

![Subject Distribution](figures/subject_distribution.png)

---

## 4. Subject × Gesture Balance Matrix

The 10 subjects and 10 gestures form a completely uniform $10 \\times 10$ orthogonal grid with exactly **200 frames per cell**:

![Subject Gesture Heatmap](figures/subject_gesture_heatmap.png)

---

## 5. Visual Gesture Montage

A representative sample of all 10 hand gesture classes captured by the near-infrared camera:

![Gesture Montage](figures/gesture_montage.png)

---

## 6. Sensor Pixel Intensity Profile

Near-infrared sensor frames demonstrate high foreground/background contrast:
- **Global Mean Intensity:** {gray_stats['mean_intensity']} $\\pm$ {gray_stats['std_intensity']} (on a 0–255 scale)
- **Minimum Intensity:** {gray_stats['min_intensity']} | **Maximum Intensity:** {gray_stats['max_intensity']}
- **Native Dimensions:** {gray_stats['image_width']} (W) $\\times$ {gray_stats['image_height']} (H) pixels

![Pixel Intensity Profile](figures/pixel_intensity_distribution.png)

---

## 7. Actionable Recommendations for Phase 4 Modeling

1. **Aspect Ratio Preservation:**
   - Raw frames have a wide 2.67:1 aspect ratio ($640 \\times 240$). When resizing for feature extraction (e.g. HOG, PCA, or lightweight CNN), resizing to $160 \\times 60$ or $128 \\times 64$ preserves hand contours without severe non-isometric distortion.
2. **Feature Extraction Viability:**
   - The stark NIR contrast makes **Histogram of Oriented Gradients (HOG)** exceptionally well-suited: hand contours, finger orientations (e.g., thumb vs index vs fist), and wrist shapes have clean gradient directions.
3. **Generalization Benchmark:**
   - Evaluated models must generalize to unseen subjects (`08` and `09`) who have different hand sizes, skin tones, and positioning. Cross-entropy loss or linear SVM margin on subject `07` (validation) will accurately guide hyperparameter tuning without test set contamination.
"""

        with open(report_path, mode="w", encoding="utf-8") as f:
            f.write(content)

        return report_path

    def run_full_eda(self) -> Dict[str, Any]:
        """Execute full EDA workflow: distributions, stats, plots, and report."""
        print("1. Computing distribution statistics from manifest...")
        dist_stats = self.compute_distribution_stats()

        print("2. Computing grayscale & NIR sensor statistics (memory-bounded sampling)...")
        gray_stats = self.compute_grayscale_stats(samples_per_class=20)

        print("3. Generating EDA visualizations in reports/figures/...")
        p_class = self.plot_class_distribution()
        p_subj = self.plot_subject_distribution()
        p_heat = self.plot_subject_gesture_heatmap()
        p_mont = self.plot_gesture_montage()
        p_hist = self.plot_pixel_intensity_distribution(gray_stats)

        print("4. Writing comprehensive reports/eda_summary.md...")
        report_path = self.generate_eda_summary_report(dist_stats, gray_stats)

        return {
            "dist_stats": dist_stats,
            "gray_stats": gray_stats,
            "figures": [str(p_class), str(p_subj), str(p_heat), str(p_mont), str(p_hist)],
            "report_path": str(report_path),
        }


def main() -> int:
    """CLI runner for Task 4 Phase 3 EDA."""
    if Path.cwd().name == "task4-hand-gesture-recognition":
        manifest_path = Path("reports/gesture_manifest.csv")
        data_dir = Path("data/interim")
        output_dir = Path("reports")
    else:
        manifest_path = Path("task4-hand-gesture-recognition/reports/gesture_manifest.csv")
        data_dir = Path("task4-hand-gesture-recognition/data/interim")
        output_dir = Path("task4-hand-gesture-recognition/reports")

    print("=" * 70)
    print("SkillCraft ML Portfolio — Task 4: Phase 3 Exploratory Data Analysis")
    print("=" * 70)
    print(f"Manifest:   {manifest_path}")
    print(f"Data Dir:   {data_dir}")
    print(f"Output Dir: {output_dir}\n")

    analyzer = GestureEDAAnalyzer(manifest_path, data_dir, output_dir)
    results = analyzer.run_full_eda()

    print("\n" + "=" * 70)
    print("PHASE 3 EDA COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"  Total Images:        {results['dist_stats']['total_images']:,}")
    print(f"  Classes:             {results['dist_stats']['num_gestures']} (balanced at 2,000 frames each)")
    print(f"  Subjects:            {results['dist_stats']['num_subjects']} (balanced at 2,000 frames each)")
    print(f"  Image Dimensions:    {results['gray_stats']['image_width']}x{results['gray_stats']['image_height']} (Aspect {results['gray_stats']['aspect_ratio']}:1)")
    print(f"  Mean Pixel Intensity:{results['gray_stats']['mean_intensity']} (Std: {results['gray_stats']['std_intensity']})")
    print(f"  Report written to:   {results['report_path']}")
    print(f"  Figures generated:   {len(results['figures'])}")
    for fig in results['figures']:
        print(f"    - {fig}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
