# Task 3: Dogs vs. Cats Benchmark — Exploratory Data Analysis (EDA)

## Executive Summary
- **Total Benchmark Images**: 4,000
- **Corrupt Images**: 0
- **Color Mode**: 100% RGB (3-channel color)
- **Image Format**: 100% JPEG
- **Class Distribution**: 2,000 Cats (50.0%) vs. 2,000 Dogs (50.0%) — perfectly balanced

---

## 1. Dataset Partition & Class Balance

| Split | Cats | Dogs | Total | Proportion |
| :--- | :--- | :--- | :--- | :--- |
| **TRAIN** | 1,400 | 1,400 | 2,800 | 70.0% |
| **VAL** | 300 | 300 | 600 | 15.0% |
| **TEST** | 300 | 300 | 600 | 15.0% |
| **TOTAL** | **2,000** | **2,000** | **4,000** | **100.0%** |

---

## 2. Image Dimensions & Resolution Statistics

Image dimensions vary considerably across photographs in both classes, highlighting the necessity of an aspect-preserving resize pipeline before HOG feature extraction.

| Metric | Dimension | Mean ± Std | Median | Min | Max | IQR (Q25 – Q75) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cat** | Width | 409.6 ± 108.5 px | 468 px | 55 px | 500 px | 335 – 499 px |
| | Height | 355.2 ± 96.5 px | 374 px | 45 px | 500 px | 299 – 400 px |
| | Aspect Ratio | 1.19 ± 0.29 | 1.33 | 0.37 | 2.99 | 0.99 – 1.34 |
| | File Size (KB) | 20.4 ± 8.9 KB | 20.4 KB | 1.3 KB | 67.8 KB | 14.1 – 25.9 KB |
| **Dog** | Width | 397.5 ± 111.0 px | 427 px | 60 px | 500 px | 318 – 499 px |
| | Height | 363.3 ± 97.0 px | 374 px | 50 px | 500 px | 309 – 426 px |
| | Aspect Ratio | 1.13 ± 0.29 | 1.20 | 0.44 | 2.21 | 0.89 – 1.34 |
| | File Size (KB) | 24.1 ± 11.7 KB | 22.9 KB | 1.4 KB | 76.6 KB | 16.0 – 30.7 KB |
| **Overall Benchmark** | Width | 403.6 ± 109.9 px | 448 px | 55 px | 500 px | 320 – 499 px |
| | Height | 359.2 ± 96.9 px | 374 px | 45 px | 500 px | 301 – 415 px |
| | Aspect Ratio | 1.16 ± 0.29 | 1.30 | 0.37 | 2.99 | 0.94 – 1.34 |
| | File Size (KB) | 22.2 ± 10.6 KB | 21.5 KB | 1.3 KB | 76.6 KB | 14.9 – 28.0 KB |

---

## 3. Orientation & Color Mode Distribution

| Orientation | Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| **Landscape** | 2,581 | 64.5% | Width > Height * 1.05 |
| **Portrait** | 1,030 | 25.8% | Height > Width * 1.05 |
| **Square** | 389 | 9.7% | Approx. 1:1 ratio |

### Color Modes and File Formats:
- **Color Mode `RGB`**: 4,000 images (100.0%)
- **Format `JPEG`**: 4,000 images (100.0%)

---

## 4. Key Engineering Insights for HOG + SVM Feature Pipeline

1. **Uniform Normalization Target (e.g. 64x64 or 128x128)**:
   - Raw images span from 40 px up to 1,050 px in dimensions, with median resolutions near 400x375 px.
   - Because HOG requires fixed-dimension input grids to generate uniform feature vectors, images must be resized prior to descriptor computation.
   - Resizing to $64 \times 64$ produces 1,764 features per image (L2-Hys normalized), yielding a very lightweight 4,000-image dataset matrix of ~28 MB in memory.

2. **Color Mode Handling**:
   - All 4,000 images are standard 3-channel RGB JPEG files. Grayscale conversion (`cv2.cvtColor` or PIL `.convert('L')`) is appropriate and uniform across all samples.

3. **Zero Corruptions Found**:
   - All 4,000 extracted benchmark images successfully passed Pillow verification (`img.verify()`).

4. **Class Balance Integrity**:
   - The dataset contains exactly 2,000 cats and 2,000 dogs with identical class balance across train (1,400:1,400), validation (300:300), and test (300:300) splits.

---

## 5. Visual Artifacts Generated
- `reports/figures/dimension_distribution.png`: Width, height, and resolution distributions.
- `reports/figures/aspect_ratio_distribution.png`: Aspect ratio histograms and orientation breakdowns.
- `reports/figures/class_samples_montage.png`: Representative visual sample montage of cats and dogs.
