# Task 4: Hand Gesture Recognition — Archive Inspection & Dataset Audit

## 1. Executive Summary
- **Archive File**: `archive.zip`
- **File Location**: `C:\Users\HP\my new project (skillcraft)\SkillCraft-ML-Portfolio\task4-hand-gesture-recognition\archive.zip`
- **Archive Size**: **2,286,085,038 bytes** (**2.129 GiB**)
- **Uncompressed Payload**: **2,289,665,140 bytes** (**2.132 GiB**)
- **ZIP Format & Integrity**: Valid ZIP archive (`zipfile.is_zipfile = True`).
- **Full CRC32 Check (`testzip()`)**: **PASSED** (0 corrupt files across all entries, completed in 15.08s).
- **Suspicious Paths / Directory Traversal**: **0 suspicious paths** detected.
- **Image Format**: **100% PNG** (`.png`).
- **Image Dimensions & Mode**: $640 \times 240$ pixels, 8-bit Grayscale (`L` mode, near-infrared sensor frames).

---

## 2. Critical Structural Finding: Nested Duplicate Hierarchy

The archive contains **40,000 total ZIP entries**, but represents exactly **20,000 unique image frames**:

```text
archive.zip
├── leapGestRecog/               [20,000 entries] <-- CANONICAL DATASET
│   ├── 00/ ... 09/              (10 Subjects)
│   │   ├── 01_palm/ ... 10_down/ (10 Gestures, 200 frames each)
│   │   │   └── frame_XX_YY_ZZZZ.png
└── leapgestrecog/               [20,000 entries] <-- NESTED DUPLICATE
    └── leapGestRecog/
        ├── 00/ ... 09/
        │   ├── 01_palm/ ... 10_down/
        │   │   └── frame_XX_YY_ZZZZ.png
```

### Safety & Filesystem Collisions:
1. Every file in `leapGestRecog/` has an identical counterpart in `leapgestrecog/leapGestRecog/` with **100% matching CRC32 checksums and file sizes** (verified across all 20,000 pairs).
2. On Windows (case-insensitive NTFS), extracting both directories naively risks name collisions and duplicate extraction (consuming ~4.3 GB instead of 2.1 GB).
3. **Requirement for Phase 2 Extraction**: Filter archive members strictly by prefix `leapGestRecog/`, discarding the redundant `leapgestrecog/` hierarchy.

---

## 3. Dataset Classes & Subject Distribution

The dataset comprises **10 Subjects** performing **10 Distinct Hand Gestures**, with exactly **200 frames per gesture per subject**:

### Subject Distribution (Canonical `leapGestRecog/`):
| Subject ID | Total Frames | Gestures Recorded | Frames per Gesture | Description |
| :--- | :--- | :--- | :--- | :--- |
| `00` | 2,000 | 10 | 200 | Subject 0 |
| `01` | 2,000 | 10 | 200 | Subject 1 |
| `02` | 2,000 | 10 | 200 | Subject 2 |
| `03` | 2,000 | 10 | 200 | Subject 3 |
| `04` | 2,000 | 10 | 200 | Subject 4 |
| `05` | 2,000 | 10 | 200 | Subject 5 |
| `06` | 2,000 | 10 | 200 | Subject 6 |
| `07` | 2,000 | 10 | 200 | Subject 7 |
| `08` | 2,000 | 10 | 200 | Subject 8 |
| `09` | 2,000 | 10 | 200 | Subject 9 |
| **TOTAL** | **20,000** | **10** | **200** | **10 Subjects** |

### Gesture Categories (10 Classes):
| # | Gesture Label | Description | Total Frames | Balance |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `01_palm` | Open flat palm facing sensor | 2,000 | 10.0% |
| 2 | `02_l` | L-shape (thumb + index extended) | 2,000 | 10.0% |
| 3 | `03_fist` | Closed static fist | 2,000 | 10.0% |
| 4 | `04_fist_moved` | Dynamic moving fist gesture | 2,000 | 10.0% |
| 5 | `05_thumb` | Thumbs-up gesture | 2,000 | 10.0% |
| 6 | `06_index` | Index finger pointing | 2,000 | 10.0% |
| 7 | `07_ok` | OK sign (thumb + index ring) | 2,000 | 10.0% |
| 8 | `08_palm_moved` | Dynamic moving palm gesture | 2,000 | 10.0% |
| 9 | `09_c` | C-shape hand posture | 2,000 | 10.0% |
| 10 | `10_down` | Downward pointing gesture | 2,000 | 10.0% |

The dataset is **perfectly balanced** across all 10 classes (exactly 2,000 images per class).

---

## 4. Leakage Risk & Recommended Split Strategy

### ⚠️ Critical Leakage Consideration: Temporal & Subject Leakage
Each subject's gesture is recorded as a continuous video sequence of 200 frames. 
- **Random Frame Splitting (WRONG)**: Randomly shuffling frames across train, validation, and test splits would cause extreme temporal leakage (frame $t$ in train, frame $t+1$ in test). This yields artificially high metrics (~99.9%) that fail completely on unseen hands.
- **Subject-Independent Partitioning (CORRECT & RECOMMENDED)**:
  To assess genuine real-world generalization, the dataset must be partitioned **by subject identity** (Group K-Fold / Leave-Subjects-Out):
  
  | Partition | Subjects Assigned | Total Images | Proportion | Purpose |
  | :--- | :--- | :--- | :--- | :--- |
  | **Train Set** | `00`, `01`, `02`, `03`, `04`, `05`, `06` (7 Subjects) | 14,000 (or subsample) | 70% | Feature extraction & model fitting |
  | **Validation Set** | `07` (1 Subject) | 2,000 (or subsample) | 10% | Hyperparameter tuning & model selection |
  | **Held-Out Test Set** | `08`, `09` (2 Subjects) | 4,000 (or subsample) | 20% | Final unbiased unseen-subject generalization |

### Resource-Conscious Benchmark Sizing (for 8 GB RAM Laptop):
- Extracting all 20,000 frames at full $640 \times 240$ resolution requires ~2.1 GB of disk space.
- An optimal, resource-conscious strategy for fast training and experimentation is to downsample spatial dimensions to $128 \times 64$ or $64 \times 64$ and optionally construct a balanced benchmark (e.g., 4,000 or 5,000 frames, keeping temporal spacing of every 4th frame, or retaining full subject-wise groups).

---

## 5. Dependency & Environment Audit
Inspected `.venv` on Python 3.11.9:
- `numpy`: 2.4.6
- `scipy`: 1.17.1
- `scikit-learn`: 1.9.1
- `scikit-image`: 0.26.0
- `opencv-python` (`cv2`): 5.0.0
- `Pillow` (`PIL`): 12.3.0
- `matplotlib`: 3.10.8
- `pandas`: 3.0.0
- `joblib`: 1.6.0
- `streamlit`: 1.55.0
- `pytest`: 9.1.1
- *No deep learning packages (`torch`, `tensorflow`, `mediapipe`) are present or required. Classical computer vision (HOG / Contours / Moments + SVM / Random Forest) is supported natively.*
