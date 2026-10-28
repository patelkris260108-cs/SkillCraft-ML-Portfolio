# Task 4: Hand Gesture Recognition (Near-Infrared / Leap Motion Dataset)

## SkillCraft Technology — Machine Learning Internship

### 1. Project Overview
This project builds a multi-class hand gesture recognition pipeline capable of categorizing human hand poses from near-infrared sensor frames. Using the **LeapGestRecog (GTI-UPM)** dataset, the system implements an interpretable, resource-conscious computer vision and machine learning pipeline without heavy deep-learning dependencies, optimized for an 8 GB RAM laptop.

---

### 2. Dataset Overview & Discovery Findings
- **Source Archive**: `archive.zip` (~2.13 GiB, 2,286,085,038 bytes)
- **Archive Status**: Verified intact with standard library `testzip()` (0 corruptions, 0 path traversals).
- **Structure**:
  - Contains 10 subjects (`00` through `09`) performing 10 distinct gestures.
  - Exactly 200 consecutive frames per gesture per subject.
  - Image properties: $640 \times 240$ pixels, 8-bit Grayscale (`L` mode), PNG format.
- **Critical Archive Quirk**:
  - Contains **40,000 ZIP entries**, but represents **20,000 unique frames**.
  - A duplicate tree exists under `leapgestrecog/leapGestRecog/` with 100% matching CRC32 checksums.
  - Phase 2 extraction must filter strictly to canonical `leapGestRecog/` members.

---

### 3. Subject-Wise Generalization & Split Strategy
To avoid severe **temporal autocorrelation** and **identity leakage** between consecutive video frames, data must be split **by subject** rather than randomly shuffling individual frames:

| Partition | Subjects Assigned | Frames (Full) | Proportion | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Training Set** | `00`, `01`, `02`, `03`, `04`, `05`, `06` (7 Subjects) | 14,000 | 70% | Feature extraction and classifier training |
| **Validation Set** | `07` (1 Subject) | 2,000 | 10% | Model and hyperparameter selection |
| **Held-Out Test Set** | `08`, `09` (2 Subjects) | 4,000 | 20% | Final unbiased unseen-subject generalization |

---

### 4. Ten Gesture Classes
1. `01_palm`: Flat open palm facing sensor
2. `02_l`: L-shape posture (thumb + index extended)
3. `03_fist`: Closed static fist
4. `04_fist_moved`: Dynamic moving fist
5. `05_thumb`: Thumbs-up gesture
6. `06_index`: Pointing index finger
7. `07_ok`: Circular OK sign
8. `08_palm_moved`: Dynamic moving open palm
9. `09_c`: Curved C-shape hand
10. `10_down`: Downward pointing gesture

---

### 5. Directory Structure
```text
task4-hand-gesture-recognition/
├── archive.zip                      # User-supplied raw archive (2.13 GiB, git-ignored)
├── data/
│   ├── interim/                     # Extracted canonical gesture images (20,000 frames, git-ignored)
│   └── processed/                   # Precomputed feature arrays (.npy, git-ignored)
├── models/                          # Serialized classifier pipeline & config
│   ├── best_gesture_model.joblib    # Trained SGD Linear SVM pipeline (git-ignored)
│   └── model_config.json            # Model classes & HOG configuration (tracked)
├── reports/                         # Manifests, EDA reports, and metrics
│   ├── gesture_manifest.csv         # 20,000-frame subject-partitioned manifest
│   ├── eda_summary.md               # EDA statistical report
│   ├── modeling_metrics.json        # Machine-readable evaluation metrics
│   ├── modeling_report.md           # HOG + SVM modeling report
│   └── figures/                     # EDA and evaluation figures (7 PNGs)
├── src/                             # Modular source code
│   ├── __init__.py
│   ├── dataset_inspector.py         # Archive inspection & safety validator
│   ├── dataset_extractor.py         # Canonical streaming extraction & manifest generator
│   ├── eda_analyzer.py              # Bounded-memory EDA & figure generator
│   ├── feature_extractor.py         # 2,700D HOG feature extraction
│   └── model_trainer.py             # Subject-independent model training & evaluation
├── tests/                           # Unit and integration test suite (49 passing tests)
│   ├── __init__.py
│   ├── test_dataset_inspector.py    # Path safety and mock archive tests
│   ├── test_dataset_extractor.py    # Extraction, filter & manifest tests
│   ├── test_eda_analyzer.py         # Distribution analysis & plot tests
│   ├── test_feature_extractor.py    # Feature extraction & HOG dimension tests
│   ├── test_model_trainer.py        # Model training & serialization tests
│   └── test_app.py                  # Streamlit frontend & inference tests
├── .gitignore                       # Task 4 specific exclusion rules
├── app.py                           # Interactive Streamlit Web Application
├── DATASET_INSPECTION.md            # Detailed archive inspection report
└── README.md                        # Project documentation
```

---

### 6. Development Roadmap
- [x] **Phase 1: Dataset Inspection, Safety Validation & Strategy Formulation** *(Completed)*:
  - Non-extracting CRC integrity verification (`testzip()` passed with 0 corruptions).
  - Discovery of nested duplicate directory (`leapgestrecog/` vs `leapGestRecog/`).
  - Analysis of 10 subjects, 10 gestures, and $640 \times 240$ grayscale PNG specifications.
  - Identification of temporal/subject leakage risk and formulation of subject-independent split.
  - Unit test suite (5 passing tests).
- [x] **Phase 2: Safe Extraction & Benchmark Manifest Generation** *(Completed)*:
  - Memory-bounded streaming extraction of canonical `leapGestRecog/` files only (20,000 images, 1.066 GB).
  - Strict exclusion of 20,000 duplicate hierarchy entries under `leapgestrecog/leapGestRecog/`.
  - Zero image corruption, verified with Pillow headers on write.
  - Generated deterministic `reports/gesture_manifest.csv` with strict subject-wise partitioning:
    - Train: 14,000 images (70%, subjects 00–06)
    - Validation: 2,000 images (10%, subject 07)
    - Test: 4,000 images (20%, subjects 08–09)
  - Unit and integration test suite (20 passing tests).
- [x] **Phase 3: Exploratory Data Analysis (EDA)** *(Completed)*:
  - Bounded-memory statistical analysis of gesture classes, subjects, and split partitions.
  - Near-Infrared (NIR) grayscale property profile ($640 \times 240$, 2.67:1 aspect ratio, bimodal intensity).
  - Generated 5 high-resolution figures in `reports/figures/`:
    - `class_distribution.png`: 10 balanced classes (2,000 frames each).
    - `subject_distribution.png`: 10 balanced subjects colored by split (Train: 00-06, Val: 07, Test: 08-09).
    - `subject_gesture_heatmap.png`: $10 \times 10$ orthogonal balance matrix (200 frames per cell).
    - `gesture_montage.png`: Visual $2 \times 5$ montage across all 10 hand gestures.
    - `pixel_intensity_distribution.png`: NIR bimodal intensity profile (dark background vs. reflective hand signal).
  - Formatted comprehensive `reports/eda_summary.md` with modeling recommendations.
  - Complete test suite (26 passing tests).
- [x] **Phase 4: Feature Extraction & Multi-Class Classification Pipeline** *(Completed)*:
  - Preserved 2.67:1 aspect ratio by resizing to 160x60 grayscale (scikit-image HOG).
  - Extracted 2,700 features per frame sequentially (14,000 train, 2,000 val, 4,000 test) into `data/processed/`.
  - Trained and compared 5 lightweight candidate pipelines strictly on training data:
    - `Dummy_Baseline`: Val Macro F1 = 1.82% (10.0% accuracy)
    - `Linear_SVM_C1.0`: Val Macro F1 = 70.67% (72.50% accuracy)
    - `Linear_SVM_C0.1`: Val Macro F1 = 70.79% (72.75% accuracy)
    - `SGD_LogisticRegression`: Val Macro F1 = 79.54% (80.90% accuracy)
    - `SGD_LinearSVM`: **Val Macro F1 = 79.75% (79.95% accuracy) [SELECTED WINNER]**
  - Evaluated winning model on completely unseen Held-Out Test Subjects (`08` and `09`):
    - **Held-Out Test Accuracy: 62.20%**
    - **Held-Out Test Macro F1: 61.25%**
    - **Held-Out Test Precision: 65.20%**
    - **Held-Out Test Recall: 62.20%**
  - Generated `reports/figures/model_comparison.png`, `reports/figures/test_confusion_matrix.png`, `reports/modeling_metrics.json`, and `reports/modeling_report.md`.
  - Serialized best model to `models/best_gesture_model.joblib` and `models/model_config.json`.
  - Complete test suite (40 passing tests).
- [x] **Phase 5: Interactive Streamlit Web Application** *(Completed)*:
  - Interactive multi-tab web application (`app.py`) powered by Streamlit and Pillow.
  - Image upload support (PNG, JPG, JPEG, up to 10 MB) with validation and corruption protection.
  - Preloaded benchmark gallery for instant evaluation against held-out test frames (Subject `08`).
  - Native 2.67:1 aspect ratio normalization ($160 \times 60$) and live 2D HOG gradient orientation map rendering.
  - Full class decision margins visualization and explicit callouts on uncalibrated SVM margins and static 2D frame limitations.
  - Complete test suite (49 passing tests).

---

### 7. Execution, Model Reproduction & Launch Instructions

All commands run using the repository root `.venv` in Windows PowerShell.

#### A. Reproducing the Model from Scratch (Fresh Clone Workflow)

The raw archive (`archive.zip`, ~2.13 GiB), interim images (`data/interim/`, 1.06 GB), processed feature matrices (`data/processed/*.npy`), and trained binary model (`models/best_gesture_model.joblib`, 170 KB) are intentionally excluded from Git version control to comply with repository size limits. To reproduce the model deterministically on a fresh clone:

1. **Place Raw Archive**:
   Download `archive.zip` (~2.13 GiB) from [Kaggle LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) following [`DATASET_INSPECTION.md`](DATASET_INSPECTION.md) and place it directly at:
   ```text
   task4-hand-gesture-recognition/archive.zip
   ```

2. **Execute Ingestion, Extraction & Training Pipeline**:
   ```powershell
   cd "C:\Users\HP\my new project (skillcraft)\SkillCraft-ML-Portfolio\task4-hand-gesture-recognition"

   # Step 1: Run non-extracting archive discovery & CRC32 integrity check
   & "..\.venv\Scripts\python.exe" src/dataset_inspector.py

   # Step 2: Stream-extract canonical leapGestRecog/ frames (skipping duplicates) and write gesture_manifest.csv
   & "..\.venv\Scripts\python.exe" src/dataset_extractor.py

   # Step 3: Run EDA distribution analysis & generate high-resolution figures
   & "..\.venv\Scripts\python.exe" src/eda_analyzer.py

   # Step 4: Extract 2,700-dimensional HOG features into data/processed/ (~107s)
   & "..\.venv\Scripts\python.exe" src/feature_extractor.py

   # Step 5: Train SGD Linear SVM & serialize models/best_gesture_model.joblib and models/model_config.json (~45s)
   & "..\.venv\Scripts\python.exe" src/model_trainer.py
   ```

#### B. Running the Pytest Test Suite
```powershell
cd "C:\Users\HP\my new project (skillcraft)\SkillCraft-ML-Portfolio\task4-hand-gesture-recognition"
& "..\.venv\Scripts\python.exe" -m pytest tests -v
```
*(All 49 unit and integration tests passing in ~20.5s)*

#### C. Launching the Interactive Web Application
From repository root:
```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run task4-hand-gesture-recognition/app.py --server.port 8503
```
Or from `task4-hand-gesture-recognition` directory:
```powershell
cd "C:\Users\HP\my new project (skillcraft)\SkillCraft-ML-Portfolio\task4-hand-gesture-recognition"
& "..\.venv\Scripts\python.exe" -m streamlit run app.py --server.port 8503
```
Then navigate to `http://localhost:8503` in your web browser.

---

### 9. Technical Limitations & Practical Considerations

1. **Participant Population Size ($N=10$):**
   - While the dataset provides 20,000 frames, they originate from 10 distinct human subjects.
   - The subject-independent split isolates 7 subjects for training, 1 for validation, and 2 for held-out test ($N=4,000$). Performance on the test set reflects genuine inter-individual generalization, where subject anatomy, hand size, and articulation habits create natural variance.
2. **Static 2D Spatial Ambiguity on Dynamic Gestures:**
   - Single-frame 2D representations share identical contours between stationary and moving pairs (e.g. `01_palm` vs. `08_palm_moved`, and `03_fist` vs. `04_fist_moved`).
   - Disambiguating dynamic gestures from single frames is an inherent limitation of spatial-only feature extractors (HOG); sequence modeling or optical flow is recommended for production video streams.
3. **Sensor Modality & Lighting Domain:**
   - The pipeline is engineered for Leap Motion Near-Infrared (NIR) 8-bit grayscale imagery exhibiting distinct skin reflectance over dark backgrounds.
   - Standard ambient RGB photos with complex background clutter or uncontrolled illumination may require background segmentation or domain adaptation.
4. **Decision Margins vs. Probabilities:**
   - The selected linear SVM uses hinge loss (`loss='hinge'`). Output scores represent signed geometric distances from hyperplanes in 2,700-dimensional feature space, not Bayesian probabilities.
