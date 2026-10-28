# Task 3: Dogs vs. Cats Image Classification (HOG + Support Vector Machine)

## SkillCraft Technology — Machine Learning Internship

### 1. Project Overview
This project develops an interpretable, resource-conscious binary image classification pipeline to categorize photographs of dogs and cats. Rather than relying on heavyweight deep-learning architectures, this project implements a classical computer vision pipeline utilizing **Histogram of Oriented Gradients (HOG)** feature descriptors coupled with **Support Vector Classifiers (Linear and Kernel SVM)**.

---

### 2. Dataset Setup & Acquisition
- **Competition**: [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
- **Designated Archive Location**: `data/raw/train.zip`
- **Acquisition Protocol**: Manual download by the user. Detailed step-by-step instructions are documented in [`DATASET_SETUP.md`](DATASET_SETUP.md).
- **Integrity & Security**:
  - No Kaggle API credentials, tokens, or passwords are required or stored.
  - The raw `train.zip` archive is treated as strictly read-only and preserved untouched.
  - Ingestion and validation are performed in a bounded-memory streaming manner to accommodate machines with constrained physical RAM.

---

### 3. Benchmark Dataset Strategy (4,000-Image Subset)
While the full Kaggle training set contains 25,000 images, Task 3 establishes a balanced, high-efficiency **4,000-image benchmark** for rapid experimentation, hyperparameter tuning, and leak-free evaluation:

| Partition | Total Images | Cats | Dogs | Proportion | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Training Set** | 2,800 | 1,400 | 1,400 | 70% | Fit `StandardScaler` and train SVM classifiers |
| **Validation Set** | 600 | 300 | 300 | 15% | Hyperparameter tuning ($C$, kernel, block size) |
| **Test Set** | 600 | 300 | 300 | 15% | Final unbiased performance assessment (held-out) |
| **Total Benchmark** | **4,000** | **2,000** | **2,000** | **100%** | Balanced evaluation benchmark |

- **Seed**: `random_state=42`
- **Leakage Prevention**: Partitions are mutually exclusive; scaler and feature pipelines are fitted strictly on the training partition.

---

### 4. Technical Architecture & Memory-Conscious Design

```
Raw Image (ZIP / Disk)
       │
       ▼ (Single-image stream; no bulk array in RAM)
Aspect-Preserving Resize (64x64 or 128x128)
       │
       ▼
Grayscale Conversion (cv2.cvtColor)
       │
       ▼
HOG Feature Extraction (skimage.feature.hog)
  - Orientations: 9 gradient bins (0°–180°)
  - Pixels per cell: (8, 8)
  - Cells per block: (2, 2)
  - Normalization: L2-Hys (L2-Hysteresis)
       │
       ▼ (Feature vector size: 1,764 floats ≈ 7 KB / image; 4,000 images ≈ 28 MB RAM)
StandardScaler (fitted on train split only)
       │
       ▼
Support Vector Machine (LinearSVC vs. Kernel SVC vs. SGDClassifier)
       │
       ▼
Metrics & Reports (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
```

---

### 5. Directory Structure
```text
task3-dogs-vs-cats/
├── data/
│   ├── raw/                         # dogs-vs-cats.zip, train.zip (git-ignored)
│   ├── interim/                     # Extracted 4,000-image benchmark (git-ignored)
│   │   ├── train/ (cat/, dog/)      # 1,400 cats, 1,400 dogs
│   │   ├── val/ (cat/, dog/)        # 300 cats, 300 dogs
│   │   └── test/ (cat/, dog/)       # 300 cats, 300 dogs
│   └── processed/                   # Precomputed HOG feature matrices (.npy, git-ignored)
│       ├── X_train.npy, y_train.npy # (2800, 1764)
│       ├── X_val.npy, y_val.npy     # (600, 1764)
│       └── X_test.npy, y_test.npy   # (600, 1764)
├── models/                          # Serialized scaler and SVM pipelines (.joblib)
│   ├── best_svm_pipeline.joblib     # Fitted StandardScaler + RBF Kernel SVM (C=1.0)
│   └── hog_config.json              # HOG parameters (1,764 features)
├── reports/
│   ├── dataset_validation_report.md # Summary audit from dataset_validator.py
│   ├── benchmark_manifest.csv       # 4,000-image split manifest
│   ├── eda_summary.md               # Exploratory Data Analysis report
│   ├── modeling_metrics.json        # Machine-readable evaluation metrics
│   ├── modeling_report.md           # HOG + SVM modeling report
│   └── figures/                     # EDA and modeling figures
│       ├── dimension_distribution.png
│       ├── aspect_ratio_distribution.png
│       ├── class_samples_montage.png
│       ├── model_comparison.png
│       ├── confusion_matrix.png
│       └── roc_curve.png
├── src/                             # Core modular source code
│   ├── __init__.py
│   ├── dataset_validator.py         # Archive inspection & split planner
│   ├── extractor.py                 # Streaming benchmark extractor & verifier
│   ├── eda.py                       # Streaming image metadata & distribution analysis
│   ├── feature_extractor.py         # Memory-bounded HOG descriptor extractor
│   └── model_trainer.py             # SVM training, validation selection & evaluation
├── tests/                           # Pytest unit tests (50 passing tests)
│   ├── __init__.py
│   ├── test_dataset_validator.py    # Archive, corruption, and split tests
│   ├── test_extractor.py            # Streaming extraction & zip-slip protection tests
│   ├── test_eda.py                  # Metadata extraction & figure generation tests
│   ├── test_feature_extractor.py    # HOG dimension, determinism, and partition tests
│   ├── test_model_trainer.py        # Model fitting, evaluation, and serialization tests
│   └── test_app.py                  # Streamlit loading, validation, and inference tests
├── .gitignore                       # Task 3 specific exclusion rules
├── app.py                           # Interactive Streamlit Web Application
├── DATASET_SETUP.md                 # Manual dataset acquisition guide
└── README.md                        # Project documentation
```

---

### 6. Development Roadmap
- [x] **Phase 1: Safe Setup, Dataset Validator & Benchmark Preparation** *(Completed)*:
  - Branch creation (`task3-dogs-vs-cats`) from verified `master` baseline.
  - Complete project directory scaffold and Task 3 `.gitignore`.
  - Step-by-step manual dataset acquisition instructions (`DATASET_SETUP.md`).
  - Memory-bounded `DatasetValidator` utility and 4,000-image split planner.
  - Unit test suite (13 passing tests).
- [x] **Phase 2: Dataset Verification & Benchmark Manifest Generation** *(Completed)*:
  - Validated outer archive `dogs-vs-cats.zip` (812.13 MiB) and inner `train.zip` (543.16 MiB).
  - Validated all 25,000 training images (12,500 cats, 12,500 dogs) with 0 corruptions.
  - Generated reproducible 4,000-image manifest (`benchmark_manifest.csv`, seed=42).
- [x] **Phase 3: Benchmark Extraction & Exploratory Data Analysis (EDA)** *(Completed)*:
  - Safe, streaming extraction of strictly the 4,000 benchmark images into `data/interim/`.
  - Verification of 100% of extracted images via Pillow with 0 corruptions and exact balance.
  - Lightweight EDA analyzing resolutions, aspect ratios, orientations, and color modes.
  - Generated publication-ready figures (`reports/figures/`) and `reports/eda_summary.md`.
  - Unit test suite (29 passing tests).
- [x] **Phase 4: HOG Preprocessing, Feature Extraction & SVM Training** *(Completed)*:
  - Standardized HOG feature extraction pipeline (64x64 grayscale, 9 orientations, 8x8 cell, 2x2 block = 1,764 features).
  - Memory-bounded caching in `data/processed/` (~28 MB RAM footprint).
  - Model selection on validation split: Baseline Dummy (50.0%), LinearSVM (66.7%), RBF Kernel SVM (**74.7%**).
  - Final held-out test evaluation on selected RBF SVM: **72.5% Accuracy**, **72.5% Macro F1**, **0.7856 ROC-AUC**.
  - Generated evaluation figures (`model_comparison.png`, `confusion_matrix.png`, `roc_curve.png`).
  - Comprehensive unit test suite with 41 passing tests.
- [x] **Phase 5: Interactive Streamlit Web Application** *(Completed)*:
  - Interactive multi-tab UI for image upload (JPG, JPEG, PNG) and benchmark test sample selection.
  - Exact preprocessing parity with model pipeline (64x64 grayscale bilinear resize).
  - Real-time HOG gradient orientation visualizer (Inferno colormap overlay).
  - Signed SVM decision margin gauge and non-probabilistic margin explanation.
  - Full test suite expanded to 50 passing tests.
- [ ] **Phase 6: Final Review & Submission Documentation**.

---

### 7. Execution, Model Reproduction & Launch Instructions

All commands run using the repository root `.venv` in Windows PowerShell.

#### A. Reproducing the Model from Scratch (Fresh Clone Workflow)

The trained binary model (`models/best_svm_pipeline.joblib`, ~32.85 MB) is intentionally excluded from Git version control to prevent repository bloat. To reproduce the exact model and artifacts on a fresh clone:

1. **Place Raw Archive**:
   Download `train.zip` (~543 MB) from [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) following [`DATASET_SETUP.md`](DATASET_SETUP.md) and place it at:
   ```text
   task3-dogs-vs-cats/data/raw/train.zip
   ```

2. **Execute Data Pipeline & Model Training**:
   ```powershell
   cd "C:\Users\HP\my new project (skillcraft)\SkillCraft-ML-Portfolio\task3-dogs-vs-cats"

   # Step 1: Inspect archive integrity and plan 4,000-image balanced benchmark
   & "..\.venv\Scripts\python.exe" src/dataset_validator.py

   # Step 2: Safely extract benchmark images into data/interim/ (1,400 train, 300 val, 300 test per class)
   & "..\.venv\Scripts\python.exe" src/extractor.py

   # Step 3: Extract 1,764-dimensional HOG features into data/processed/ (~28 MB RAM, ~1.5 min)
   & "..\.venv\Scripts\python.exe" src/feature_extractor.py

   # Step 4: Train RBF Kernel SVM & serialize models/best_svm_pipeline.joblib and models/hog_config.json
   & "..\.venv\Scripts\python.exe" src/model_trainer.py
   ```

#### B. Running the Pytest Test Suite
```powershell
cd "C:\Users\HP\my new project (skillcraft)\SkillCraft-ML-Portfolio\task3-dogs-vs-cats"
& "..\.venv\Scripts\python.exe" -m pytest tests -v
```
*(All 50 unit and integration tests passing)*

#### C. Launching the Streamlit Web Application
From repository root:
```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run task3-dogs-vs-cats/app.py --server.port 8502
```
Or from `task3-dogs-vs-cats` directory:
```powershell
cd "C:\Users\HP\my new project (skillcraft)\SkillCraft-ML-Portfolio\task3-dogs-vs-cats"
& "..\.venv\Scripts\python.exe" -m streamlit run app.py --server.port 8502
```
Then navigate to `http://localhost:8502` in your web browser.
