# SkillCraft ML Portfolio

Comprehensive Machine Learning internship portfolio for **SkillCraft Technology**, spanning four distinct machine learning tracks: supervised regression, unsupervised clustering, binary computer vision classification, and multi-class near-infrared gesture recognition.

---

## 1. Portfolio Overview

| Task | Domain | Implemented Algorithm | Dataset | Verified Status | Test Suite |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **[Task 1](task1-house-price/)** | Supervised Regression | Linear Regression & Ridge | [Kaggle Ames Housing](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) | **Complete** | 98 / 98 Passed |
| **[Task 2](task2-customer-segmentation/)** | Unsupervised Clustering | K-Means ($k=5$ Classic & Extended) | [Kaggle Mall Customers](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) | **Complete** | 59 / 59 Passed |
| **[Task 3](task3-dogs-vs-cats/)** | Binary Image Classification | HOG + RBF Kernel SVM | [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) (4,000 Subset) | **Complete** | 50 / 50 Passed |
| **[Task 4](task4-hand-gesture-recognition/)** | Multi-Class Image Classification | HOG (2,700D) + Linear SVM (`SGDClassifier`) | [Kaggle LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) (20,000 NIR) | **Complete** | 49 / 49 Passed |

**Total Portfolio Tests:** **256 / 256 Passed (100%)**

---

## 2. Environment Setup

The portfolio is structured as a monorepo sharing a single lightweight virtual environment (`.venv`) based on Python 3.11:

```powershell
# 1. Clone repository and navigate to root
cd SkillCraft-ML-Portfolio

# 2. Create Python 3.11 virtual environment
py -3.11 -m venv .venv

# 3. Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 4. Install dependencies
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

---

## 3. Running Automated Test Suites

Each task contains an independent, comprehensive pytest test suite covering data loaders, leak-free preprocessing, model serialization, evaluation metrics, and interactive web application workflows:

```powershell
# Run Task 1 tests (98 passed in ~14s)
& ".\.venv\Scripts\python.exe" -m pytest task1-house-price/tests -v

# Run Task 2 tests (59 passed in ~7s)
& ".\.venv\Scripts\python.exe" -m pytest task2-customer-segmentation/tests -v -o pythonpath=task2-customer-segmentation

# Run Task 3 tests (50 passed in ~12s)
& ".\.venv\Scripts\python.exe" -m pytest task3-dogs-vs-cats/tests -v

# Run Task 4 tests (49 passed in ~30s)
& ".\.venv\Scripts\python.exe" -m pytest task4-hand-gesture-recognition/tests -v

# Run ALL 256 portfolio tests sequentially
& ".\.venv\Scripts\python.exe" -m pytest task1-house-price/tests task2-customer-segmentation/tests task3-dogs-vs-cats/tests task4-hand-gesture-recognition/tests -v -o pythonpath=task2-customer-segmentation
```

---

## 4. Interactive Streamlit Web Applications & Port Configuration

Each task features a dedicated, production-grade Streamlit web interface. Because Streamlit applications bind to local TCP ports, two applications cannot bind to the same port simultaneously. Each application is assigned a unique port below so that all four can run concurrently:

| Task | Web Application | Assigned Port | Launch Command (from Repository Root) | Local Access URL |
| :--- | :--- | :---: | :--- | :--- |
| **Task 1** | EstateIQ Property Valuation | **`8501`** | `& ".\.venv\Scripts\python.exe" -m streamlit run task1-house-price/app.py --server.port 8501` | `http://localhost:8501` |
| **Task 2** | Customer Persona Explorer | **`8504`** | `& ".\.venv\Scripts\python.exe" -m streamlit run task2-customer-segmentation/app.py --server.port 8504` | `http://localhost:8504` |
| **Task 3** | Dogs vs. Cats Classifier | **`8502`** | `& ".\.venv\Scripts\python.exe" -m streamlit run task3-dogs-vs-cats/app.py --server.port 8502` | `http://localhost:8502` |
| **Task 4** | Hand Gesture Recognition AI | **`8503`** | `& ".\.venv\Scripts\python.exe" -m streamlit run task4-hand-gesture-recognition/app.py --server.port 8503` | `http://localhost:8503` |

*Note on Port Conflicts:* If port `8501` is already in use by Task 1, Task 2 will fail to bind if launched without a port override. Launching Task 2 with `--server.port 8504` prevents port conflicts.

---

## 5. Task Summaries & Key Evaluation Benchmarks

### Task 1: House Price Prediction (`task1-house-price/`)
- **Objective:** Predict residential sale prices in Ames, Iowa using linear regression.
- **Leakage Prevention:** `ColumnTransformer` with numeric median imputation/scaling and categorical one-hot encoding fitted strictly on training folds.
- **Key Metrics (Held-Out Validation Set, 292 samples):**
  - **Baseline (3 Features):** $\text{Val RMSE} = \$52,976$, $\text{Val } R^2 = 0.6341$, $\text{Val RMSLE} = 0.2724$
  - **All-Features Linear Regression:** $\text{Val RMSE} = \$28,204$, $\text{Val } R^2 = 0.8963$, $\text{Val RMSLE} = 0.1799$
  - **5-Fold Cross-Validation (Ridge $\alpha=10$):** $\text{CV RMSE} = \$31,115 \pm 7,486$, $\text{CV } R^2 = 0.8326 \pm 0.0695$
- **Documentation:** [`task1-house-price/README.md`](task1-house-price/README.md) and [`docs/task1-report.md`](docs/task1-report.md)

### Task 2: Mall Customer Segmentation (`task2-customer-segmentation/`)
- **Objective:** Segment retail shoppers into actionable behavioral clusters using K-Means.
- **Methodology:** Strictly excludes arbitrary `CustomerID`; scales continuous features to prevent magnitude dominance; compares Classic 2D vs. Extended 4D clustering.
- **Key Metrics ($k=5$):**
  - **Classic 2D (Income + Spending):** $\text{Silhouette Score} = \mathbf{0.5547}$, $\text{Davies-Bouldin} = \mathbf{0.5722}$ (clean geometric quadrants)
  - **Extended 4D (Age + Income + Spending + Gender):** $\text{Silhouette Score} = 0.3498$, $\text{Davies-Bouldin} = 1.0245$ (demographic personas)
  - **Model Stability:** Adjusted Rand Index $\ge 0.98$ across multiple random initializations.
- **Documentation:** [`task2-customer-segmentation/README.md`](task2-customer-segmentation/README.md)

### Task 3: Dogs vs. Cats Image Classification (`task3-dogs-vs-cats/`)
- **Objective:** Classical computer vision binary classification without heavy deep-learning dependencies.
- **Architecture:** 64x64 grayscale conversion $\rightarrow$ 1,764-dimensional HOG feature descriptor $\rightarrow$ `StandardScaler` $\rightarrow$ RBF Kernel SVM ($C=1.0$).
- **Benchmark Strategy:** 4,000-image balanced subset (2,800 train, 600 validation, 600 held-out test).
- **Key Metrics (Held-Out Test Set, 600 images):**
  - **Test Accuracy:** $\mathbf{72.50\%}$
  - **Macro F1 Score:** $\mathbf{72.50\%}$
  - **Multiclass ROC-AUC:** $\mathbf{0.7856}$
  - **Validation Accuracy:** $74.67\%$ (Baseline Guessing: $50.0\%$)
- **Documentation:** [`task3-dogs-vs-cats/README.md`](task3-dogs-vs-cats/README.md) and [`task3-dogs-vs-cats/DATASET_SETUP.md`](task3-dogs-vs-cats/DATASET_SETUP.md)

### Task 4: Hand Gesture Recognition (`task4-hand-gesture-recognition/`)
- **Objective:** Multi-class classification across 10 hand gestures from Leap Motion near-infrared (NIR) frames.
- **Architecture:** 160x60 grayscale preserving native 2.67:1 aspect ratio $\rightarrow$ 2,700-dimensional HOG features $\rightarrow$ Linear SVM (`SGDClassifier`, hinge loss).
- **Evaluation Paradigm:** Strict **Subject-Independent Partitioning** with zero identity overlap (Train: Subjects `00`–`06`, 14,000 frames; Val: Subject `07`, 2,000 frames; Test: Subjects `08`–`09`, 4,000 frames).
- **Key Metrics:**
  - **Validation Macro F1 (Subject 07):** $\mathbf{79.75\%}$ (Accuracy: $79.95\%$)
  - **Held-Out Test Accuracy (Subjects 08 & 09):** $\mathbf{62.20\%}$
  - **Held-Out Test Macro F1:** $\mathbf{61.25\%}$
  - **Baseline Guessing:** $10.00\%$
- **Documentation:** [`task4-hand-gesture-recognition/README.md`](task4-hand-gesture-recognition/README.md) and [`task4-hand-gesture-recognition/DATASET_INSPECTION.md`](task4-hand-gesture-recognition/DATASET_INSPECTION.md)

---

## 6. Repository Structure

```text
SkillCraft-ML-Portfolio/
├── .gitignore                                 # Root Git hygiene rules (excludes datasets, venv, caches)
├── AGENTS.md                                  # Coding standards, engineering rules & audit protocols
├── PROJECT_STATUS.md                          # Comprehensive chronological implementation log
├── README.md                                  # Portfolio master documentation (this file)
├── requirements.txt                           # Unified Python dependencies
│
├── task1-house-price/                         # Task 1: Supervised Regression
│   ├── app.py                                 # EstateIQ Streamlit web application
│   ├── src/                                   # Data loader, preprocessors, training, predictors
│   ├── tests/                                 # Unit & integration test suite (98 tests)
│   ├── reports/                               # Baseline metrics, model comparison JSON, figures
│   └── README.md                              # Task 1 specific documentation
│
├── task2-customer-segmentation/               # Task 2: Unsupervised Clustering
│   ├── app.py                                 # Customer Segmentation Streamlit dashboard
│   ├── src/                                   # Data loader, preprocessor, clustering, evaluation
│   ├── tests/                                 # Complete test suite (59 tests)
│   ├── reports/                               # Cluster evaluation report, EDA summary, figures
│   └── README.md                              # Task 2 specific documentation
│
├── task3-dogs-vs-cats/                        # Task 3: Binary Computer Vision
│   ├── app.py                                 # Dogs vs. Cats Streamlit web application
│   ├── src/                                   # Dataset validator, streaming extractor, HOG, trainer
│   ├── tests/                                 # Test suite (50 tests)
│   ├── reports/                               # Benchmark manifest, modeling metrics JSON, figures
│   ├── DATASET_SETUP.md                       # Manual dataset acquisition guide
│   └── README.md                              # Task 3 specific documentation
│
├── task4-hand-gesture-recognition/            # Task 4: Multi-Class Gesture Recognition
│   ├── app.py                                 # Near-Infrared Gesture Classifier Streamlit app
│   ├── src/                                   # Archive inspector, streaming extractor, HOG, trainer
│   ├── tests/                                 # Test suite (49 tests)
│   ├── reports/                               # Gesture manifest (20,000 frames), metrics, figures
│   ├── DATASET_INSPECTION.md                  # Non-extracting archive discovery & CRC audit
│   └── README.md                              # Task 4 specific documentation
│
└── docs/
    └── task1-report.md                        # Task 1 comprehensive modeling report
```

---

## 7. Data Storage & Git Hygiene Policy

To maintain clean Git history and comply with GitHub's 100 MB file size limit:
- **Raw & Interim Datasets Are Excluded:** Raw archives (`archive.zip` at 2.13 GB, `train.zip` at 543 MB) and extracted image folders (`data/raw/`, `data/interim/` containing 24,000+ loose frames) are strictly ignored by `.gitignore`.
- **Precomputed Feature Matrices Are Excluded:** NumPy arrays (`data/processed/*.npy`) containing dense float32 matrices are excluded from version control.
- **Reproducibility Guides Provided:** Each task includes explicit dataset setup guides (`DATASET_SETUP.md`, `DATASET_INSPECTION.md`, or README sections) explaining how to download or extract the data locally.
- **Tracked Assets:** All source code, tests, Streamlit applications, CSV manifests (`gesture_manifest.csv`, `benchmark_manifest.csv`), evaluation metrics JSONs, markdown reports, and publication figures (`reports/figures/*.png`) are version-controlled.

---

## 8. License & Acknowledgments

- **Code:** MIT License
- **Ames Housing Dataset:** Provided by Dean De Cock (2011) via Kaggle.
- **Mall Customers Dataset:** Provided by Vijay Choudhary via Kaggle.
- **Dogs vs. Cats Dataset:** Provided by Kaggle and Microsoft Research.
- **LeapGestRecog Dataset:** Provided by GTI-UPM (Grupo de Tratamiento de Imágenes, Universidad Politécnica de Madrid) via Kaggle.
- **Organization:** [SkillCraft Technology](https://skillcrafttech.com) Machine Learning Internship Program.