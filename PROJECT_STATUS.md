# PROJECT_STATUS.md - Implementation Log

## Project: SkillCraft ML Portfolio - Task 1: House Price Prediction

**Started**: 2026-10-19  
**Python**: 3.11.9  
**Environment**: Windows PowerShell  

---

## Phase 1: Environment & Workspace Setup

### ✅ 1.1 Folder Structure Created
- [x] `SkillCraft-ML-Portfolio/` root
- [x] `task1-house-price/data/raw/`
- [x] `task1-house-price/src/`
- [x] `task1-house-price/models/`
- [x] `task1-house-price/reports/figures/`
- [x] `task1-house-price/tests/`
- [x] `docs/`

### ✅ 1.2 Core Files Created
- [x] `.gitignore` - Excludes data, venv, models, cache, secrets
- [x] `requirements.txt` - Core ML stack + visualization + testing
- [x] `AGENTS.md` - Agent instructions and principles
- [x] `README.md` - Project overview and quick start
- [x] `PROJECT_STATUS.md` - This file

### ✅ 1.3 Python Environment
- [x] Check available Python versions (`py -0p`)
- [x] Create virtual environment (`py -3.11 -m venv .venv`)
- [x] Activate environment (`.\.venv\Scripts\Activate.ps1`)
- [x] Install dependencies (`python -m pip install -r requirements.txt`)
- [x] Verify installation (`python -m pip check`)

### ✅ 1.4 Dataset Download
- [x] User provided downloaded zip file
- [x] Extracted to `task1-house-price/data/raw/`
- [x] Verified files: `train.csv`, `test.csv`, `data_description.txt`, `sample_submission.csv`
- [x] Confirmed `SalePrice` in train, absent in test

---

## Phase 2: Data Audit & Exploration (Milestone A)

### ✅ 2.1 Data Loader Module (`src/data_loader.py`)
- [x] Load train/test CSVs with dtype handling
- [x] Validate expected columns
- [x] Check duplicate `Id` values
- [x] Basic info logging

### ✅ 2.2 Audit Script (run via data_loader)
- [x] `train.shape`: (1460, 81), `test.shape`: (1459, 80)
- [x] Column names and data types logged
- [x] `SalePrice` distribution: mean=$180,921, median=$163,000, skew=1.88
- [x] Missing value counts/percentages per column
- [x] Feature descriptions from `data_description.txt`
- [x] Target column presence in train, absence in test

### ✅ 2.3 Unit Tests for Data Loading (`tests/test_data_loader.py`)
- [x] Test file existence and readability (14 tests)
- [x] Test column name validation
- [x] Test target column presence/absence
- [x] Test duplicate ID detection
- [x] Test data description loading
- [x] Test convenience function

---

## Phase 3: Baseline Model Implementation (Milestone B)

### ✅ 3.1 Preprocessing Pipeline (`src/preprocessing.py`)
- [x] Feature selection: GrLivArea, BedroomAbvGr, FullBath
- [x] Median imputation for numeric features
- [x] StandardScaler
- [x] Target log1p transformation (LogTargetTransformer)
- [x] sklearn Pipeline to prevent leakage

### ✅ 3.2 Train/Validation Split
- [x] 80/20 split, random_state=42
- [x] Save split indices for reproducibility

### ✅ 3.3 Baseline Training (`src/train.py`)
- [x] LinearRegression in full pipeline
- [x] Fit on training data only
- [x] Predict on validation set
- [x] Compute MAE, RMSE, R² (log and dollar scale)
- [x] Save model: `models/baseline_pipeline.joblib`
- [x] Save metrics: `reports/metrics_baseline.json`
- [x] Save coefficients: `reports/coefficients_baseline.csv`

### ✅ 3.4 Visualization (`src/visualization.py`)
- [x] Actual vs Predicted scatter plot
- [x] Residual plot (predicted vs residual)
- [x] SalePrice distribution histogram (with log scale option)
- [x] Residuals vs each feature
- [x] Save to `reports/figures/`

### ✅ 3.5 Unit Tests for Training (`tests/test_training.py`, `tests/test_preprocessing.py`, `tests/test_prediction.py`)
- [x] Test pipeline fitting
- [x] Test imputer/scaler fit only on training
- [x] Test prediction shape
- [x] Test metrics finite and positive
- [x] Test model save/load roundtrip
- [x] Test prediction validation
- [x] Test reproducibility

### ✅ 3.6 All Tests Pass
- [x] 98 tests passing across all test modules

---

## Phase 4: Extended All-Features Model (Milestone C)

### ✅ 4.1 Full Feature Preprocessing (`src/preprocessing.py` - implemented)
- [x] ColumnTransformer: numeric (median + scale) + categorical (most-frequent + OneHot)
- [x] handle_unknown='ignore' for categorical
- [x] Automatic feature type detection

### ✅ 4.2 Model Comparison
- [x] DummyRegressor (median baseline)
- [x] 3-feature LinearRegression
- [x] All-feature LinearRegression
- [x] Ridge (alpha=10)
- [x] ElasticNet (alpha=1, l1_ratio=0.5)

### ✅ 4.3 Cross-Validation
- [x] 5-fold CV on training data only
- [x] Mean ± std for each metric
- [x] Identical folds for fair comparison (KFold with random_state=42)

### ✅ 4.4 Error Analysis
- [x] Largest absolute errors inspection (in report)
- [x] Outlier property analysis (in report)
- [x] Log-target experiment (in report)

### 📊 Model Comparison Results

**Best on Validation Set**: LinearRegression (all features)
- Val_RMSE: $28,204, Val_R²: 0.8963

**Best on Cross-Validation**: Ridge (alpha=10)
- CV_RMSE: $31,115 ± 7,486, CV_R²: 0.8326 ± 0.0695

**Full Results** (Validation Set):
| Model | MAE | RMSE | R² | Log_RMSE |
|-------|-----|------|-----|----------|
| DummyRegressor (median) | $59,568 | $88,667 | -0.025 | 0.432 |
| LinearRegression (3 features) | $35,788 | $52,976 | 0.634 | 0.272 |
| **LinearRegression (all features)** | **$17,073** | **$28,204** | **0.896** | **0.180** |
| Ridge (α=10) | $18,284 | $29,797 | 0.884 | 0.164 |
| ElasticNet (α=1, l1=0.5) | $20,997 | $37,688 | 0.815 | 0.171 |

---

## Phase 5: Streamlit Demo (Milestone D)

### ✅ 5.1 App Structure (`app.py`)
- [x] Load saved baseline pipeline
- [x] Input widgets: GrLivArea, BedroomAbvGr, FullBath
- [x] Predict button → estimated price
- [x] Disclaimer: Educational estimates only
- [x] Input validation

### ✅ 5.2 Demo Features
- [x] Model metrics on sidebar
- [x] Residual plot display
- [x] Batch prediction via CSV upload
- [x] Verified working: `streamlit run app.py` launches on port 8501

---

## Phase 6: Final Report & Documentation

### ✅ 6.1 Report (`docs/task1-report.md`)
- [x] Problem statement & dataset provenance
- [x] Data audit findings
- [x] Feature engineering decisions
- [x] Split strategy
- [x] Model comparison table
- [x] Measured validation metrics (actual outputs)
- [x] Error analysis & residual diagnostics
- [x] Limitations
- [x] Run instructions

### ✅ 6.2 Final Verification
- [x] `python -m compileall task1-house-price/` - No syntax errors
- [x] `python task1-house-price/src/train.py` - Training completes successfully
- [x] `python -m pytest task1-house-price/tests/ -v` - 98 tests pass
- [x] `streamlit run task1-house-price/app.py` - Demo launches successfully
- [x] `git status` - No data/credentials committed (verified via .gitignore)

---

## Actual Command Outputs Log

### Baseline Training (Final Run)
```
TASK 1: HOUSE PRICE PREDICTION - BASELINE TRAINING
Data directory: data\raw
Validation split: 0.2
Random state: 42
Log target: False

Step 1: Loading and validating data...
✓ Loaded train: (1460, 81), test: (1459, 80)

Step 2: Preparing features and target...
Features: ['GrLivArea', 'BedroomAbvGr', 'FullBath']
Target: SalePrice
X shape: (1460, 3), y shape: (1460,)

Step 3: Creating train/validation split...
Train: 1168 samples
Validation: 292 samples

Step 6: Training Linear Regression...
✓ Model trained
  Intercept: 181441.5420
  Coefficients: GrLivArea=54531.44, BedroomAbvGr=-21444.35, FullBath=16408.18

Step 7: Evaluating on validation set...

VALIDATION METRICS
Val_MAE: 35788.06
Val_RMSE: 52975.72
Val_R2: 0.6341
Val_Log_MAE: 0.2045
Val_Log_RMSE: 0.2724
Val_Log_R2: 0.6024

Step 8: Saving model and metrics...
✓ Model saved: models\baseline_pipeline.joblib
Metrics saved to reports\metrics_baseline.json
✓ Coefficients saved: reports\coefficients_baseline.csv

Step 9: Generating diagnostic plots...
Saved plot: reports\figures\baseline_linearregression_price_distribution.png
Saved plot: reports\figures\baseline_linearregression_actual_vs_predicted.png
Saved plot: reports\figures\baseline_linearregression_residuals.png
Saved plot: reports\figures\baseline_linearregression_residuals_vs_features.png
Generated 4 diagnostic plots for Baseline_LinearRegression

Step 10: Testing prediction on sample...
Sample input: {'GrLivArea': 1068, 'BedroomAbvGr': 3, 'FullBath': 1}
True price: $154,500.00
Predicted:  $113,410.67
Error:      $41,089.33

TRAINING COMPLETE
```

### Tests
```
============================= test session starts =============================
collected 98 items
tests/test_data_loader.py ... 14 passed
tests/test_preprocessing.py ... 13 passed
tests/test_training.py ... 13 passed
tests/test_prediction.py ... 14 passed
tests/test_frontend_inference.py ... 44 passed
============================= 98 passed in 4.12s ==============================
```

### Model Comparison
```
5-Fold Cross-Validation Results:
DummyRegressor (median)       $54,586±2,677  $78,860±4,112  -0.0459±0.0145
LinearRegression (3 features) $35,017±1,378  $51,111±3,930   0.5575±0.0636
LinearRegression (all)        $18,428±907    $34,926±8,031   0.7873±0.0850
Ridge (alpha=10)              $17,518±830    $31,115±7,486   0.8326±0.0695
ElasticNet (alpha=1, l1=0.5)  $20,125±785    $35,510±6,725   0.7851±0.0674

Held-Out Validation Set Results:
LinearRegression (all)        $17,073 $28,204  0.8963  0.1799
Ridge (alpha=10)              $18,284 $29,797  0.8842  0.1639
LinearRegression (3 features) $35,788 $52,976  0.6341  0.2724
```

---

## Issues & Blockers

| Date | Issue | Status | Resolution |
|------|-------|--------|------------|
| 2026-10-19 | DataLoader column count validation failed for test fixtures | Fixed | Added `strict_validation=False` parameter |
| 2026-10-19 | compute_log_metrics double prefix "Log_Log_MAE" | Fixed | Updated test to use prefix="" |
| 2026-10-19 | pytest fixture scope - fixtures in class not accessible | Fixed | Moved fixtures to module level |
| 2026-10-19 | visualization.py NameError: missing Dict import | Fixed | Added Dict to typing imports |
| 2026-10-19 | visualization.py AttributeError: numpy.ndarray has no median() | Fixed | Changed y.median() to np.median(y) |
| 2026-10-19 | kagglehub authentication not available | Workaround | User provided downloaded zip file |
| 2026-10-19 | load_model_comparison() ambiguous file selection | Fixed | Now only loads model_comparison.json with schema validation |
| 2026-10-19 | get_training_data_stats() hardcoded fallbacks | Fixed | Returns None instead of misleading fallbacks; UI shows unavailable state |
| 2026-10-19 | predict_batch() sequential inference | Fixed | Vectorized inference for better performance |

---

## Git Commits Log

| Commit | Message | Files Changed |
|--------|---------|---------------|
| (initial) | Initial commit - Project structure, data loader, preprocessing, training, evaluation, visualization, tests, Streamlit app, documentation | All project files |

---

## Summary

**Task 1 Complete**: House Price Prediction with Linear Regression

### Deliverables Created:
1. **Baseline Model**: 3-feature LinearRegression (GrLivArea, BedroomAbvGr, FullBath)
   - Val_RMSE: $52,976, Val_R²: 0.634
   - Saved: `models/baseline_pipeline.joblib`
   - Metrics: `reports/metrics_baseline.json`

2. **Extended Models**: All-feature comparison with 5-fold CV
   - Best validation: LinearRegression (all features) - RMSE $28,204, R² 0.896
   - Best CV: Ridge (α=10) - RMSE $31,115 ± 7,486, R² 0.833
   - Results: `reports/model_comparison.json`, `reports/model_comparison.csv`

3. **Visualizations**: 4 diagnostic plots per model
   - Price distribution (with log scale)
   - Actual vs Predicted
   - Residuals (vs predicted + histogram)
   - Residuals vs each feature

4. **Streamlit Demo**: Interactive prediction app
   - Single prediction with input validation
   - Batch CSV prediction
   - Model metrics sidebar
   - Residual plot display

5. **Documentation**: Comprehensive report in `docs/task1-report.md`

6. **Tests**: 98 unit tests covering all modules - all passing

7. **Git Hygiene**: `.gitignore` excludes data, venv, models, cache, credentials

---

## Phase 7: Task 1 Test Suite Regression Update
**Date**: 2026-10-27  
**Status**: ✅ All 98 Tests Passing

### Summary:
- Resolved Pandas 2.x integer assignment issue in `tests/test_frontend_inference.py` (`test_validate_batch_csv_non_numeric`) by casting column to `object` prior to setting non-numeric test strings.
- Updated `test_predict_batch_partial_failure` to use `pd.isna()` for missing prediction error representations.
- Executed full test suite: **98 passed in 14.01s** (0 failed, 0 skipped).
- Created dedicated [`task1-house-price/README.md`](task1-house-price/README.md).

---

## Phase 8: Task 2 — Mall Customer Segmentation Completion Log
**Date**: 2026-10-23  
**Status**: ✅ Complete (59 Passing Tests)  
**Track**: Unsupervised Clustering (K-Means)

### Deliverables Created:
1. **Data Preprocessing & Audit**:
   - Ingested 200 customer records from Kaggle Mall Customers.
   - Cleanly excluded non-behavioral `CustomerID`.
   - Scaled continuous features (`StandardScaler`) and encoded binary gender (`OneHotEncoder`).
2. **Model Selection & Evaluation**:
   - Compared Classic 2D (`Annual_Income`, `Spending_Score`) vs. Extended 4D (`Age`, `Annual_Income`, `Spending_Score`, `Gender`) across $k \in [1, 10]$.
   - Evaluated Inertia/Elbow, Silhouette Coefficient, Davies-Bouldin, and Calinski-Harabasz indices.
   - Selected $k=5$ Classic (Silhouette: **0.5547**, Davies-Bouldin: **0.5722**).
   - Serialized models: `models/kmeans_classic_k5.joblib` and `models/kmeans_extended_k5.joblib`.
3. **Interactive Streamlit Dashboard**:
   - 6-tab analytical interface in `task2-customer-segmentation/app.py` with Altair interactive scatter plots, unscaled centroid overlays, dynamic filtering, real-time prediction, and CSV export.
4. **Documentation & Tests**:
   - Generated 16 diagnostic charts in `reports/figures/`.
   - Comprehensive reports: `reports/cluster_evaluation.md` and `reports/eda_summary.md`.
   - Complete test suite: **59 passed in 6.66s**.

---

## Phase 9: Task 3 — Dogs vs. Cats Image Classification Completion Log
**Date**: 2026-10-25  
**Status**: ✅ Complete (50 Passing Tests)  
**Track**: Binary Image Classification (HOG + RBF Kernel SVM)

### Deliverables Created:
1. **Dataset Ingestion & Benchmark Generation**:
   - Validated Kaggle `dogs-vs-cats.zip` (812 MB) and inner `train.zip` (543 MB, 25,000 images, 0 corruptions).
   - Created reproducible 4,000-image balanced benchmark (`reports/benchmark_manifest.csv`, seed 42): Train (2,800), Val (600), Test (600).
   - Safe streaming extraction into `data/interim/`.
2. **HOG Feature Extraction & SVM Training**:
   - Extracted 1,764-dimensional HOG features (64x64 grayscale, 9 orientations, 8x8 cell, 2x2 block, L2-Hys).
   - Evaluated Linear SVM vs. RBF Kernel SVM vs. SGD.
   - Selected RBF Kernel SVM ($C=1.0$) based on validation accuracy (**74.67%**).
   - Evaluated once on held-out test set: **72.50% Accuracy**, **72.50% Macro F1**, **0.7856 ROC-AUC**.
   - Serialized pipeline to `models/best_svm_pipeline.joblib` and `models/hog_config.json`.
3. **Interactive Streamlit Web App**:
   - Multi-tab application in `task3-dogs-vs-cats/app.py` supporting custom image uploads (up to 10 MB) and test gallery evaluation.
   - Live HOG gradient orientation visualizer (Inferno colormap) and signed SVM decision margin gauge.
4. **Documentation & Tests**:
   - Reports in `reports/modeling_report.md` and `reports/eda_summary.md`.
   - Complete test suite: **50 passed in 12.40s**.

---

## Phase 10: Task 4 — Hand Gesture Recognition Completion Log
**Date**: 2026-10-26  
**Status**: ✅ Complete (49 Passing Tests)  
**Track**: Multi-Class Near-Infrared Image Classification (HOG + Linear SVM)

### Deliverables Created:
1. **Archive Safety Audit & Canonical Extraction**:
   - Non-extracting CRC32 verification of `archive.zip` (2.13 GB).
   - Discovered 20,000 duplicate hierarchy frames (`leapgestrecog/leapGestRecog/`) and filtered strictly to canonical `leapGestRecog/` members.
   - Extracted 20,000 canonical NIR frames with Pillow header integrity verification.
   - Generated deterministic `reports/gesture_manifest.csv` with strict **Subject-Independent Partitioning** (Train: Subjects 00–06, Val: Subject 07, Test: Subjects 08–09; zero subject leakage).
2. **Feature Extraction & Modeling**:
   - Extracted 2,700 HOG features per frame (160x60 grayscale preserving native 2.67:1 aspect ratio, 9 orientations, 10x10 cell, 2x2 block, L2-Hys).
   - Compared 5 candidate models; selected `SGD_LinearSVM` by Validation Macro F1 on Subject 07 (**79.75%**, Accuracy **79.95%**).
   - Evaluated on held-out test subjects 08 & 09: **62.20% Accuracy**, **61.25% Macro F1**, **65.20% Precision**, **62.20% Recall**.
   - Serialized model to `models/best_gesture_model.joblib` and `models/model_config.json`.
3. **Interactive Streamlit Web App**:
   - Multi-tab application in `task4-hand-gesture-recognition/app.py` on port 8503.
   - Real-time 2D HOG orientation map rendering, class decision margin bars, and explicit technical callouts on static 2D frame ambiguity and uncalibrated margins.
4. **Documentation & Tests**:
   - Generated 7 figures in `reports/figures/`, `reports/modeling_report.md`, and `reports/eda_summary.md`.
   - Complete test suite: **49 passed in 30.35s**.

---

## Portfolio Final Verification Summary
**Audit Date**: 2026-10-27  
**Total Tests Executed & Passing**: **256 / 256 Passed (100%)**
- Task 1 (House Price Prediction): 98 / 98 Passed
- Task 2 (Customer Segmentation): 59 / 59 Passed
- Task 3 (Dogs vs. Cats Classification): 50 / 50 Passed
- Task 4 (Hand Gesture Recognition): 49 / 49 Passed
- All 4 interactive Streamlit applications verified operational.
- Large datasets (3.5+ GB archives, 24,000+ extracted images) safely excluded by `.gitignore`.
- Portfolio verified ready for internship submission.