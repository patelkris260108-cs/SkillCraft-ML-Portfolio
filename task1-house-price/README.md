# Task 1: House Price Prediction (Linear Regression & Ridge)

## SkillCraft Technology — Machine Learning Internship

### 1. Project Overview
This project builds a supervised regression pipeline to predict residential real estate sale prices in Ames, Iowa based on property features. Using the **Ames Housing** dataset (Kaggle *House Prices — Advanced Regression Techniques*), the project implements both a leak-free 3-feature baseline model satisfying core internship requirements and an extended all-feature regularized pipeline for comprehensive predictive modeling.

---

### 2. Dataset Provenance & Features
- **Dataset**: Ames Housing Dataset
- **Source**: [Kaggle House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (Dean De Cock, 2011)
- **Local Location**: `data/raw/` (`train.csv`, `test.csv`, `data_description.txt`)
- **Sample Counts**: 1,460 training houses (80 features + `SalePrice`), 1,459 test houses (unlabeled)
- **Target Variable**: `SalePrice` in USD (continuous, right-skewed, median: $163,000, skew: 1.88)

#### Feature Sets
1. **Baseline Model (Core Requirement)**:
   - `GrLivArea`: Above-grade living area in square feet
   - `BedroomAbvGr`: Bedrooms above grade
   - `FullBath`: Full bathrooms above grade
2. **Extended Model (All Features)**:
   - All 79 explanatory features (24 numeric, 55 categorical) covering building geometry, condition ratings, neighborhood, and quality ratings.

---

### 3. Preprocessing & Model Architecture

```
Raw Property Features
       │
       ▼
ColumnTransformer (Leak-Free Pipeline)
 ├── Numeric: SimpleImputer(strategy="median") + StandardScaler()
 └── Categorical: SimpleImputer(strategy="most_frequent") + OneHotEncoder(handle_unknown="ignore")
       │
       ▼
Regressor Pipeline
 ├── Baseline: LinearRegression() on 3 features
 └── Extended: LinearRegression() / Ridge(alpha=10) on all features
       │
       ▼
Predictions & Residual Diagnostics (Dollar-scale and Log-scale metrics)
```

- **Leakage Prevention**: All imputers, scalers, and encoders are encapsulated inside `sklearn.pipeline.Pipeline` and fitted strictly on training data folds (`random_state=42`, 80/20 train/validation split).

---

### 4. Evaluation Metrics & Benchmark Results

Metrics are reported on the held-out validation set (292 samples) and via 5-fold cross-validation on training data (1,168 samples):

| Model | Val MAE ($) | Val RMSE ($) | Val $R^2$ | Val RMSLE | 5-Fold CV $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `DummyRegressor` (Median) | $59,568 | $88,667 | -0.0250 | 0.4323 | -0.0459 ± 0.0145 |
| **Linear Regression (3 Features)** | **$35,788** | **$52,976** | **0.6341** | **0.2724** | **0.5575 ± 0.0636** |
| **Linear Regression (All Features)** | **$17,073** | **$28,204** | **0.8963** | **0.1799** | **0.7873 ± 0.0850** |
| **Ridge Regression ($\alpha=10$)** | **$18,284** | **$29,797** | **0.8842** | **0.1639** | **0.8326 ± 0.0695** |
| `ElasticNet` ($\alpha=1, l_1=0.5$) | $20,997 | $37,688 | 0.8152 | 0.1712 | 0.7851 ± 0.0674 |

- Complete evaluation details: [`../docs/task1-report.md`](../docs/task1-report.md)
- Machine-readable metrics: [`reports/metrics_baseline.json`](reports/metrics_baseline.json) and [`reports/model_comparison.json`](reports/model_comparison.json)

---

### 5. Project Directory Structure
```text
task1-house-price/
├── app.py                       # EstateIQ Interactive Streamlit Web Application
├── data/
│   └── raw/                     # train.csv, test.csv, data_description.txt (git-ignored)
├── models/                      # Serialized pipelines (git-ignored)
│   └── baseline_pipeline.joblib # 3-feature LinearRegression pipeline
├── reports/
│   ├── metrics_baseline.json    # Baseline model evaluation metrics
│   ├── model_comparison.json   # Multi-model comparative results
│   ├── split_indices.json       # Fixed reproducible train/validation indices
│   └── figures/                 # Diagnostic and residual visualization plots
│       ├── baseline_linearregression_actual_vs_predicted.png
│       ├── baseline_linearregression_price_distribution.png
│       ├── baseline_linearregression_residuals.png
│       └── baseline_linearregression_residuals_vs_features.png
├── src/                         # Modular source code
│   ├── data_loader.py           # Ingestion, validation & type casting
│   ├── preprocessing.py         # Pipeline transformers & ColumnTransformer
│   ├── train.py                 # Baseline training and report export
│   ├── compare_models.py        # All-feature cross-validation & model selection
│   ├── predict.py               # HousePricePredictor inference wrapper
│   ├── evaluation.py            # Regression metrics calculation
│   ├── visualization.py         # 2D diagnostic plot generation
│   └── viz_3d.py                # Interactive 3D surface generator
├── tests/                       # Unit and integration test suite (98 passing tests)
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_training.py
│   ├── test_prediction.py
│   └── test_frontend_inference.py
└── README.md                    # Project documentation (this file)
```

---

### 6. Setup, Training & Execution Guide

All commands run using the repository root `.venv` in Windows PowerShell.

#### A. Running the Test Suite
```powershell
cd "C:\Users\HP\my new project (skillcraft)\SkillCraft-ML-Portfolio\task1-house-price"
& "..\.venv\Scripts\python.exe" -m pytest tests -v
```
*(All 98 tests passing)*

#### B. Running Model Training & Comparison
```powershell
cd "C:\Users\HP\my new project (skillcraft)\SkillCraft-ML-Portfolio\task1-house-price"
# Train baseline model (generates models/baseline_pipeline.joblib and diagnostic plots)
& "..\.venv\Scripts\python.exe" src/train.py

# Run all-feature comparison & cross-validation
& "..\.venv\Scripts\python.exe" src/compare_models.py
```

#### C. Launching the EstateIQ Interactive Web App
```powershell
cd "C:\Users\HP\my new project (skillcraft)\SkillCraft-ML-Portfolio\task1-house-price"
& "..\.venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501
```
Then open `http://localhost:8501` in your web browser.

---

### 7. Important Technical Limitations & Practical Caveats

1. **Geographic & Temporal Specificity**:
   - The model is trained exclusively on residential sales in Ames, Iowa between 2006 and 2010. It cannot be directly generalized to other metropolitan markets or current economic cycles without retraining.
2. **Extreme Outliers**:
   - The dataset contains atypical properties (e.g. homes with > 4,000 sq ft living area sold under distress conditions). As noted in Dean De Cock's documentation, extreme luxury outliers can disproportionately impact ordinary least squares.
3. **Educational Valuation Estimates Only**:
   - Predictions generated by this tool are statistical approximations designed for portfolio demonstration; they do not constitute formal real estate appraisals.
4. **Local Data Requirement**:
   - Raw CSVs (`train.csv`, `test.csv`) are excluded from Git repository tracking by design and must be placed in `data/raw/` prior to running training scripts from scratch.
