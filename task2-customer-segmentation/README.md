# Task 2: Mall Customer Segmentation (K-Means Clustering)

## SkillCraft Technology — Machine Learning Internship

### 1. Project Overview
This project segments retail mall customers into distinct behavioral groups based on demographic and purchasing attributes. By identifying customer clusters, mall management and marketing teams can design targeted promotional strategies, enhance customer retention, and optimize resource allocation.

---

### 2. Dataset Provenance & Schema
- **Dataset**: Mall Customer Segmentation Data
- **Kaggle Source**: [`vjchoudhary7/customer-segmentation-tutorial-in-python`](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- **Local Location**: `data/raw/Mall_Customers.csv`
- **Integrity Audit**: 200 records, 5 columns, 0 missing values, 0 duplicate rows.

| Raw Column Header | Normalized Python Field | Data Type | Valid Domain / Description |
| :--- | :--- | :--- | :--- |
| `CustomerID` | `CustomerID` | `int64` | `[1, 200]` — Sequential identifier (strictly excluded from models) |
| `Gender` | `Gender` | `object` / `str` | `{'Male', 'Female'}` — Customer biological gender |
| `Age` | `Age` | `int64` | `[18, 70]` — Customer age in years |
| `Annual Income (k$)` | `Annual_Income` | `int64` | `[15, 137]` — Annual income in thousands of USD |
| `Spending Score (1-100)` | `Spending_Score` | `int64` | `[1, 99]` — Proprietary behavioral spending score |

---

### 3. Engineering & Preprocessing Architecture

#### A. Feature Set Definitions
1. **Classic Segmentation (2D Benchmark)**:
   - Features: `["Annual_Income", "Spending_Score"]`
   - Purpose: Direct benchmark for K-Means clustering, matching the canonical 5-cluster visual separation in 2D Euclidean space.
2. **Extended Segmentation (4D Demographic + Behavioral)**:
   - Features: `["Age", "Annual_Income", "Spending_Score", "Gender_Male"]`
   - Purpose: Richer persona profiling incorporating age cohorts and binary gender indicator (`Female=0.0`, `Male=1.0`).

#### B. Why `CustomerID` is Strictly Excluded
`CustomerID` is an arbitrary sequential key assigned upon record creation. It possesses zero behavioral, economic, or demographic meaning. Including it in a distance-based algorithm like K-Means would corrupt the feature space: customers would cluster according to their position in the database rather than their actual shopping behavior.

#### C. Why Feature Scaling is Essential for Clustering
K-Means relies on **Euclidean distance**:
$$d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$

In the unscaled dataset:
- `Annual_Income` spans $15 to $137 (range: 122 units, variance: ~690).
- `Spending_Score` spans 1 to 99 (range: 98 units, variance: ~667).
- `Age` spans 18 to 70 (range: 52 units, variance: ~195).
- Binary `Gender` spans 0 to 1 (range: 1 unit, variance: ~0.25).

Without standardization (`StandardScaler`), attributes with larger magnitudes and variances artificially dominate the squared distance calculation, rendering smaller-magnitude features (like age and gender) nearly invisible to the clustering objective function. `StandardScaler` centers each continuous feature to $\mu = 0$ and $\sigma = 1$, giving each feature an equitable geometric weighting.

#### D. Leakage Prevention
`CustomerPreprocessor` cleanly separates `.fit()` and `.transform()`. When train/test splits or cross-validation are used, scaler statistics ($\mu, \sigma$) and encoder categories are computed strictly on training partitions to prevent data leakage.

---

### 4. Project Directory Structure
```text
task2-customer-segmentation/
├── app.py                           # Phase 4 Interactive Streamlit dashboard
├── data/
│   ├── raw/                         # Raw Mall_Customers.csv (audit baseline)
│   └── processed/                   # mall_customers_segmented.csv
├── models/                          # Serialized K-Means models
│   ├── kmeans_classic_k5.joblib
│   └── kmeans_extended_k5.joblib
├── reports/
│   ├── eda_summary.md               # Human-readable statistical audit
│   ├── cluster_evaluation.csv       # Quantitative metrics (k=1..10)
│   ├── cluster_evaluation.md        # Comprehensive model evaluation report
│   └── figures/                     # 16 publication-ready visualizations
│       ├── age_distribution.png
│       ├── annual_income_distribution.png
│       ├── spending_score_distribution.png
│       ├── gender_distribution.png
│       ├── income_vs_spending.png
│       ├── age_vs_spending.png
│       ├── elbow_curve_classic.png
│       ├── elbow_curve_extended.png
│       ├── silhouette_curve_classic.png
│       ├── silhouette_curve_extended.png
│       └── cluster_scatter_*.png
├── src/                             # Core modular source code
│   ├── __init__.py
│   ├── data_loader.py               # Path resolution, schema validation & normalization
│   ├── preprocessing.py             # CustomerPreprocessor, scaling & encoding
│   ├── eda.py                       # Automated EDA workflow & chart generator
│   ├── clustering.py                # CustomerClustering, K-Means & unscaled centroids
│   ├── evaluation.py                # Clustering metrics, stability assessment & reports
│   └── app_helpers.py               # Testable dashboard filtering, validation & metrics
├── tests/                           # Complete test suite (59 passing tests)
│   ├── __init__.py
│   ├── test_data_loader.py          # Schema, path, validation, and error tests
│   ├── test_preprocessing.py        # Scaling, encoding, row preservation, and leakage tests
│   ├── test_clustering.py           # Model fit, predict, centroids, and serialization tests
│   ├── test_evaluation.py           # Metric calculation and reporting tests
│   └── test_app_helpers.py          # Dashboard filters, input validation, and summary tests
└── README.md                        # Comprehensive project documentation
```

---

### 5. Execution & Verification Guide

#### Running EDA and Generating Visual Reports
To execute the automated EDA workflow from the project root:
```powershell
& .venv\Scripts\python.exe task2-customer-segmentation/src/eda.py
```
Or from within `task2-customer-segmentation/`:
```powershell
& ..\.venv\Scripts\python.exe src/eda.py
```

Outputs generated:
- `reports/eda_summary.md`: Detailed markdown summary of descriptive statistics, skewness, and demographic breakdowns.
- `reports/figures/*.png`: Six 300-DPI EDA charts.

#### Running Clustering Model Selection & Evaluation
To execute the complete K-Means model evaluation across both feature sets:
```powershell
& .venv\Scripts\python.exe task2-customer-segmentation/src/evaluation.py
```
Or from within `task2-customer-segmentation/`:
```powershell
& ..\.venv\Scripts\python.exe src/evaluation.py
```

Outputs generated:
- `reports/cluster_evaluation.csv`: Quantitative metrics ($WSS$, Silhouette, Davies-Bouldin, Calinski-Harabasz) for $k \in [2, 10]$.
- `reports/cluster_evaluation.md`: Detailed model selection analysis, stability assessments, and demographic cluster profiles.
- `reports/figures/elbow_curve_*.png` & `silhouette_curve_*.png`: High-resolution evaluation curves.
- `reports/figures/cluster_scatter_*.png`: 2D cluster scatters with unscaled centroids.
- `data/processed/mall_customers_segmented.csv`: Segmented customer records with assigned cluster labels.
- `models/kmeans_classic_k5.joblib` & `models/kmeans_extended_k5.joblib`: Persisted, verified clustering models.

#### Running Unit Tests
Execute the full Task 2 test suite (59 passing tests):
```powershell
& .venv\Scripts\python.exe -m pytest task2-customer-segmentation/tests -v -o pythonpath=task2-customer-segmentation
```
Or from within `task2-customer-segmentation/`:
```powershell
& ..\.venv\Scripts\python.exe -m pytest tests -v
```

---

### 6. Interactive Streamlit Dashboard (`app.py`)

A professional, portfolio-grade analytics web application built with Streamlit and Altair.

#### Prerequisites
The dashboard uses the existing repository virtual environment (`.venv`). No additional package installation or dependency modification is required.
- Python 3.11+
- `streamlit` (v1.44+)
- `altair` (v5+)
- `scikit-learn`, `joblib`, `pandas`, `numpy`, `matplotlib`, `seaborn`

#### Exact Launch Command (Windows PowerShell)
To launch the dashboard from within `task2-customer-segmentation/`:
```powershell
cd "C:\Users\HP\my new project (skillcraft)\SkillCraft-ML-Portfolio\task2-customer-segmentation"
& "..\.venv\Scripts\python.exe" -m streamlit run app.py --server.port 8504
```

Alternatively, from the repository root:
```powershell
cd "C:\Users\HP\my new project (skillcraft)\SkillCraft-ML-Portfolio"
& ".\.venv\Scripts\python.exe" -m streamlit run task2-customer-segmentation/app.py --server.port 8504
```

The application will start and serve at `http://localhost:8504`.

#### Dashboard Structure & Features
The interface is organized into six functional tabs designed with high-contrast, light-themed typography and accessible color palettes:

1. **Executive Overview**:
   - High-level KPIs: Total Customers (200), Average Annual Income ($60.56k), Average Spending Score (50.20), Total Clusters (5).
   - Dynamic cluster population breakdown with absolute counts and percentages.
   - Comprehensive, data-grounded overview of the dataset.

2. **Interactive Cluster Explorer**:
   - Model selection toggle: **Classic (2D: Income + Spending)** vs. **Extended (4D: Age, Income, Spending, Gender)**.
   - High-performance, responsive Altair scatter plot mapping Annual Income vs. Spending Score.
   - Cluster-colored marks with cross-shaped, high-contrast cluster centroids in original dollar/score units.
   - Customer-level hover tooltips displaying Customer ID, Age, Gender, Income, Spending Score, and assigned Cluster.
   - Non-destructive sidebar filters (Cluster selection, Income range, Spending range, Gender) affecting display only without altering model data.

3. **Segment Profiles**:
   - Population share metrics per cluster.
   - Income and spending distributions compared across clusters.
   - Demographic summaries (median age, gender distribution) presented as descriptive profiles without claiming causality.
   - Objective, evidence-based behavioral persona descriptions (e.g., Target Group, Frugal High Earners, Impulsive Shoppers).

4. **New Customer Prediction**:
   - Interactive input controls for Age, Annual Income ($k), Spending Score (1–100), and Gender.
   - Predicts customer segment in real-time using the active saved model (`kmeans_classic_k5.joblib` or `kmeans_extended_k5.joblib`).
   - Classic model feeds only income and spending into inference; age and gender are displayed as contextual metadata.
   - Extended model passes full 4-feature schema through fitted `CustomerPreprocessor`.
   - Displays unscaled centroid comparisons and an explicit disclaimer that cluster labels are arbitrary identifiers, not performance ranks.
   - Robust validation with friendly UI error notices without exposing raw stack traces.

5. **Model Evaluation & Diagnostics**:
   - Comparative evaluation metrics table: WSS / Inertia, Silhouette Coefficient, Davies–Bouldin Index, and Calinski–Harabasz Index.
   - Publication-quality diagnostic plots (Elbow curve and Silhouette score curve) sourced directly from `reports/figures/`.
   - Clear technical explanation of why Classic $k=5$ is the primary retail segmentation benchmark.
   - Honest reporting that Extended $k=6$ achieved a marginally higher silhouette score (0.3565) than $k=5$ (0.3498), clarifying that $k=5$ balances parsimony and business interpretability.
   - Explicit guidance on why clustering metrics from different feature spaces (2D vs. 4D) cannot be compared naively.

6. **Data & Methodology**:
   - Clear dataset provenance and feature engineering description.
   - Architectural justification for strictly excluding `CustomerID` and standardizing continuous variables.
   - Review of K-Means mathematical assumptions (spherical clusters, isotropic variance) and limitations in retail applications.
   - Paginated/filtered data table preview with record limit selector.
   - Dedicated `Download Filtered Data (CSV)` button to export current filtered records.

#### Model Comparison: Classic vs. Extended

| Dimension | Classic K-Means ($k=5$) | Extended K-Means ($k=5$) |
| :--- | :--- | :--- |
| **Feature Set** | `Annual_Income`, `Spending_Score` | `Age`, `Annual_Income`, `Spending_Score`, `Gender` |
| **Dimensionality** | 2D Euclidean continuous space | 4D mixed continuous & one-hot encoded space |
| **Preprocessing** | `StandardScaler` on 2 features | `StandardScaler` (continuous) + `OneHotEncoder` (`drop='first'`) |
| **Silhouette Score** | **0.5547** | **0.3498** (Note: $k=6$ scored 0.3565) |
| **Davies–Bouldin** | **0.5722** (lower is better) | **1.0245** |
| **Interpretability** | Immediate visual separation into 5 distinct behavioral quadrants | Richer demographic personas, but reduced geometric separation due to curse of dimensionality |
| **Primary Use Case** | Behavioral spending segmentation | Demographic-augmented lifestyle profiling |

#### Limitations & Safety Notes
- **Arbitrary Cluster Identifiers**: Cluster labels (0, 1, 2, 3, 4) are mathematical partition IDs assigned by K-Means initialization. They imply no ranking, quality, or creditworthiness.
- **Correlation vs. Causation**: Unsupervised clusters identify geometric groupings in historical data; they do not prove customer intent or guarantee future purchasing behavior.
- **Feature Space Comparability**: Silhouette and Davies-Bouldin metrics cannot be directly compared between 2D and 4D models because distances are measured across different manifolds.

---

### 7. Development Roadmap
- [x] **Phase 1: Setup & Data Audit**: Repository scaffolding, dataset acquisition, schema verification.
- [x] **Phase 2: EDA & Preprocessing Pipeline**: Modular data loader, schema normalization, leak-free `CustomerPreprocessor`, 6 EDA figures, structured summary report, and 29 unit tests.
- [x] **Phase 3: Model Development & Optimal $k$ Selection**:
  - Inertia / Elbow Method ($k = 1$ to $10$).
  - Silhouette Coefficient and Davies-Bouldin index evaluations.
  - Multi-seed stability verification (Adjusted Rand Index $\ge 0.98$).
  - K-Means model training on both `classic` ($k = 5$) and `extended` ($k = 5$) feature sets.
  - Data-supported customer cluster profiles and cautious retail interpretations.
  - Model persistence (`models/*.joblib`), segmented data export, and canonical class serialization fix (50 tests passing).
- [x] **Phase 4: Streamlit Interactive Dashboard** *(Completed)*:
  - Six-section portfolio dashboard in `app.py` with Altair interactive charting and dynamic filtering.
  - Testable helper architecture in `src/app_helpers.py` with 9 dedicated unit tests (59 total tests passing).
  - Model-switching, unscaled centroid overlays, real-time prediction, and CSV export.
  - Full startup and browser subagent interaction verification across all tabs.
- [ ] **Phase 5: Final Review & Submission Documentation**.

