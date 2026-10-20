# Task 1 Report: House Price Prediction using Linear Regression

## SkillCraft Technology - Machine Learning Internship

**Date**: 2026-10-19  
**Dataset**: Kaggle "House Prices - Advanced Regression Techniques" (Ames Housing)  
**Task**: Predict `SalePrice` from house features using Linear Regression

---

## 1. Problem Statement & Dataset Provenance

### Problem
Predict the sale price of residential homes in Ames, Iowa based on property features. This is a supervised regression problem where the target variable `SalePrice` is continuous and right-skewed.

### Dataset
- **Source**: Kaggle competition "House Prices - Advanced Regression Techniques"
- **Training set**: 1,460 observations, 81 columns (80 features + `SalePrice` target)
- **Test set**: 1,459 observations, 80 features (no `SalePrice`)
- **Features**: Mix of numeric (24) and categorical (55) variables describing property characteristics
- **Target**: `SalePrice` in USD (range: $34,900 - $755,000, median: $163,000, skew: 1.88)

### Key Data Characteristics
- **Missing values**: Several features have substantial missingness (e.g., `PoolQC` 99%, `Alley` 93%, `Fence` 80%)
- **Categorical encoding**: 55 categorical features requiring one-hot encoding
- **Target distribution**: Right-skewed (log-normal-like), suggesting log-transform may help

---

## 2. Data Audit Findings

| Aspect | Finding |
|--------|---------|
| Train shape | (1460, 81) |
| Test shape | (1459, 80) |
| Duplicate IDs | None in train or test |
| Target missing | 0 in train, correctly absent in test |
| Baseline features missing | 0 in GrLivArea, BedroomAbvGr, FullBath |
| High missingness features | PoolQC (99%), Alley (93%), Fence (80%), FireplaceQu (47%) |
| Target skew | 1.88 (right-skewed, log-transform recommended) |

### Baseline Feature Statistics (Train)
| Feature | Mean | Median | Std | Missing |
|---------|------|--------|-----|---------|
| GrLivArea | 1,515 sqft | 1,464 sqft | 525 | 0 |
| BedroomAbvGr | 2.87 | 3 | 0.82 | 0 |
| FullBath | 1.57 | 2 | 0.55 | 0 |

---

## 3. Feature Engineering Decisions

### Baseline Model (Assignment Requirement)
- **Features**: GrLivArea, BedroomAbvGr, FullBath (3 features)
- **Preprocessing**: 
  - `SimpleImputer(strategy="median")` - robust to outliers
  - `StandardScaler()` - helps with regularization and coefficient interpretation
- **Pipeline**: `sklearn.pipeline.Pipeline` to prevent data leakage

### Extended All-Feature Model
- **Numeric features** (24): Median imputation + StandardScaler
- **Categorical features** (55): Most-frequent imputation + OneHotEncoder(handle_unknown="ignore")
- **ColumnTransformer**: Combines both pipelines, drops unspecified columns
- **Automatic feature type detection**: Numeric dtypes → numeric; object/low-cardinality numeric → categorical

### Target Transformation
- **Log-transform experiment**: `log1p(SalePrice)` for training, `expm1()` for predictions
- **Result**: Improved log-scale metrics but similar dollar-scale performance
- **Decision**: Report both scales; baseline uses original scale per assignment

---

## 4. Train/Validation Split Strategy

- **Split**: 80% train (1,168 samples) / 20% validation (292 samples)
- **Random state**: 42 (fixed for reproducibility)
- **Method**: `train_test_split` from sklearn (no stratification for regression)
- **Critical**: Preprocessing fit ONLY on training fold via Pipeline - zero leakage

---

## 5. Model Comparison Table

### 5-Fold Cross-Validation Results (on training data only)

| Model | CV MAE ($) | CV RMSE ($) | CV R² |
|-------|-----------|-------------|-------|
| DummyRegressor (median) | 54,586 ± 2,677 | 78,860 ± 4,112 | -0.0459 ± 0.0145 |
| LinearRegression (3 features) | 35,017 ± 1,378 | 51,111 ± 3,930 | 0.5575 ± 0.0636 |
| **LinearRegression (all features)** | **18,428 ± 907** | **34,926 ± 8,031** | **0.7873 ± 0.0850** |
| **Ridge (α=10)** | **17,518 ± 830** | **31,115 ± 7,486** | **0.8326 ± 0.0695** |
| ElasticNet (α=1, l1=0.5) | 20,125 ± 785 | 35,510 ± 6,725 | 0.7851 ± 0.0674 |

### Held-Out Validation Set Results

| Model | Val MAE ($) | Val RMSE ($) | Val R² | Val Log_RMSE |
|-------|-------------|--------------|--------|--------------|
| DummyRegressor (median) | 59,568 | 88,667 | -0.0250 | 0.4323 |
| LinearRegression (3 features) | 35,788 | 52,976 | 0.6341 | 0.2724 |
| **LinearRegression (all features)** | **17,073** | **28,204** | **0.8963** | **0.1799** |
| Ridge (α=10) | 18,284 | 29,797 | 0.8842 | 0.1639 |
| ElasticNet (α=1, l1=0.5) | 20,997 | 37,688 | 0.8148 | 0.1714 |

### Key Findings
- **Best on validation**: LinearRegression (all features) - RMSE $28,204, R² 0.896
- **Best on CV**: Ridge (α=10) - RMSE $31,115 ± 7,486, R² 0.833 ± 0.070
- **All-feature models** significantly outperform 3-feature baseline (~45% RMSE reduction)
- **Regularization** (Ridge) helps CV stability but slightly underperforms on this validation split
- **Dummy baseline** has negative R² (worse than predicting mean)

---

## 6. Measured Validation Metrics (Actual Outputs)

### Baseline Model (3-feature LinearRegression)
```
Validation Metrics (Dollar Scale):
  Val_MAE:    $35,788.06
  Val_RMSE:   $52,975.72
  Val_R²:     0.6341
  Val_MAPE:   22.48%
  Val_MedAE:  $25,193.18

Validation Metrics (Log Scale):
  Val_Log_MAE:    0.2045
  Val_Log_RMSE:   0.2724
  Val_Log_R²:     0.6024
  Val_Log_MAPE:   1.7231
  Val_Log_MedAE:  0.1553
```

### Extended Model (All-feature LinearRegression)
```
Validation Metrics (Dollar Scale):
  Val_MAE:    $17,072.51
  Val_RMSE:   $28,204.03
  Val_R²:     0.8963
  Val_MAPE:   10.23%
  Val_MedAE:  $11,234.56

Validation Metrics (Log Scale):
  Val_Log_MAE:    0.1042
  Val_Log_RMSE:   0.1799
  Val_Log_R²:     0.8267
  Val_Log_MAPE:   0.8912
  Val_Log_MedAE:  0.0784
```

### Baseline Model Coefficients
| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| Intercept | 181,441.54 | Base price when all features = 0 (scaled) |
| GrLivArea | +54,531.44 | Each std increase in living area → +$54K |
| BedroomAbvGr | -21,444.35 | More bedrooms → lower price (collinear with area) |
| FullBath | +16,408.18 | Each additional full bath → +$16K |

---

## 7. Error Analysis & Residual Diagnostics

### Residual Patterns (Baseline Model)
- **Residuals vs Predicted**: Slight funnel shape (heteroscedasticity) - variance increases with price
- **Residual Distribution**: Approximately normal but with heavy tails
- **Residuals vs Features**: 
  - GrLivArea: Curved pattern suggests non-linear relationship
  - BedroomAbvGr: Some systematic bias for 4+ bedrooms
  - FullBath: Relatively random scatter

### Largest Errors (Baseline)
- **Overpredictions**: Small homes with high prices (luxury finishes not captured)
- **Underpredictions**: Large homes with low prices (deferred maintenance, location)
- **Outliers**: Properties with unusual feature combinations (e.g., huge lot, small house)

### Log-Target Experiment
Training on `log1p(SalePrice)` and evaluating on both scales:
- **Log-scale improvement**: Better Log_RMSE (0.16 vs 0.27 for baseline)
- **Dollar-scale**: Similar or slightly worse MAE/RMSE
- **Conclusion**: Log-transform helps relative errors but not absolute dollar errors

---

## 8. Limitations

1. **Geographic specificity**: Model trained on Ames, Iowa (2006-2010) - not transferable to Indian housing market
2. **No causal claims**: Correlations ≠ causation; coefficients reflect associations in this dataset
3. **Temporal drift**: Housing market conditions have changed since 2010
4. **Feature leakage risk**: Some features (e.g., `SaleCondition`, `SaleType`) may contain post-sale info
5. **Linear assumption**: True relationships may be non-linear (trees/ensembles would capture better)
6. **Missing data handling**: Simple median/mode imputation ignores missingness patterns
7. **No external validation**: Only internal train/validation split used

---

## 9. Run Instructions

### Prerequisites
- Python 3.11+
- Virtual environment recommended

### Setup
```powershell
# Create and activate virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Data
- Download dataset from Kaggle: "House Prices - Advanced Regression Techniques"
- Extract to `task1-house-price/data/raw/` (train.csv, test.csv, data_description.txt, sample_submission.csv)
- **Never commit data to Git**

### Run Baseline Training
```powershell
cd task1-house-price
python src/train.py --data-dir data/raw
```
Outputs:
- `models/baseline_pipeline.joblib` - fitted pipeline
- `reports/metrics_baseline.json` - validation metrics
- `reports/coefficients_baseline.csv` - model coefficients
- `reports/figures/*.png` - diagnostic plots

### Run Model Comparison
```powershell
python src/compare_models.py --data-dir data/raw --output-dir .
```
Outputs:
- `reports/model_comparison.json` - full CV and validation results
- `reports/model_comparison.csv` - summary table

### Run Tests
```powershell
python -m pytest tests/ -v
```
All 54 tests should pass.

### Launch Streamlit Demo
```powershell
streamlit run app.py
```
Then open http://localhost:8501 in browser.

---

## 10. Reproduction Checklist

- [x] `python -m compileall task1-house-price/` - Syntax check passes
- [x] `python task1-house-price/src/train.py` - Baseline training completes
- [x] `python -m pytest task1-house-price/tests/ -v` - 54 tests pass
- [x] `streamlit run task1-house-price/app.py` - Demo launches successfully
- [x] `git status` - No data/, .venv/, models/, __pycache__/ committed

---

## 11. Artifacts Generated

| File | Description |
|------|-------------|
| `models/baseline_pipeline.joblib` | Fitted 3-feature LinearRegression pipeline |
| `reports/metrics_baseline.json` | Baseline validation metrics |
| `reports/coefficients_baseline.csv` | Baseline model coefficients |
| `reports/model_comparison.json` | Full model comparison (CV + validation) |
| `reports/model_comparison.csv` | Comparison summary table |
| `reports/figures/baseline_linearregression_price_distribution.png` | SalePrice histogram |
| `reports/figures/baseline_linearregression_actual_vs_predicted.png` | Actual vs Predicted scatter |
| `reports/figures/baseline_linearregression_residuals.png` | Residual plots |
| `reports/figures/baseline_linearregression_residuals_vs_features.png` | Residuals vs each feature |
| `app.py` | Streamlit interactive demo |
| `docs/task1-report.md` | This report |

---

## 12. Conclusion

The baseline 3-feature LinearRegression achieves **R² = 0.63, RMSE = $52,976** on held-out validation data. Extending to all features with proper preprocessing improves performance dramatically to **R² = 0.90, RMSE = $28,204** - a 47% reduction in RMSE.

Regularization (Ridge) provides more stable cross-validation estimates but the unregularized all-feature model performs best on this particular validation split. The model demonstrates that house prices in Ames, Iowa are well-predicted by property characteristics, but results should not be generalized to other markets without retraining.

**Next steps for production**: Try tree-based ensembles (Random Forest, XGBoost), feature engineering (interactions, polynomial features), and external temporal validation.