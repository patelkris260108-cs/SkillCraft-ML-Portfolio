# AGENTS.md - SkillCraft ML Portfolio Agent Instructions

## Project Overview
This repository contains four Machine Learning internship tasks for SkillCraft Technology:
1. **Task 1**: House Price Prediction (Linear Regression) - `task1-house-price/`
2. **Task 2**: Mall Customer Segmentation (K-Means) - `task2-customer-segmentation/`
3. **Task 3**: Dogs vs Cats Classification (SVM) - `task3-dogs-vs-cats/`
4. **Task 4**: Hand Gesture Recognition (CNN) - `task4-hand-gesture/`

Currently implementing: **Task 1 only**

## Core Principles (Non-Negotiable)

### Data Integrity
- **Never fabricate metrics** - Run code, record actual outputs only
- **Never fit preprocessing on validation/test data** - Use `sklearn.pipeline.Pipeline`, fit ONLY on training fold
- **Never use Kaggle test labels for validation** - They don't exist; use train/validation split
- **Never commit raw data, credentials, or model files** - Enforced via `.gitignore`

### Reproducibility
- Use `random_state=42` for all splits, model initialization, and CV folds
- Save exact command outputs in `PROJECT_STATUS.md`
- Document every experiment with actual results

### Code Quality
- Small, understandable Python modules with comments explaining ML decisions
- Each task independently runnable
- Clear separation: data loading, preprocessing, training, evaluation, visualization
- Type hints where helpful

### Git Hygiene
- Commit after each working milestone
- No raw data, `.venv/`, `__pycache__/`, secrets, or large model files in repo
- Meaningful commit messages

## Task 1 Specific Requirements

### Algorithm
- **Baseline**: LinearRegression on 3 features (GrLivArea, BedroomAbvGr, FullBath)
- **Extended**: All-feature model with ColumnTransformer (numeric + categorical)
- **Comparison**: DummyRegressor, LinearRegression, Ridge/ElasticNet (optional)

### Data Pipeline
```
Raw CSV → Data Loader → Train/Val Split → Pipeline(Imputer + Scaler) → LinearRegression → Metrics + Plots
```

### Metrics to Report
- MAE (Mean Absolute Error) - dollar scale
- RMSE (Root Mean Squared Error) - dollar scale  
- R² (Coefficient of Determination)
- Log-scale variants (Kaggle RMSLE-style)

### Visualizations
- SalePrice distribution histogram
- Actual vs Predicted scatter plot
- Residual plot (predicted vs residual)
- Save to `reports/figures/`

### Deliverables
- `models/baseline_pipeline.joblib` - saved fitted pipeline
- `reports/metrics_baseline.json` - validation metrics
- `reports/figures/` - all plots
- `docs/task1-report.md` - comprehensive report
- `app.py` - Streamlit demo

## Agent Workflow

### Before Any Implementation
1. Inspect existing codebase and data structure
2. Verify expected files exist (don't download without asking)
3. Create short implementation plan for the specific milestone
4. Wait for approval before coding

### Implementation Milestones
**Milestone A**: Data loader + audit + tests (no training)
**Milestone B**: Baseline model + validation + metrics + plots + tests
**Milestone C**: Extended model + CV comparison + error analysis
**Milestone D**: Streamlit demo + final report

### After Each Milestone
1. Run tests: `python -m pytest task1-house-price/tests/ -v`
2. Run training: `python task1-house-price/src/train.py`
3. Verify outputs exist and are sensible
4. Commit changes
5. Update `PROJECT_STATUS.md` with actual results

## Windows PowerShell Compatibility
- All terminal commands must work in Windows PowerShell
- Use `py -3.11 -m venv .venv` for venv creation
- Use `.\.venv\Scripts\Activate.ps1` for activation
- Path separators: forward slashes in Python, backslashes in shell

## Python Version
- Target: Python 3.11
- Do not modify global Python configuration
- Use virtual environment exclusively

## Forbidden Actions
- ❌ Download large datasets without explicit approval
- ❌ Invent/fake model performance numbers
- ❌ Overwrite or delete existing data files
- ❌ Add heavy dependencies without justification
- ❌ Commit data/, credentials, or model files
- ❌ Use test.csv labels for any model decision
- ❌ Skip tests or validation steps

## Communication Style
- Explain ML decisions in beginner-friendly English
- Show actual command outputs, not expected ones
- Flag assumptions and limitations clearly
- Ask before ambiguous actions