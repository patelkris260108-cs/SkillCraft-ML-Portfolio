# SkillCraft ML Portfolio

Machine Learning internship projects for **SkillCraft Technology**.

## Tasks Overview

| Task | Problem Type | Algorithm | Dataset | Status |
|------|--------------|-----------|---------|--------|
| 1. House Price Prediction | Supervised Regression | Linear Regression | [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) | 🚧 In Progress |
| 2. Mall Customer Segmentation | Unsupervised Clustering | K-Means | [Kaggle Mall Customers](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) | ⏳ Planned |
| 3. Dogs vs Cats Classification | Binary Image Classification | SVM + HOG | [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats) | ⏳ Planned |
| 4. Hand Gesture Recognition | Multi-class Image Classification | CNN | [Kaggle LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) | ⏳ Planned |

## Quick Start (Task 1)

### Prerequisites
- Python 3.11
- Windows PowerShell (or compatible shell)

### Setup
```powershell
# Clone and navigate
cd SkillCraft-ML-Portfolio

# Create virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Download Dataset (Task 1)
```powershell
# Option A: Using kagglehub (recommended)
python -c "import kagglehub; path = kagglehub.competition_download('house-prices-advanced-regression-techniques'); print('Downloaded to:', path)"
# Then copy files to task1-house-price/data/raw/

# Option B: Manual download from Kaggle website
# 1. Go to https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# 2. Accept competition rules
# 3. Download train.csv, test.csv, data_description.txt, sample_submission.csv
# 4. Place in task1-house-price/data/raw/
```

### Run Baseline Training
```powershell
# From project root
python task1-house-price/src/train.py
```

### Run Tests
```powershell
python -m pytest task1-house-price/tests/ -v
```

### Launch Streamlit Demo
```powershell
streamlit run task1-house-price/app.py
```

## Project Structure

```
SkillCraft-ML-Portfolio/
├── AGENTS.md                 # Agent instructions
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── PROJECT_STATUS.md        # Implementation log
│
├── task1-house-price/
│   ├── data/
│   │   └── raw/             # train.csv, test.csv, data_description.txt, sample_submission.csv
│   ├── src/
│   │   ├── train.py         # Main training script
│   │   ├── predict.py       # Prediction utilities
│   │   ├── data_loader.py   # Data loading & validation
│   │   ├── preprocessing.py # Preprocessing pipelines
│   │   ├── evaluation.py    # Metrics computation
│   │   └── visualization.py # Plotting functions
│   ├── models/              # Saved model artifacts (.joblib)
│   ├── reports/
│   │   └── figures/         # Generated plots
│   ├── tests/               # Unit tests
│   └── app.py               # Streamlit demo
│
├── task2-customer-segmentation/
├── task3-dogs-vs-cats/
├── task4-hand-gesture/
│
└── docs/
    └── task1-report.md      # Final report
```

## Key Principles

1. **No fabricated results** - All metrics from actual runs
2. **Leakage prevention** - Pipeline fits only on training data
3. **Reproducibility** - Fixed seeds, saved artifacts, documented commands
4. **Clean separation** - Each task independently runnable
5. **Git hygiene** - No data, venv, or secrets in repo

## Task 1 Details

### Baseline Features (Assignment Requirement)
- `GrLivArea` - Above ground living area (square footage)
- `BedroomAbvGr` - Bedrooms above grade
- `FullBath` - Full bathrooms above grade
- Target: `SalePrice`

### Extended Features (All-Feature Model)
All 79 explanatory variables from Ames Housing dataset with proper numeric/categorical preprocessing.

### Evaluation Metrics
- **MAE** - Average absolute dollar error
- **RMSE** - Penalizes large errors
- **R²** - Variance explained
- **Log-scale RMSE** - Kaggle competition metric (RMSLE)

## License & Data Rights

- **Code**: MIT License
- **Datasets**: Subject to Kaggle Terms of Use and individual competition/dataset licenses
- **Ames Housing Data**: Original source - Dean De Cock (2011), used under Kaggle competition terms
- **Predictions**: Educational estimates only - not verified market prices for any real estate market

## References

- [Kaggle House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- [Ames Housing Dataset Description](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
- [SkillCraft Technology](https://skillcrafttech.com)
- [Google Antigravity Docs](https://antigravity.google/docs/)

## Contributing

This is an internship portfolio project. For questions about the implementation, refer to `AGENTS.md` or the task-specific documentation in `docs/`.