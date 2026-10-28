# Mall Customer Segmentation — Exploratory Data Analysis (EDA) Summary

## 1. Dataset Dimensions & Health Check
- **Total Customer Records**: `200`
- **Total Attributes**: `5`
- **Attributes**: `CustomerID, Gender, Age, Annual_Income, Spending_Score`
- **Duplicate Rows**: `0` (clean, zero redundancy)
- **Missing / Null Values**: Zero missing values across all columns

| Column | Type | Missing Count | Valid Range / Categories |
| :--- | :--- | :--- | :--- |
| `CustomerID` | `int64` | 0 | 1 to 200 (Sequential ID) |
| `Gender` | `str` | 0 | {Female, Male} |
| `Age` | `int64` | 0 | 18 to 70 years |
| `Annual_Income` | `int64` | 0 | $15k to $137k |
| `Spending_Score` | `int64` | 0 | 1 to 99 (Scale: 1-100) |

## 2. Descriptive Statistics (Numeric Features)

| Statistic | Age (years) | Annual Income (k$) | Spending Score (1-100) |
| :--- | :---: | :---: | :---: |
| **Count** | 200 | 200 | 200 |
| **Mean** | 38.85 | 60.56 | 50.20 |
| **Std Dev** | 13.97 | 26.26 | 25.82 |
| **Min** | 18.0 | 15.0 | 1.0 |
| **25% (Q1)** | 28.8 | 41.5 | 34.8 |
| **50% (Median)** | 36.0 | 61.5 | 50.0 |
| **75% (Q3)** | 49.0 | 78.0 | 73.0 |
| **Max** | 70.0 | 137.0 | 99.0 |
| **Skewness** | 0.49 | 0.32 | -0.05 |
| **Kurtosis** | -0.67 | -0.10 | -0.83 |

## 3. Demographic & Behavioral Insights by Gender

- **Female Customers**: `112` (56.0%)
- **Male Customers**: `88` (44.0%)

| Gender | Average Age | Average Income (k$) | Average Spending Score |
| :--- | :---: | :---: | :---: |
| **Female** | 38.1 yrs | $59.2k | 51.5 |
| **Male** | 39.8 yrs | $62.2k | 48.5 |

## 4. Key EDA Observations
1. **Bivariate Natural Groupings**: The scatter plot of `Annual_Income` vs. `Spending_Score` clearly displays 5 distinct visual clusters:
   - Low Income, Low Spending (Frugal / Budget)
   - Low Income, High Spending (Carefree / Impulsive)
   - Middle Income, Middle Spending (Average / Balanced majority)
   - High Income, Low Spending (Conservative / Savers)
   - High Income, High Spending (Affluent / Target Luxury Shoppers)
2. **Age Threshold in Spending**: Customers under 40 display both high and low spending scores (reaching up to 99), whereas customers above 40 have spending scores consistently capped below 60.
3. **Income and Spending Symmetry**: Spending scores exhibit near-zero skewness (-0.04), indicating a symmetric, well-balanced distribution centered around a median of 50. Annual income exhibits a slight right skew (0.32) with a top earner at $137k.
4. **No Premature Cluster Assignments**: All visual observations reflect empirical data distributions only. Clustering models (K-Means, optimal k evaluation) will be formally executed in Phase 3.

## 5. Generated Visual Artifacts
- `reports/figures/age_distribution.png`: Age Distribution
- `reports/figures/annual_income_distribution.png`: Annual Income Distribution
- `reports/figures/spending_score_distribution.png`: Spending Score Distribution
- `reports/figures/gender_distribution.png`: Gender Distribution
- `reports/figures/income_vs_spending.png`: Income Vs Spending
- `reports/figures/age_vs_spending.png`: Age Vs Spending
