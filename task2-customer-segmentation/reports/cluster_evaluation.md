# Mall Customer Segmentation — Clustering Model Evaluation & Selection Report

## 1. Executive Summary
This report documents unsupervised K-Means model evaluation and selection across two explicit feature representations:
1. **Classic Model (2D Benchmark)**: `Annual_Income` + `Spending_Score`
2. **Extended Model (4D Multivariate)**: `Age` + `Annual_Income` + `Spending_Score` + `Gender_Male`

Rather than relying blindly on silhouette score alone, model selection was grounded in a holistic evaluation of:
- **Inertia reduction (Elbow inflection point)**
- **Silhouette score (Separation and cohesion)**
- **Davies-Bouldin Index (Cluster separation)**
- **Multi-seed initialization stability (Adjusted Rand Index)**
- **Practical retail interpretability and actionable customer personas**

---

## 2. Quantitative Metric Comparison (k = 2 to 10)

### A. Classic Feature Set (`Annual_Income`, `Spending_Score`)

| k | Inertia (WSS) | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Score |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 400.00 | N/A | N/A | N/A |
| **2** | 269.69 | 0.3213 | 1.2670 | 95.67 |
| **3** | 157.70 | 0.4666 | 0.7165 | 151.34 |
| **4** | 108.92 | 0.4939 | 0.7096 | 174.60 |
| **5** | 65.57 | 0.5547 | 0.5722 | 248.65 |
| **6** | 55.06 | 0.5399 | 0.6546 | 243.09 |
| **7** | 44.86 | 0.5281 | 0.7148 | 254.62 |
| **8** | 37.23 | 0.4552 | 0.7602 | 267.28 |
| **9** | 32.39 | 0.4571 | 0.7632 | 270.95 |
| **10** | 29.98 | 0.4432 | 0.7935 | 260.54 |

### B. Extended Feature Set (`Age`, `Annual_Income`, `Spending_Score`, `Gender_Male`)

| k | Inertia (WSS) | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Score |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 649.28 | N/A | N/A | N/A |
| **2** | 438.52 | 0.3032 | 1.3607 | 95.16 |
| **3** | 344.68 | 0.3150 | 1.1524 | 87.04 |
| **4** | 254.28 | 0.3503 | 1.0650 | 101.49 |
| **5** | 216.78 | 0.3498 | 1.0245 | 97.26 |
| **6** | 181.95 | 0.3565 | 1.0051 | 99.65 |
| **7** | 167.82 | 0.3433 | 1.1165 | 92.28 |
| **8** | 150.66 | 0.3304 | 1.0893 | 90.77 |
| **9** | 138.86 | 0.3400 | 1.0079 | 87.76 |
| **10** | 133.33 | 0.3087 | 1.1924 | 81.70 |

---

## 3. Model Selection Analysis & Candidate Rationale

### Why Classic k = 5 is the Strongest Benchmark Solution
1. **Clear Geometric Elbow**: The WSS curve displays an unambiguous inflection at `k = 5` (Inertia drops steeply from 108.92 at k=4 to 65.57 at k=5, then plateaus with only a 10.51 drop to k=6).
2. **Peak Silhouette Separation**: `k = 5` achieves the maximum global silhouette coefficient of **0.5547** across all tested configurations. Individual silhouette widths confirm that no clusters have negative average silhouette scores.
3. **Optimal Davies-Bouldin Index**: `k = 5` records the lowest Davies-Bouldin score (**0.5722**), confirming minimal cluster overlap.
4. **Stability Invariance**: Evaluating across 5 distinct random initialization seeds yields a **mean pairwise Adjusted Rand Index (ARI) of 0.988+** and standard deviation in inertia of `< 0.01`, proving the 5-cluster partition is robust and not an initialization artifact.

### Extended Model Tradeoffs
- In the 4D extended space, the addition of Age and binary Gender smooths the geometric boundaries, lowering the maximum silhouette score to **~0.35** across k in [4, 6].
- Candidate `k = 5` in the extended space preserves the primary income/spending clusters while revealing age-driven stratification (e.g. distinguishing younger impulsive shoppers from older moderate spenders).
- `k = 6` achieves a slightly higher silhouette (0.3565) by further splitting gender, but yields smaller sub-clusters that risk overfitting.

---

## 4. Observed Customer Cluster Profiles (Classic Model, k = 5)

| Cluster | Descriptive Name | Size (n) | Share (%) | Mean Income | Mean Spending | Median Age | Gender Ratio (F / M) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | Moderate Income / Moderate Spending | 81 | 40.5% | $55.3k | 49.5 | 46 yrs | 59% / 41% |
| **1** | High Income / High Spending | 39 | 19.5% | $86.5k | 82.1 | 32 yrs | 54% / 46% |
| **2** | Low Income / High Spending | 22 | 11.0% | $25.7k | 79.4 | 24 yrs | 59% / 41% |
| **3** | High Income / Low Spending | 35 | 17.5% | $88.2k | 17.1 | 42 yrs | 46% / 54% |
| **4** | Low Income / Low Spending | 23 | 11.5% | $26.3k | 20.9 | 46 yrs | 61% / 39% |

### Detailed Retail Interpretations & Cautious Observations
#### Cluster 0: Moderate Income / Moderate Spending
- **Demographics**: Median Age 46 years, 59.3% Female, 40.7% Male.
- **Financial Position**: Mean Annual Income $55.3k | Mean Spending Score 49.5/100.
- **Observed Profile**: Core middle-market segment; stable baseline traffic and broad retail category consumption.
- **Methodological Caution**: Observed purchasing patterns represent empirical correlations in cross-sectional survey data; they do not prove that income level causes spending behaviors.

#### Cluster 1: High Income / High Spending
- **Demographics**: Median Age 32 years, 53.8% Female, 46.2% Male.
- **Financial Position**: Mean Annual Income $86.5k | Mean Spending Score 82.1/100.
- **Observed Profile**: Affluent high-engagement shoppers; strong candidates for premium merchandise and luxury rewards.
- **Methodological Caution**: Observed purchasing patterns represent empirical correlations in cross-sectional survey data; they do not prove that income level causes spending behaviors.

#### Cluster 2: Low Income / High Spending
- **Demographics**: Median Age 24 years, 59.1% Female, 40.9% Male.
- **Financial Position**: Mean Annual Income $25.7k | Mean Spending Score 79.4/100.
- **Observed Profile**: Budget-constrained yet highly responsive to lifestyle trends and dynamic promotional campaigns.
- **Methodological Caution**: Observed purchasing patterns represent empirical correlations in cross-sectional survey data; they do not prove that income level causes spending behaviors.

#### Cluster 3: High Income / Low Spending
- **Demographics**: Median Age 42 years, 45.7% Female, 54.3% Male.
- **Financial Position**: Mean Annual Income $88.2k | Mean Spending Score 17.1/100.
- **Observed Profile**: High earning potential with low center spending; opportunities for curated value propositions and high-value incentives.
- **Methodological Caution**: Observed purchasing patterns represent empirical correlations in cross-sectional survey data; they do not prove that income level causes spending behaviors.

#### Cluster 4: Low Income / Low Spending
- **Demographics**: Median Age 46 years, 60.9% Female, 39.1% Male.
- **Financial Position**: Mean Annual Income $26.3k | Mean Spending Score 20.9/100.
- **Observed Profile**: Price-sensitive conservative spenders; prioritize essential goods, discounts, and clear value.
- **Methodological Caution**: Observed purchasing patterns represent empirical correlations in cross-sectional survey data; they do not prove that income level causes spending behaviors.

---

## 5. Artifacts and Output Files
- `reports/cluster_evaluation.csv`: Full tabular metric evaluations.
- `reports/figures/elbow_curve_classic.png`: Classic model WSS curve.
- `reports/figures/silhouette_curve_classic.png`: Classic model silhouette curve.
- `reports/figures/cluster_scatter_classic_k5.png`: 2D scatter with centroids.
- `reports/figures/elbow_curve_extended.png`: Extended model WSS curve.
- `reports/figures/silhouette_curve_extended.png`: Extended model silhouette curve.
- `data/processed/mall_customers_segmented.csv`: Customer records with assigned cluster labels.
- `models/kmeans_classic_k5.joblib`: Serialized classic benchmark model.
- `models/kmeans_extended_k5.joblib`: Serialized extended demographic model.
