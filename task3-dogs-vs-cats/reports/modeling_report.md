# Task 3: Dogs vs. Cats Classification — HOG + SVM Modeling Report

## 1. Executive Summary
- **Selected Best Model**: `RBF Kernel SVM (C=1.0)`
- **Final Test Accuracy**: **72.50%**
- **Final Test Macro F1**: **72.50%**
- **Final Test ROC-AUC**: **0.7856**
- **Feature Dimension**: 1,764 float32 features per image
- **Dataset Split**: 2,800 Train / 600 Validation / 600 Test (Strict leak-free isolation)

---

## 2. Feature Extraction Configuration
Images are normalized to grayscale and resized to a consistent $64 \times 64$ grid before computing the Histogram of Oriented Gradients (HOG) descriptor:
- **Image Target Size**: `64 x 64`
- **Gradient Orientations**: `9` bins ($0^\circ$–$180^\circ$)
- **Pixels per Cell**: `8 x 8`
- **Cells per Block**: `2 x 2`
- **Block Normalization**: `L2-Hys` (L2-Hysteresis)
- **Square Root Variance Stabilization**: `True`
- **Verified Output Feature Length**: `1764` floats per image (~7 KB / image)

---

## 3. Validation Model Selection (Train vs. Validation)

All candidate pipelines were trained strictly on the training partition ($N=2,800$) with a `StandardScaler()` fitted exclusively on training features. Hyperparameter selection was conducted strictly on the validation partition ($N=600$):

| Model Candidate | Val Accuracy | Val Macro F1 | Val Precision | Val Recall | Val ROC-AUC | Fit Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline Dummy (Stratified)** | 50.00% | 49.98% | 50.00% | 50.00% | 0.5000 | 0.07s |
| **LinearSVM (C=0.01)** | 66.67% | 66.66% | 66.68% | 66.67% | 0.7064 | 4.41s |
| **LinearSVM (C=0.1)** | 63.83% | 63.83% | 63.84% | 63.83% | 0.6762 | 12.41s |
| **LinearSVM (C=1.0)** | 63.17% | 63.15% | 63.18% | 63.17% | 0.6658 | 33.21s |
| **RBF Kernel SVM (C=1.0)** *(Selected)* | 74.67% | 74.67% | 74.67% | 74.67% | 0.7947 | 6.06s |

---

## 4. Final Held-Out Test Evaluation

The selected model (`RBF Kernel SVM (C=1.0)`) was evaluated exactly once on the held-out test split ($N=600$, 300 cats and 300 dogs):

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Cat** | 72.28% | 73.00% | 72.64% | 300 |
| **Dog** | 72.73% | 72.00% | 72.36% | 300 |
| **Macro Average** | **72.50%** | **72.50%** | **72.50%** | 600 |

### Test Confusion Matrix:
```text
                Predicted Cat    Predicted Dog
True Cat:           219            81            
True Dog:           84             216           
```

---

## 5. Resource Consumption & RAM Footprint
- **Training Memory**: Peak memory remained under 150 MB throughout feature extraction and model fitting.
- **Inference Speed**: Single-image feature extraction and classification executes in ~2.5 ms on CPU.
- **Serialized Artifact Size**: The trained RBF kernel SVM model pipeline is 33.6 MB on disk (contains dual-coefficient and support vector float matrices for 2,400+ vectors in R^1764).

---

## 6. Generated Figures
- `reports/figures/model_comparison.png`: Comparison of candidate models on the validation split.
- `reports/figures/confusion_matrix.png`: Test set confusion matrix heatmap.
- `reports/figures/roc_curve.png`: Test set ROC curve with AUC score.
