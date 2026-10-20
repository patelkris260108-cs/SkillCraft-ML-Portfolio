"""
Evaluation metrics and utilities for Task 1: House Price Prediction.

This module provides functions to compute regression metrics,
compare models, and format results for reporting.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.base import RegressorMixin


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute standard regression metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        prefix: Optional prefix for metric names

    Returns:
        Dictionary of metrics
    """
    # Ensure arrays are 1D
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Check for valid values
    if len(y_true) != len(y_pred):
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("NaN values found in y_true or y_pred")

    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Infinite values found in y_true or y_pred")

    metrics = {}

    # Mean Absolute Error (dollar scale)
    mae = mean_absolute_error(y_true, y_pred)
    metrics[f"{prefix}MAE"] = float(mae)

    # Root Mean Squared Error (dollar scale)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics[f"{prefix}RMSE"] = float(rmse)

    # R-squared
    r2 = r2_score(y_true, y_pred)
    metrics[f"{prefix}R2"] = float(r2)

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))) * 100
    metrics[f"{prefix}MAPE"] = float(mape)

    # Median Absolute Error (robust)
    medae = np.median(np.abs(y_true - y_pred))
    metrics[f"{prefix}MedAE"] = float(medae)

    return metrics


def compute_log_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute metrics on log-transformed scale (Kaggle RMSLE style).

    Args:
        y_true: Ground truth values (original scale)
        y_pred: Predicted values (original scale)
        prefix: Optional prefix for metric names

    Returns:
        Dictionary of log-scale metrics
    """
    # Apply log1p transform
    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(y_pred)

    return compute_regression_metrics(y_true_log, y_pred_log, prefix=f"{prefix}Log_")


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute both dollar-scale and log-scale metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        prefix: Optional prefix

    Returns:
        Combined metrics dictionary
    """
    metrics = {}
    metrics.update(compute_regression_metrics(y_true, y_pred, prefix))
    metrics.update(compute_log_metrics(y_true, y_pred, prefix))
    return metrics


def cross_validate_model(
    estimator: RegressorMixin,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
    random_state: int = 42
) -> Dict[str, float]:
    """
    Perform cross-validation and return mean/std metrics.

    Args:
        estimator: sklearn estimator (unfitted)
        X: Feature matrix
        y: Target values
        cv: Number of folds
        scoring: Scoring metric
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with mean and std for each metric
    """
    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Cross-validation for multiple metrics
    scoring_metrics = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2"
    }

    results = {}
    for name, score in scoring_metrics.items():
        scores = cross_val_score(estimator, X, y, cv=kfold, scoring=score, n_jobs=-1)
        if score.startswith("neg_"):
            scores = -scores  # Convert to positive
        results[f"CV_{name}_mean"] = float(np.mean(scores))
        results[f"CV_{name}_std"] = float(np.std(scores))
        results[f"CV_{name}_scores"] = scores.tolist()

    return results


def compare_models(
    models: Dict[str, RegressorMixin],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cv: int = 5
) -> pd.DataFrame:
    """
    Compare multiple models on validation set and cross-validation.

    Args:
        models: Dictionary of {name: estimator}
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        cv: CV folds

    Returns:
        DataFrame with comparison results
    """
    results = []

    for name, estimator in models.items():
        print(f"Evaluating {name}...")

        # Fit on training data
        estimator.fit(X_train, y_train)

        # Validation predictions
        y_val_pred = estimator.predict(X_val)

        # Compute metrics
        val_metrics = compute_all_metrics(y_val, y_val_pred, prefix="Val_")

        # Cross-validation
        cv_metrics = cross_validate_model(estimator, X_train, y_train, cv=cv)

        # Combine
        row = {"Model": name}
        row.update(val_metrics)
        row.update(cv_metrics)
        results.append(row)

    return pd.DataFrame(results)


def save_metrics(metrics: Dict, filepath: str) -> None:
    """
    Save metrics dictionary to JSON file.

    Args:
        metrics: Metrics dictionary
        filepath: Output path
    """
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    metrics_serializable = convert(metrics)

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics_serializable, f, indent=2)

    print(f"Metrics saved to {filepath}")


def load_metrics(filepath: str) -> Dict:
    """
    Load metrics from JSON file.

    Args:
        filepath: Path to metrics file

    Returns:
        Metrics dictionary
    """
    with open(filepath, "r") as f:
        return json.load(f)


def print_metrics(metrics: Dict, title: str = "Metrics") -> None:
    """
    Pretty print metrics dictionary.

    Args:
        metrics: Metrics dictionary
        title: Title to display
    """
    print(f"\n{'=' * 50}")
    print(f"{title}")
    print(f"{'=' * 50}")

    # Group by prefix
    groups = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            # Determine group
            if key.startswith("Val_"):
                group = "Validation"
            elif key.startswith("CV_"):
                group = "Cross-Validation"
            elif key.startswith("Log_"):
                group = "Log-Scale"
            else:
                group = "Dollar-Scale"

            if group not in groups:
                groups[group] = {}
            groups[group][key] = value

    for group, group_metrics in groups.items():
        print(f"\n{group}:")
        for key, value in sorted(group_metrics.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


def format_metrics_table(metrics_dict: Dict[str, Dict]) -> str:
    """
    Format multiple model metrics as a markdown table.

    Args:
        metrics_dict: Dictionary of {model_name: metrics_dict}

    Returns:
        Markdown table string
    """
    if not metrics_dict:
        return ""

    # Get all unique metric keys
    all_keys = set()
    for m in metrics_dict.values():
        all_keys.update(m.keys())

    # Sort keys
    metric_keys = sorted(all_keys)

    # Build table
    lines = []
    lines.append("| Model | " + " | ".join(metric_keys) + " |")
    lines.append("| " + " | ".join(["---"] * (len(metric_keys) + 1)) + " |")

    for model_name, metrics in metrics_dict.items():
        row = [model_name]
        for key in metric_keys:
            val = metrics.get(key, "N/A")
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    print("Testing evaluation metrics...")

    # Generate sample data
    np.random.seed(42)
    y_true = np.random.lognormal(mean=12, sigma=0.5, size=100)
    y_pred = y_true * np.random.lognormal(mean=0, sigma=0.1, size=100)

    # Test metrics
    metrics = compute_all_metrics(y_true, y_pred, prefix="Test_")
    print_metrics(metrics)

    # Test log metrics
    log_metrics = compute_log_metrics(y_true, y_pred)
    print(f"\nLog RMSE: {log_metrics.get('Log_RMSE', 'N/A'):.4f}")

    print("\n✓ Evaluation module test passed!")