"""
Model comparison script for Task 1: House Price Prediction.

Compares multiple models using 5-fold cross-validation on training data only:
1. DummyRegressor (median) - baseline
2. 3-feature LinearRegression - assignment baseline
3. All-feature LinearRegression - full model
4. Ridge - regularized linear model
5. ElasticNet - L1+L2 regularized linear model

All models use identical CV folds for fair comparison.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from data_loader import DataLoader, load_and_validate, BASELINE_FEATURES, TARGET_COLUMN
from preprocessing import (
    create_baseline_preprocessor,
    create_full_preprocessor,
    create_full_pipeline,
    identify_feature_types,
)
from evaluation import compute_all_metrics


# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)


def create_model_pipelines(train_df: pd.DataFrame):
    """
    Create all model pipelines for comparison.

    Args:
        train_df: Training DataFrame (used for feature type inference)

    Returns:
        Dictionary of model_name -> pipeline
    """
    # Identify feature types for full model
    numeric_features, categorical_features = identify_feature_types(train_df)

    print(f"Numeric features ({len(numeric_features)}): {numeric_features[:10]}...")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features[:10]}...")

    # 1. DummyRegressor (median) - baseline
    dummy_pipeline = Pipeline([
        ("preprocessor", create_baseline_preprocessor()),
        ("model", DummyRegressor(strategy="median"))
    ])

    # 2. 3-feature LinearRegression - assignment baseline
    baseline_lr_pipeline = Pipeline([
        ("preprocessor", create_baseline_preprocessor()),
        ("model", LinearRegression())
    ])

    # 3. All-feature LinearRegression
    full_lr_pipeline = create_full_pipeline(
        train_df, LinearRegression(), numeric_features, categorical_features
    )

    # 4. Ridge (regularized) - alpha=10.0 chosen as reasonable default
    ridge_pipeline = create_full_pipeline(
        train_df, Ridge(alpha=10.0, random_state=42), numeric_features, categorical_features
    )

    # 5. ElasticNet (L1+L2 regularization) - alpha=1.0, l1_ratio=0.5
    elastic_pipeline = create_full_pipeline(
        train_df,
        ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=10000),
        numeric_features,
        categorical_features
    )

    return {
        "DummyRegressor (median)": dummy_pipeline,
        "LinearRegression (3 features)": baseline_lr_pipeline,
        "LinearRegression (all features)": full_lr_pipeline,
        "Ridge (alpha=10)": ridge_pipeline,
        "ElasticNet (alpha=1, l1=0.5)": elastic_pipeline,
    }


def run_cross_validation(
    pipelines: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42
) -> dict:
    """
    Run cross-validation for all models using identical folds.

    Args:
        pipelines: Dictionary of model_name -> pipeline
        X: Feature DataFrame
        y: Target array
        cv_folds: Number of CV folds
        random_state: Random seed for fold splitting

    Returns:
        Dictionary of model_name -> CV results
    """
    # Create identical KFold for all models
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    cv_results = {}

    print(f"\n{'='*60}")
    print(f"RUNNING {cv_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*60}")

    for name, pipeline in pipelines.items():
        print(f"\n--- {name} ---")

        # Cross-validate with multiple metrics
        scoring = ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]

        cv_result = cross_validate(
            pipeline, X, y,
            cv=kf,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )

        # Convert to positive metrics
        cv_results[name] = {
            "CV_MAE_mean": -cv_result["test_neg_mean_absolute_error"].mean(),
            "CV_MAE_std": cv_result["test_neg_mean_absolute_error"].std(),
            "CV_RMSE_mean": -cv_result["test_neg_root_mean_squared_error"].mean(),
            "CV_RMSE_std": cv_result["test_neg_root_mean_squared_error"].std(),
            "CV_R2_mean": cv_result["test_r2"].mean(),
            "CV_R2_std": cv_result["test_r2"].std(),
            "CV_MAE_scores": [-s for s in cv_result["test_neg_mean_absolute_error"]],
            "CV_RMSE_scores": [-s for s in cv_result["test_neg_root_mean_squared_error"]],
            "CV_R2_scores": cv_result["test_r2"].tolist(),
        }

        print(f"  MAE:  {cv_results[name]['CV_MAE_mean']:,.2f} ± {cv_results[name]['CV_MAE_std']:,.2f}")
        print(f"  RMSE: {cv_results[name]['CV_RMSE_mean']:,.2f} ± {cv_results[name]['CV_RMSE_std']:,.2f}")
        print(f"  R²:   {cv_results[name]['CV_R2_mean']:.4f} ± {cv_results[name]['CV_R2_std']:.4f}")

    return cv_results


def evaluate_on_validation(
    pipelines: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray
) -> dict:
    """
    Fit all models on full training set and evaluate on validation set.

    Args:
        pipelines: Dictionary of model_name -> pipeline
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target

    Returns:
        Dictionary of model_name -> validation metrics
    """
    val_results = {}

    print(f"\n{'='*60}")
    print("EVALUATION ON HELD-OUT VALIDATION SET")
    print(f"{'='*60}")

    for name, pipeline in pipelines.items():
        print(f"\n--- {name} ---")

        # Fit on full training data
        pipeline.fit(X_train, y_train)

        # Predict on validation
        y_pred = pipeline.predict(X_val)

        # Compute all metrics (dollar and log scale)
        metrics = compute_all_metrics(y_val, y_pred, prefix="Val_")

        val_results[name] = metrics

        print(f"  Val_MAE:  ${metrics['Val_MAE']:,.2f}")
        print(f"  Val_RMSE: ${metrics['Val_RMSE']:,.2f}")
        print(f"  Val_R²:   {metrics['Val_R2']:.4f}")
        print(f"  Val_Log_MAE:  {metrics['Val_Log_MAE']:.4f}")
        print(f"  Val_Log_RMSE: {metrics['Val_Log_RMSE']:.4f}")
        print(f"  Val_Log_R²:   {metrics['Val_Log_R2']:.4f}")

    return val_results


def save_comparison_results(
    cv_results: dict,
    val_results: dict,
    output_dir: Path
):
    """
    Save comparison results to JSON and CSV.

    Args:
        cv_results: Cross-validation results
        val_results: Validation set results
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare summary for JSON
    summary = {
        "cross_validation": {},
        "validation_set": {},
        "best_model_cv_rmse": None,
        "best_model_val_rmse": None,
    }

    # CV summary
    for name, res in cv_results.items():
        summary["cross_validation"][name] = {
            "MAE_mean": res["CV_MAE_mean"],
            "MAE_std": res["CV_MAE_std"],
            "RMSE_mean": res["CV_RMSE_mean"],
            "RMSE_std": res["CV_RMSE_std"],
            "R2_mean": res["CV_R2_mean"],
            "R2_std": res["CV_R2_std"],
        }

    # Validation summary
    for name, res in val_results.items():
        summary["validation_set"][name] = {k: v for k, v in res.items()}

    # Find best models
    best_cv = min(cv_results.items(), key=lambda x: x[1]["CV_RMSE_mean"])
    best_val = min(val_results.items(), key=lambda x: x[1]["Val_RMSE"])

    summary["best_model_cv_rmse"] = {
        "model": best_cv[0],
        "CV_RMSE_mean": best_cv[1]["CV_RMSE_mean"],
        "CV_RMSE_std": best_cv[1]["CV_RMSE_std"],
    }
    summary["best_model_val_rmse"] = {
        "model": best_val[0],
        "Val_RMSE": best_val[1]["Val_RMSE"],
    }

    # Save JSON
    json_path = output_dir / "model_comparison.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n✓ Comparison JSON saved: {json_path}")

    # Save CSV summary table
    csv_rows = []
    for name in cv_results:
        cv = cv_results[name]
        val = val_results[name]
        csv_rows.append({
            "Model": name,
            "CV_MAE_mean": cv["CV_MAE_mean"],
            "CV_MAE_std": cv["CV_MAE_std"],
            "CV_RMSE_mean": cv["CV_RMSE_mean"],
            "CV_RMSE_std": cv["CV_RMSE_std"],
            "CV_R2_mean": cv["CV_R2_mean"],
            "CV_R2_std": cv["CV_R2_std"],
            "Val_MAE": val["Val_MAE"],
            "Val_RMSE": val["Val_RMSE"],
            "Val_R2": val["Val_R2"],
            "Val_Log_MAE": val["Val_Log_MAE"],
            "Val_Log_RMSE": val["Val_Log_RMSE"],
            "Val_Log_R2": val["Val_Log_R2"],
        })

    df_summary = pd.DataFrame(csv_rows)
    csv_path = output_dir / "model_comparison.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"✓ Comparison CSV saved: {csv_path}")

    return summary


def print_comparison_table(cv_results: dict, val_results: dict):
    """Print formatted comparison table."""
    print(f"\n{'='*100}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*100}")

    # CV Results Table
    print("\n--- 5-Fold Cross-Validation Results ---")
    print(f"{'Model':<35} {'MAE':>15} {'RMSE':>15} {'R²':>10}")
    print("-" * 75)
    for name, res in cv_results.items():
        print(f"{name:<35} ${res['CV_MAE_mean']:>12,.0f}±{res['CV_MAE_std']:<4,.0f} "
              f"${res['CV_RMSE_mean']:>12,.0f}±{res['CV_RMSE_std']:<4,.0f} "
              f"{res['CV_R2_mean']:>8.4f}±{res['CV_R2_std']:<5.4f}")

    # Validation Results Table
    print("\n--- Held-Out Validation Set Results ---")
    print(f"{'Model':<35} {'MAE':>15} {'RMSE':>15} {'R²':>10} {'Log_RMSE':>12}")
    print("-" * 87)
    for name, res in val_results.items():
        print(f"{name:<35} ${res['Val_MAE']:>12,.0f} "
              f"${res['Val_RMSE']:>12,.0f} "
              f"{res['Val_R2']:>8.4f} "
              f"{res['Val_Log_RMSE']:>10.4f}")

    # Best models
    best_cv = min(cv_results.items(), key=lambda x: x[1]["CV_RMSE_mean"])
    best_val = min(val_results.items(), key=lambda x: x[1]["Val_RMSE"])

    print(f"\n🏆 Best CV RMSE: {best_cv[0]} (${best_cv[1]['CV_RMSE_mean']:,.0f} ± {best_cv[1]['CV_RMSE_std']:,.0f})")
    print(f"🏆 Best Val RMSE: {best_val[0]} (${best_val[1]['Val_RMSE']:,.0f})")


def main():
    parser = argparse.ArgumentParser(description="Compare models for House Price Prediction")
    parser.add_argument("--data-dir", default="data/raw", help="Path to data directory")
    parser.add_argument("--output-dir", default=".", help="Output directory for results")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--log-target", action="store_true", help="Use log-transformed target")
    args = parser.parse_args()

    print("=" * 60)
    print("TASK 1: MODEL COMPARISON - EXTENDED ALL-FEATURES")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Validation split: {args.val_split}")
    print(f"Random state: {args.random_state}")
    print(f"CV folds: {args.cv_folds}")
    print(f"Log target: {args.log_target}")

    # Load data
    print("\n📂 Loading data...")
    train_df, test_df = load_and_validate(args.data_dir)

    # Prepare features and target
    X = train_df.drop(columns=[TARGET_COLUMN, "Id"])
    y = train_df[TARGET_COLUMN].values

    if args.log_target:
        y = np.log1p(y)
        print("Using log-transformed target (log1p)")

    # Train/validation split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, random_state=args.random_state
    )
    print(f"\nTrain: {len(X_train)} samples, Validation: {len(X_val)} samples")

    # Create model pipelines
    print("\n🔧 Creating model pipelines...")
    pipelines = create_model_pipelines(X_train)

    # Run cross-validation
    cv_results = run_cross_validation(
        pipelines, X_train, y_train,
        cv_folds=args.cv_folds,
        random_state=args.random_state
    )

    # Evaluate on validation set
    val_results = evaluate_on_validation(
        pipelines, X_train, y_train, X_val, y_val
    )

    # Print comparison table
    print_comparison_table(cv_results, val_results)

    # Save results
    output_dir = Path(args.output_dir)
    save_comparison_results(cv_results, val_results, output_dir / "reports")

    print(f"\n{'='*60}")
    print("MODEL COMPARISON COMPLETE")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    exit(main())