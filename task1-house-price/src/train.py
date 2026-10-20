"""
Main training script for Task 1: House Price Prediction.

This script implements the baseline Linear Regression model using
3 features (GrLivArea, BedroomAbvGr, FullBath) as required by
the SkillCraft internship assignment.

Usage:
    python train.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DataLoader, BASELINE_FEATURES, TARGET_COLUMN
from preprocessing import create_baseline_preprocessor
from evaluation import compute_all_metrics, save_metrics, print_metrics
from visualization import (
    plot_target_distribution,
    plot_actual_vs_predicted,
    plot_residuals,
    generate_all_diagnostic_plots,
    PLOT_DIR
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train baseline Linear Regression for House Price Prediction"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing train.csv, test.csv, etc."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for models and reports"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split fraction (default: 0.2)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--log-target",
        action="store_true",
        help="Train on log-transformed target (log1p)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating diagnostic plots"
    )
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()

    # Convert to Path objects
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"

    # Create output directories
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TASK 1: HOUSE PRICE PREDICTION - BASELINE TRAINING")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Validation split: {args.test_size}")
    print(f"Random state: {args.random_state}")
    print(f"Log target: {args.log_target}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Load and validate data
    # ------------------------------------------------------------------
    print("Step 1: Loading and validating data...")
    loader = DataLoader(str(data_dir))

    # Check files exist
    missing = loader.validate_files_exist()
    if missing:
        print(f"✗ Missing required files: {missing}")
        print("Please download the Kaggle House Prices dataset first.")
        print("Run: python -c \"import kagglehub; kagglehub.competition_download('house-prices-advanced-regression-techniques')\"")
        sys.exit(1)

    # Load data
    train_df = loader.load_train(validate=True)
    test_df = loader.load_test(validate=True)

    print(f"✓ Loaded train: {train_df.shape}, test: {test_df.shape}")

    # ------------------------------------------------------------------
    # Step 2: Prepare features and target
    # ------------------------------------------------------------------
    print("\nStep 2: Preparing features and target...")

    # Extract baseline features
    X = loader.get_baseline_features(train_df)
    y = loader.get_target(train_df)

    print(f"Features: {list(X.columns)}")
    print(f"Target: {TARGET_COLUMN}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Check for missing values in features
    missing_features = X.isna().sum()
    if missing_features.any():
        print(f"⚠ Missing values in features:\n{missing_features[missing_features > 0]}")

    # ------------------------------------------------------------------
    # Step 3: Train/Validation split
    # ------------------------------------------------------------------
    print("\nStep 3: Creating train/validation split...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state
    )

    print(f"Train: {X_train.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples")

    # Save split indices for reproducibility
    split_info = {
        "train_indices": X_train.index.tolist(),
        "val_indices": X_val.index.tolist(),
        "test_size": args.test_size,
        "random_state": args.random_state
    }
    import json
    with open(reports_dir / "split_indices.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # ------------------------------------------------------------------
    # Step 4: Create and fit preprocessing pipeline
    # ------------------------------------------------------------------
    print("\nStep 4: Creating preprocessing pipeline...")
    preprocessor = create_baseline_preprocessor()

    # Fit on training data ONLY (prevents leakage)
    print("Fitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    print(f"Processed train shape: {X_train_processed.shape}")
    print(f"Processed val shape: {X_val_processed.shape}")

    # ------------------------------------------------------------------
    # Step 5: Prepare target (optional log transform)
    # ------------------------------------------------------------------
    if args.log_target:
        print("\nStep 5: Applying log1p transform to target...")
        y_train_transformed = np.log1p(y_train)
        y_val_transformed = np.log1p(y_val)
        target_transform = "log1p"
    else:
        y_train_transformed = y_train.values
        y_val_transformed = y_val.values
        target_transform = "none"

    # ------------------------------------------------------------------
    # Step 6: Train Linear Regression
    # ------------------------------------------------------------------
    print("\nStep 6: Training Linear Regression...")
    model = LinearRegression()
    model.fit(X_train_processed, y_train_transformed)

    print(f"✓ Model trained")
    print(f"  Intercept: {model.intercept_:.4f}")
    print(f"  Coefficients: {dict(zip(BASELINE_FEATURES, model.coef_))}")

    # ------------------------------------------------------------------
    # Step 7: Predict and evaluate
    # ------------------------------------------------------------------
    print("\nStep 7: Evaluating on validation set...")

    # Predict on validation set
    y_val_pred_transformed = model.predict(X_val_processed)

    # Inverse transform if needed
    if args.log_target:
        y_val_pred = np.expm1(y_val_pred_transformed)
        # Ensure positive predictions
        y_val_pred = np.maximum(y_val_pred, 0)
    else:
        y_val_pred = y_val_pred_transformed

    # Compute metrics
    metrics = compute_all_metrics(y_val.values, y_val_pred, prefix="Val_")

    # Add metadata
    metrics["model"] = "LinearRegression"
    metrics["features"] = BASELINE_FEATURES
    metrics["target_transform"] = target_transform
    metrics["train_size"] = int(X_train.shape[0])
    metrics["val_size"] = int(X_val.shape[0])
    metrics["random_state"] = args.random_state

    print_metrics(metrics, "VALIDATION METRICS")

    # ------------------------------------------------------------------
    # Step 8: Save model and metrics
    # ------------------------------------------------------------------
    print("\nStep 8: Saving model and metrics...")

    # Save full pipeline (preprocessor + model)
    from sklearn.pipeline import Pipeline
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    model_path = models_dir / "baseline_pipeline.joblib"
    joblib.dump(full_pipeline, model_path)
    print(f"✓ Model saved: {model_path}")

    # Save metrics
    metrics_path = reports_dir / "metrics_baseline.json"
    save_metrics(metrics, str(metrics_path))

    # Save coefficients for interpretation
    coef_df = pd.DataFrame({
        "feature": ["intercept"] + BASELINE_FEATURES,
        "coefficient": [model.intercept_] + list(model.coef_)
    })
    coef_path = reports_dir / "coefficients_baseline.csv"
    coef_df.to_csv(coef_path, index=False)
    print(f"✓ Coefficients saved: {coef_path}")

    # ------------------------------------------------------------------
    # Step 9: Generate diagnostic plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        print("\nStep 9: Generating diagnostic plots...")
        generate_all_diagnostic_plots(
            y_true=y_val.values,
            y_pred=y_val_pred,
            X=X_val_processed,
            feature_names=BASELINE_FEATURES,
            model_name="Baseline_LinearRegression"
        )
    else:
        print("\nStep 9: Skipping plots (--no-plots flag)")

    # ------------------------------------------------------------------
    # Step 10: Quick test prediction
    # ------------------------------------------------------------------
    print("\nStep 10: Testing prediction on sample...")
    sample_idx = 0
    sample_X = X_val.iloc[[sample_idx]]
    sample_y_true = y_val.iloc[sample_idx]

    sample_processed = preprocessor.transform(sample_X)
    sample_pred_transformed = model.predict(sample_processed)

    if args.log_target:
        sample_pred = np.expm1(sample_pred_transformed)[0]
    else:
        sample_pred = sample_pred_transformed[0]

    print(f"Sample input: {sample_X.iloc[0].to_dict()}")
    print(f"True price: ${sample_y_true:,.2f}")
    print(f"Predicted:  ${sample_pred:,.2f}")
    print(f"Error:      ${abs(sample_y_true - sample_pred):,.2f}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Plots: {PLOT_DIR}")
    print("\nNext steps:")
    print("  1. Review metrics and plots in reports/")
    print("  2. Run tests: python -m pytest tests/ -v")
    print("  3. Launch demo: streamlit run app.py")
    print("  4. Extend to all-feature model (see preprocessing.py)")

    return 0


if __name__ == "__main__":
    exit(main())