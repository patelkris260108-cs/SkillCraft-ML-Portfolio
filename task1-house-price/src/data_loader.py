"""
Data loading and validation module for Task 1: House Price Prediction.

This module handles loading the Kaggle House Prices dataset, validating
column names, checking for data quality issues, and providing clean
DataFrames for downstream processing.
"""

from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np


# Expected column names for the baseline features (assignment requirement)
BASELINE_FEATURES = [
    "GrLivArea",    # Above ground living area (square footage)
    "BedroomAbvGr", # Bedrooms above grade
    "FullBath"      # Full bathrooms above grade
]

TARGET_COLUMN = "SalePrice"
ID_COLUMN = "Id"

# All expected columns in train.csv (for validation)
EXPECTED_TRAIN_COLUMNS = 81  # 80 features + 1 target (SalePrice)
EXPECTED_TEST_COLUMNS = 80   # 80 features (no SalePrice)


class DataLoader:
    """Handles loading and validation of the House Prices dataset."""

    def __init__(self, data_dir: str = "data/raw", strict_validation: bool = False):
        """
        Initialize DataLoader.

        Args:
            data_dir: Path to directory containing train.csv, test.csv,
                     data_description.txt, sample_submission.csv
            strict_validation: If True, validate exact column counts (81 train, 80 test).
                              If False, only validate required columns exist.
        """
        self.data_dir = Path(data_dir)
        self.train_path = self.data_dir / "train.csv"
        self.test_path = self.data_dir / "test.csv"
        self.desc_path = self.data_dir / "data_description.txt"
        self.submission_path = self.data_dir / "sample_submission.csv"
        self.strict_validation = strict_validation

    def validate_files_exist(self) -> List[str]:
        """
        Check that all required files exist.

        Returns:
            List of missing file paths (empty if all exist)
        """
        required = [self.train_path, self.test_path, self.desc_path, self.submission_path]
        missing = [str(p) for p in required if not p.exists()]
        return missing

    def load_train(self, validate: bool = True) -> pd.DataFrame:
        """
        Load training data.

        Args:
            validate: If True, run validation checks

        Returns:
            DataFrame with training data

        Raises:
            FileNotFoundError: If train.csv doesn't exist
            ValueError: If validation fails
        """
        if not self.train_path.exists():
            raise FileNotFoundError(f"Training file not found: {self.train_path}")

        df = pd.read_csv(self.train_path)

        if validate:
            self._validate_train_data(df)

        return df

    def load_test(self, validate: bool = True) -> pd.DataFrame:
        """
        Load test data.

        Args:
            validate: If True, run validation checks

        Returns:
            DataFrame with test data

        Raises:
            FileNotFoundError: If test.csv doesn't exist
            ValueError: If validation fails
        """
        if not self.test_path.exists():
            raise FileNotFoundError(f"Test file not found: {self.test_path}")

        df = pd.read_csv(self.test_path)

        if validate:
            self._validate_test_data(df)

        return df

    def load_data_description(self) -> Optional[str]:
        """
        Load feature descriptions from data_description.txt.

        Returns:
            String content of description file, or None if not found
        """
        if self.desc_path.exists():
            return self.desc_path.read_text(encoding="utf-8")
        return None

    def _validate_train_data(self, df: pd.DataFrame) -> None:
        """Validate training data structure and content."""
        # Check shape (only if strict validation enabled)
        if self.strict_validation and df.shape[1] != EXPECTED_TRAIN_COLUMNS:
            raise ValueError(
                f"Expected {EXPECTED_TRAIN_COLUMNS} columns in train.csv, "
                f"got {df.shape[1]}. Columns: {list(df.columns)}"
            )

        # Check required columns exist
        required = [ID_COLUMN, TARGET_COLUMN] + BASELINE_FEATURES
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in train.csv: {missing}")

        # Check for duplicate IDs
        if df[ID_COLUMN].duplicated().any():
            dup_count = df[ID_COLUMN].duplicated().sum()
            raise ValueError(f"Found {dup_count} duplicate Id values in train.csv")

        # Check target column has no missing values
        if df[TARGET_COLUMN].isna().any():
            missing_count = df[TARGET_COLUMN].isna().sum()
            raise ValueError(f"Target column '{TARGET_COLUMN}' has {missing_count} missing values")

        # Check baseline features for missing values (warn, don't fail)
        for feat in BASELINE_FEATURES:
            if df[feat].isna().any():
                missing_count = df[feat].isna().sum()
                print(f"WARNING: Baseline feature '{feat}' has {missing_count} missing values")

    def _validate_test_data(self, df: pd.DataFrame) -> None:
        """Validate test data structure."""
        # Check shape (only if strict validation enabled)
        if self.strict_validation and df.shape[1] != EXPECTED_TEST_COLUMNS:
            raise ValueError(
                f"Expected {EXPECTED_TEST_COLUMNS} columns in test.csv, "
                f"got {df.shape[1]}. Columns: {list(df.columns)}"
            )

        # Check required columns exist (no target expected)
        required = [ID_COLUMN] + BASELINE_FEATURES
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in test.csv: {missing}")

        # Check for duplicate IDs
        if df[ID_COLUMN].duplicated().any():
            dup_count = df[ID_COLUMN].duplicated().sum()
            raise ValueError(f"Found {dup_count} duplicate Id values in test.csv")

        # Confirm target column is absent
        if TARGET_COLUMN in df.columns:
            raise ValueError(f"Test data should not contain target column '{TARGET_COLUMN}'")

    def get_baseline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract baseline features from DataFrame.

        Args:
            df: Input DataFrame (train or test)

        Returns:
            DataFrame with only baseline feature columns
        """
        return df[BASELINE_FEATURES].copy()

    def get_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract target column from training DataFrame.

        Args:
            df: Training DataFrame

        Returns:
            Series with SalePrice values

        Raises:
            ValueError: If target column not found
        """
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame")
        return df[TARGET_COLUMN].copy()

    def get_ids(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract ID column.

        Args:
            df: DataFrame (train or test)

        Returns:
            Series with Id values
        """
        return df[ID_COLUMN].copy()

    def print_data_summary(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Print comprehensive data summary for audit purposes."""
        print("=" * 60)
        print("DATA AUDIT SUMMARY")
        print("=" * 60)

        print(f"\nTrain shape: {train_df.shape}")
        print(f"Test shape:  {test_df.shape}")

        print(f"\nTrain columns ({len(train_df.columns)}):")
        for col in train_df.columns:
            dtype = train_df[col].dtype
            missing = train_df[col].isna().sum()
            print(f"  {col}: {dtype} (missing: {missing})")

        print(f"\nTest columns ({len(test_df.columns)}):")
        for col in test_df.columns:
            dtype = test_df[col].dtype
            missing = test_df[col].isna().sum()
            print(f"  {col}: {dtype} (missing: {missing})")

        # Target distribution
        target = train_df[TARGET_COLUMN]
        print(f"\nTarget (SalePrice) Statistics:")
        print(f"  Count: {target.count()}")
        print(f"  Mean:  ${target.mean():,.2f}")
        print(f"  Median: ${target.median():,.2f}")
        print(f"  Std:   ${target.std():,.2f}")
        print(f"  Min:   ${target.min():,.2f}")
        print(f"  Max:   ${target.max():,.2f}")
        print(f"  Skew:  {target.skew():.4f}")

        # Baseline features stats
        print(f"\nBaseline Features Statistics:")
        for feat in BASELINE_FEATURES:
            if feat in train_df.columns:
                col = train_df[feat]
                print(f"  {feat}: mean={col.mean():.2f}, median={col.median():.2f}, "
                      f"std={col.std():.2f}, missing={col.isna().sum()}")


def load_and_validate(data_dir: str = "data/raw") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load and validate both train and test data.

    Args:
        data_dir: Path to data directory

    Returns:
        Tuple of (train_df, test_df)

    Raises:
        FileNotFoundError: If required files missing
        ValueError: If validation fails
    """
    loader = DataLoader(data_dir)

    # Check files exist
    missing = loader.validate_files_exist()
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    # Load data
    train_df = loader.load_train(validate=True)
    test_df = loader.load_test(validate=True)

    # Print summary
    loader.print_data_summary(train_df, test_df)

    return train_df, test_df


if __name__ == "__main__":
    # Run as standalone script for data audit
    try:
        train_df, test_df = load_and_validate()
        print("\n✓ Data validation passed!")
    except Exception as e:
        print(f"\n✗ Data validation failed: {e}")
        exit(1)