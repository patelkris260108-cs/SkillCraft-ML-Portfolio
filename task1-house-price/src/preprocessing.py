"""
Preprocessing pipelines for Task 1: House Price Prediction.

This module provides sklearn-compatible preprocessing pipelines for both
the baseline (3-feature) model and the extended all-feature model.
All transformers are fit ONLY on training data to prevent leakage.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression

# Baseline feature names (assignment requirement)
BASELINE_FEATURES = ["GrLivArea", "BedroomAbvGr", "FullBath"]
TARGET_COLUMN = "SalePrice"


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select specific columns from DataFrame.
    Compatible with sklearn Pipeline.
    """

    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "FeatureSelector":
        # Validate columns exist
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns].copy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.columns)


class LogTargetTransformer(BaseEstimator, TransformerMixin):
    """
    Transform target using log1p for training, with inverse expm1 for predictions.
    This helps with the right-skewed SalePrice distribution.
    """

    def __init__(self):
        self._fitted = False

    def fit(self, y: np.ndarray) -> "LogTargetTransformer":
        # Just validate input
        if np.any(y <= 0):
            raise ValueError("Target values must be positive for log transform")
        self._fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return np.log1p(y)

    def inverse_transform(self, y_log: np.ndarray) -> np.ndarray:
        return np.expm1(y_log)


def create_baseline_preprocessor() -> Pipeline:
    """
    Create preprocessing pipeline for baseline 3-feature model.

    Pipeline steps:
    1. FeatureSelector - select GrLivArea, BedroomAbvGr, FullBath
    2. SimpleImputer - median imputation for missing values
    3. StandardScaler - standardize features (helps with regularization)

    Returns:
        sklearn Pipeline object
    """
    return Pipeline([
        ("selector", FeatureSelector(BASELINE_FEATURES)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


def identify_feature_types(df: pd.DataFrame, exclude: List[str] = None) -> Tuple[List[str], List[str]]:
    """
    Automatically identify numeric and categorical feature columns.

    Args:
        df: Input DataFrame
        exclude: Columns to exclude from feature set (e.g., Id, SalePrice)

    Returns:
        Tuple of (numeric_features, categorical_features)
    """
    if exclude is None:
        exclude = ["Id", "SalePrice"]

    feature_cols = [c for c in df.columns if c not in exclude]

    numeric_features = []
    categorical_features = []

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Additional check: if low cardinality numeric, might be categorical
            if df[col].nunique() <= 10 and col not in ["GrLivArea", "LotArea", "TotalBsmtSF", "GarageArea"]:
                categorical_features.append(col)
            else:
                numeric_features.append(col)
        else:
            categorical_features.append(col)

    return numeric_features, categorical_features


def create_full_preprocessor(
    train_df: pd.DataFrame,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> ColumnTransformer:
    """
    Create preprocessing pipeline for all-feature model.

    Uses ColumnTransformer to apply different preprocessing to
    numeric and categorical features.

    Args:
        train_df: Training DataFrame (used to infer feature types if not provided)
        numeric_features: Optional list of numeric column names
        categorical_features: Optional list of categorical column names

    Returns:
        ColumnTransformer object
    """
    # Infer feature types if not provided
    if numeric_features is None or categorical_features is None:
        num_feat, cat_feat = identify_feature_types(train_df)
        if numeric_features is None:
            numeric_features = num_feat
        if categorical_features is None:
            categorical_features = cat_feat

    # Numeric pipeline: median imputation + scaling
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline: most-frequent imputation + one-hot encoding
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Combine with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features)
        ],
        remainder="drop",  # Drop any columns not specified
        verbose_feature_names_out=False
    )

    return preprocessor


def create_full_pipeline(
    train_df: pd.DataFrame,
    estimator,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> Pipeline:
    """
    Create complete pipeline: preprocessing + estimator.

    Args:
        train_df: Training DataFrame (for feature type inference)
        estimator: sklearn estimator (e.g., LinearRegression, Ridge)
        numeric_features: Optional list of numeric columns
        categorical_features: Optional list of categorical columns

    Returns:
        Complete sklearn Pipeline
    """
    preprocessor = create_full_preprocessor(
        train_df, numeric_features, categorical_features
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("estimator", estimator)
    ])


def get_feature_names_after_preprocessing(
    preprocessor: ColumnTransformer,
    numeric_features: List[str],
    categorical_features: List[str]
) -> List[str]:
    """
    Get feature names after ColumnTransformer transformation.

    Useful for model interpretation and debugging.

    Args:
        preprocessor: Fitted ColumnTransformer
        numeric_features: Original numeric feature names
        categorical_features: Original categorical feature names

    Returns:
        List of transformed feature names
    """
    # For numeric features, names stay the same (after scaling)
    numeric_names = numeric_features

    # For categorical, get names from OneHotEncoder
    cat_encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    if hasattr(cat_encoder, "get_feature_names_out"):
        cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    else:
        cat_names = categorical_features

    return numeric_names + cat_names


# Convenience function for baseline pipeline with target transform
def create_baseline_pipeline_with_target(estimator) -> Pipeline:
    """
    Create baseline pipeline including target transformation.

    Note: In practice, target transform is usually handled outside
    the feature pipeline. This is provided for convenience.

    Args:
        estimator: sklearn estimator

    Returns:
        Pipeline with feature preprocessing only (target handled separately)
    """
    return create_baseline_preprocessor()


if __name__ == "__main__":
    # Quick test
    print("Testing preprocessing pipelines...")

    # Create sample data
    sample_df = pd.DataFrame({
        "GrLivArea": [1500, 2000, np.nan, 1800],
        "BedroomAbvGr": [3, 4, 2, 3],
        "FullBath": [2, 3, 2, 2],
        "SalePrice": [200000, 300000, 180000, 250000],
        "Neighborhood": ["A", "B", "A", "C"],
        "OverallQual": [7, 8, 6, 7]
    })

    # Test baseline preprocessor
    baseline_pipe = create_baseline_preprocessor()
    X_transformed = baseline_pipe.fit_transform(sample_df)
    print(f"Baseline transformed shape: {X_transformed.shape}")
    print(f"Baseline feature names: {baseline_pipe.named_steps['selector'].get_feature_names_out()}")

    # Test full preprocessor
    full_preprocessor = create_full_preprocessor(sample_df)
    print(f"Full preprocessor numeric features: {full_preprocessor.transformers_[0][2]}")
    print(f"Full preprocessor categorical features: {full_preprocessor.transformers_[1][2]}")

    print("\n✓ Preprocessing module test passed!")