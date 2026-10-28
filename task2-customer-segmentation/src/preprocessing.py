"""Customer data preprocessing and feature engineering module for Task 2.

This module provides reproducible, leakage-free preprocessing pipelines for customer clustering:
- Explicit feature sets:
    1. 'classic': 2D segmentation benchmark ['Annual_Income', 'Spending_Score']
    2. 'extended': Demographic + financial ['Age', 'Annual_Income', 'Spending_Score', 'Gender_Male']
- Complete exclusion of arbitrary identifier 'CustomerID' from all feature vectors.
- Reproducible numeric scaling via scikit-learn StandardScaler.
- Categorical encoding of 'Gender' via OneHotEncoder (drop='first', yielding Gender_Male binary indicator).
- Preservation of row order and index alignment.
- Leakage-safe fit/transform separation for cross-validation or train/test splits.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from src.data_loader import normalize_columns
except ImportError:
    from data_loader import normalize_columns

SUPPORTED_FEATURE_SETS: Dict[str, Dict[str, List[str]]] = {
    "classic": {
        "numeric": ["Annual_Income", "Spending_Score"],
        "categorical": [],
    },
    "extended": {
        "numeric": ["Age", "Annual_Income", "Spending_Score"],
        "categorical": ["Gender"],
    },
}


class CustomerPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessing and feature transformation pipeline for Mall Customer Segmentation.

    Supports 'classic' (2D) and 'extended' (4D) segmentation feature sets with
    scikit-learn transformer standards.

    Attributes:
        feature_set: Name of feature set ('classic' or 'extended').
        transformer_: Fitted scikit-learn ColumnTransformer instance.
        is_fitted_: Boolean indicating whether transformer has been fitted.
    """

    def __init__(self, feature_set: str = "classic") -> None:
        """Initialize CustomerPreprocessor.

        Args:
            feature_set: 'classic' for Annual_Income + Spending_Score,
                         'extended' for Age + Annual_Income + Spending_Score + Gender.

        Raises:
            ValueError: If feature_set is not 'classic' or 'extended'.
        """
        self.feature_set = self._validate_feature_set_name(feature_set)
        self.transformer_: Optional[ColumnTransformer] = None
        self.is_fitted_: bool = False
        self._feature_names: List[str] = []

    @staticmethod
    def _validate_feature_set_name(feature_set: str) -> str:
        """Validate and normalize feature set name."""
        if not isinstance(feature_set, str):
            raise ValueError(
                f"feature_set must be a string, got {type(feature_set).__name__}."
            )
        name = feature_set.strip().lower()
        if name not in SUPPORTED_FEATURE_SETS:
            raise ValueError(
                f"Invalid feature_set '{feature_set}'. "
                f"Supported options: {list(SUPPORTED_FEATURE_SETS.keys())}"
            )
        return name

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize columns, ensuring CustomerID is excluded from features."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Input must be a pandas DataFrame, got {type(df).__name__}.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        # Normalize column names if raw headers are present
        df_norm = normalize_columns(df)

        # Check required columns for the chosen feature set
        feature_spec = SUPPORTED_FEATURE_SETS[self.feature_set]
        required_cols = feature_spec["numeric"] + feature_spec["categorical"]
        missing = [col for col in required_cols if col not in df_norm.columns]
        if missing:
            raise ValueError(
                f"Missing required columns for '{self.feature_set}' feature set: {missing}. "
                f"Available columns: {list(df_norm.columns)}"
            )

        return df_norm

    def fit(self, df: pd.DataFrame, y=None) -> "CustomerPreprocessor":
        """Fit scaler on numeric features and encoder on categorical features.

        Args:
            df: Input customer DataFrame.
            y: Ignored (unsupervised).

        Returns:
            self: Fitted preprocessor.
        """
        df_norm = self._prepare_df(df)
        feature_spec = SUPPORTED_FEATURE_SETS[self.feature_set]

        transformers = []
        feature_names: List[str] = []

        if feature_spec["numeric"]:
            num_cols = feature_spec["numeric"]
            transformers.append(
                ("num", StandardScaler(), num_cols)
            )
            feature_names.extend(num_cols)

        if feature_spec["categorical"]:
            cat_cols = feature_spec["categorical"]
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                    cat_cols,
                )
            )
            # Gender with drop='first' ('Female' dropped as reference -> 'Gender_Male' indicator)
            feature_names.append("Gender_Male")

        self.transformer_ = ColumnTransformer(
            transformers=transformers,
            remainder="drop",  # CustomerID and any other columns are explicitly dropped
            verbose_feature_names_out=False,
        )

        self.transformer_.fit(df_norm)
        self.is_fitted_ = True
        self._feature_names = feature_names

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform input features using fitted scaler and encoder.

        Args:
            df: Input customer DataFrame.

        Returns:
            2D numpy array of scaled and encoded features.

        Raises:
            NotFittedError: If preprocessor has not been fitted.
        """
        if not self.is_fitted_ or self.transformer_ is None:
            raise NotFittedError(
                "CustomerPreprocessor instance is not fitted yet. Call 'fit' before 'transform'."
            )

        df_norm = self._prepare_df(df)
        transformed = self.transformer_.transform(df_norm)

        # Ensure output is a dense float numpy array
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        return np.asarray(transformed, dtype=np.float64)

    def fit_transform(self, df: pd.DataFrame, y=None) -> np.ndarray:
        """Fit preprocessor and transform features in a single step."""
        return self.fit(df, y=y).transform(df)

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features and return as a DataFrame with preserved index and clean column names.

        Args:
            df: Input customer DataFrame.

        Returns:
            pd.DataFrame with scaled/encoded values and descriptive column headers.
        """
        arr = self.transform(df)
        return pd.DataFrame(
            data=arr,
            columns=self.get_feature_names(),
            index=df.index.copy(),
        )

    def fit_transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessor and return transformed features as a DataFrame with preserved index.

        Args:
            df: Input customer DataFrame.

        Returns:
            pd.DataFrame with scaled/encoded values and descriptive column headers.
        """
        return self.fit(df).transform_df(df)

    def get_feature_names(self) -> List[str]:
        """Return list of transformed feature names.

        Raises:
            NotFittedError: If preprocessor has not been fitted.
        """
        if not self.is_fitted_:
            raise NotFittedError(
                "CustomerPreprocessor is not fitted. Feature names are unavailable."
            )
        return list(self._feature_names)


def prepare_features(
    df: pd.DataFrame,
    feature_set: str = "classic",
    fit: bool = True,
    preprocessor: Optional[CustomerPreprocessor] = None,
) -> Tuple[np.ndarray, CustomerPreprocessor, List[str]]:
    """Functional convenience helper to prepare features for clustering.

    Args:
        df: Input customer DataFrame.
        feature_set: 'classic' or 'extended'.
        fit: If True, fit a new preprocessor; if False, use provided fitted preprocessor.
        preprocessor: Existing preprocessor (required if fit=False).

    Returns:
        Tuple of (transformed_array, preprocessor, feature_names).
    """
    if fit:
        prep = CustomerPreprocessor(feature_set=feature_set)
        arr = prep.fit_transform(df)
    else:
        if preprocessor is None:
            raise ValueError("Existing preprocessor must be provided when fit=False.")
        prep = preprocessor
        arr = prep.transform(df)

    return arr, prep, prep.get_feature_names()
