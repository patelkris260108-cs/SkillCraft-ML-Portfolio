"""Data loading and schema validation module for Mall Customer Segmentation.

This module provides reliable, modular data loading functions and classes to:
- Resolve data paths relative to CWD and project root.
- Validate dataset schema and required columns against expected Mall Customer attributes.
- Normalize column names to clean, Python-friendly identifiers.
- Validate numeric ranges and categorical values with descriptive error reporting.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Union
import numpy as np
import pandas as pd

# Expected raw column names to normalized column names
RAW_TO_NORMALIZED_COLUMNS: Dict[str, str] = {
    "CustomerID": "CustomerID",
    "Gender": "Gender",
    "Age": "Age",
    "Annual Income (k$)": "Annual_Income",
    "Spending Score (1-100)": "Spending_Score",
}

NORMALIZED_COLUMNS: List[str] = [
    "CustomerID",
    "Gender",
    "Age",
    "Annual_Income",
    "Spending_Score",
]

VALID_GENDERS: Set[str] = {"Male", "Female"}


def resolve_data_path(file_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve data file path flexibly across working directories.

    Checks:
    1. Direct path if specified (relative to CWD or absolute).
    2. Relative to the task2-customer-segmentation root directory.
    3. Default location `data/raw/Mall_Customers.csv` relative to CWD or task2 root.

    Args:
        file_path: Optional path to data file.

    Returns:
        Resolved Path object pointing to an existing file.

    Raises:
        FileNotFoundError: If the file cannot be located.
    """
    task2_root = Path(__file__).resolve().parent.parent

    if file_path is not None:
        p = Path(file_path)
        if p.is_file():
            return p.resolve()
        p_task2 = task2_root / file_path
        if p_task2.is_file():
            return p_task2.resolve()
        raise FileNotFoundError(
            f"Specified dataset file not found at '{file_path}' (checked '{p}' and '{p_task2}')."
        )

    # Default candidate locations
    candidates = [
        Path("data/raw/Mall_Customers.csv"),
        task2_root / "data" / "raw" / "Mall_Customers.csv",
        Path("task2-customer-segmentation/data/raw/Mall_Customers.csv"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Default dataset 'Mall_Customers.csv' could not be found. "
        f"Checked paths: {[str(c) for c in candidates]}"
    )


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that DataFrame contains all required columns (raw or normalized).

    Args:
        df: Input DataFrame to validate.

    Raises:
        ValueError: If DataFrame is empty or missing required columns.
    """
    if df.empty:
        raise ValueError("DataFrame is empty. Expected at least one customer record.")

    cols = set(df.columns)
    raw_required = set(RAW_TO_NORMALIZED_COLUMNS.keys())
    norm_required = set(NORMALIZED_COLUMNS)

    # Check if either full raw schema or full normalized schema is present
    has_raw = raw_required.issubset(cols)
    has_norm = norm_required.issubset(cols)

    if not has_raw and not has_norm:
        # Determine which expected fields are missing
        # Match by normalized name or raw name
        missing_fields = []
        for raw_col, norm_col in RAW_TO_NORMALIZED_COLUMNS.items():
            if raw_col not in cols and norm_col not in cols:
                missing_fields.append(f"{raw_col} (or {norm_col})")
        raise ValueError(
            f"Dataset schema validation failed. Missing required columns: {missing_fields}. "
            f"Existing columns: {list(df.columns)}"
        )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names to Python-friendly identifiers.

    Renames:
    - 'Annual Income (k$)' -> 'Annual_Income'
    - 'Spending Score (1-100)' -> 'Spending_Score'
    Preserves other standard names ('CustomerID', 'Gender', 'Age').

    Args:
        df: Input DataFrame.

    Returns:
        A copy of the DataFrame with normalized column names.
    """
    df_copy = df.copy()
    rename_map = {k: v for k, v in RAW_TO_NORMALIZED_COLUMNS.items() if k in df_copy.columns}
    df_copy = df_copy.rename(columns=rename_map)
    return df_copy


def validate_numeric_fields(df: pd.DataFrame) -> None:
    """Validate numeric ranges and categorical values for customer segmentation.

    Validations:
    - No null or NaN values in required fields.
    - CustomerID: positive integer.
    - Age: numeric, 1 <= Age <= 120.
    - Annual_Income: numeric, Annual_Income >= 0.
    - Spending_Score: numeric, 1 <= Spending_Score <= 100.
    - Gender: categorical in {'Male', 'Female'}.

    Args:
        df: Normalized DataFrame to validate.

    Raises:
        ValueError: If any validation rule fails.
    """
    # 1. Null / NaN checks
    null_counts = df[NORMALIZED_COLUMNS].isnull().sum()
    if null_counts.any():
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Dataset contains null or NaN values in columns: {cols_with_nulls}")

    # 2. CustomerID validation
    if not pd.api.types.is_numeric_dtype(df["CustomerID"]):
        raise ValueError("Field 'CustomerID' must be numeric.")
    if (df["CustomerID"] <= 0).any():
        invalid_ids = df[df["CustomerID"] <= 0]["CustomerID"].tolist()
        raise ValueError(f"Field 'CustomerID' must be positive. Found invalid values: {invalid_ids}")

    # 3. Age validation
    if not pd.api.types.is_numeric_dtype(df["Age"]):
        raise ValueError("Field 'Age' must be numeric.")
    if (df["Age"] < 1).any() or (df["Age"] > 120).any():
        invalid_ages = df[(df["Age"] < 1) | (df["Age"] > 120)]["Age"].tolist()
        raise ValueError(f"Field 'Age' contains invalid values outside [1, 120]: {invalid_ages}")

    # 4. Annual_Income validation
    if not pd.api.types.is_numeric_dtype(df["Annual_Income"]):
        raise ValueError("Field 'Annual_Income' must be numeric.")
    if (df["Annual_Income"] < 0).any():
        invalid_incomes = df[df["Annual_Income"] < 0]["Annual_Income"].tolist()
        raise ValueError(f"Field 'Annual_Income' contains negative values: {invalid_incomes}")

    # 5. Spending_Score validation
    if not pd.api.types.is_numeric_dtype(df["Spending_Score"]):
        raise ValueError("Field 'Spending_Score' must be numeric.")
    if (df["Spending_Score"] < 1).any() or (df["Spending_Score"] > 100).any():
        invalid_scores = df[(df["Spending_Score"] < 1) | (df["Spending_Score"] > 100)]["Spending_Score"].tolist()
        raise ValueError(
            f"Field 'Spending_Score' contains values outside range [1, 100]: {invalid_scores}"
        )

    # 6. Gender validation
    invalid_genders = set(df["Gender"].unique()) - VALID_GENDERS
    if invalid_genders:
        raise ValueError(
            f"Field 'Gender' contains unrecognized categories: {invalid_genders}. "
            f"Expected values from: {VALID_GENDERS}"
        )


def load_data(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load, validate, and normalize Mall Customer dataset from CSV.

    Args:
        file_path: Optional custom path to dataset CSV file.

    Returns:
        pd.DataFrame with normalized columns:
        ['CustomerID', 'Gender', 'Age', 'Annual_Income', 'Spending_Score']

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If schema or data validation fails.
    """
    resolved_path = resolve_data_path(file_path)
    df = pd.read_csv(resolved_path)

    validate_schema(df)
    df_norm = normalize_columns(df)
    validate_numeric_fields(df_norm)

    return df_norm


class DataLoader:
    """Object-oriented wrapper for loading and validating Mall Customer data."""

    def __init__(self, file_path: Optional[Union[str, Path]] = None) -> None:
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        """Load and return validated, normalized customer data."""
        return load_data(self.file_path)
