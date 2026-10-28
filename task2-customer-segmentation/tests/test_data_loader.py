"""Unit tests for src.data_loader module."""

import hashlib
from pathlib import Path
import pandas as pd
import pytest

from src.data_loader import (
    DataLoader,
    NORMALIZED_COLUMNS,
    RAW_TO_NORMALIZED_COLUMNS,
    load_data,
    normalize_columns,
    resolve_data_path,
    validate_numeric_fields,
    validate_schema,
)


@pytest.fixture
def sample_valid_raw_df() -> pd.DataFrame:
    """Fixture providing a small valid raw customer DataFrame."""
    return pd.DataFrame({
        "CustomerID": [1, 2, 3],
        "Gender": ["Male", "Female", "Female"],
        "Age": [19, 21, 35],
        "Annual Income (k$)": [15, 16, 75],
        "Spending Score (1-100)": [39, 81, 5],
    })


def test_resolve_data_path_default():
    """Verify default data path resolution locates the raw dataset."""
    path = resolve_data_path()
    assert path.is_file()
    assert path.name == "Mall_Customers.csv"


def test_resolve_data_path_missing():
    """Verify missing file path raises FileNotFoundError with descriptive message."""
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_data_path("non_existent_directory/non_existent_file.csv")


def test_load_data_valid_default():
    """Verify loading the real raw dataset yields 200 rows and normalized schema."""
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 200
    assert list(df.columns) == NORMALIZED_COLUMNS
    assert df["CustomerID"].is_monotonic_increasing
    assert df.isnull().sum().sum() == 0


def test_load_data_with_explicit_path():
    """Verify loading with an explicit path functions correctly."""
    default_path = resolve_data_path()
    df = load_data(default_path)
    assert len(df) == 200
    assert "Annual_Income" in df.columns
    assert "Spending_Score" in df.columns


def test_data_loader_class():
    """Verify DataLoader class wraps load_data consistently."""
    loader = DataLoader()
    df = loader.load()
    assert len(df) == 200
    assert list(df.columns) == NORMALIZED_COLUMNS


def test_normalize_columns(sample_valid_raw_df):
    """Verify column normalization renames raw headers correctly without mutating original."""
    original_cols = list(sample_valid_raw_df.columns)
    df_norm = normalize_columns(sample_valid_raw_df)
    
    # Original should be untouched
    assert list(sample_valid_raw_df.columns) == original_cols
    # Normalized should have expected names
    assert list(df_norm.columns) == NORMALIZED_COLUMNS
    assert "Annual_Income" in df_norm.columns
    assert "Spending_Score" in df_norm.columns


def test_validate_schema_missing_column():
    """Verify validate_schema detects missing required columns."""
    incomplete_df = pd.DataFrame({
        "CustomerID": [1],
        "Gender": ["Male"],
        "Age": [25],
        # Missing income and spending score
    })
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(incomplete_df)


def test_validate_schema_empty_df():
    """Verify empty DataFrame raises ValueError."""
    empty_df = pd.DataFrame(columns=NORMALIZED_COLUMNS)
    with pytest.raises(ValueError, match="DataFrame is empty"):
        validate_schema(empty_df)


def test_validate_numeric_fields_nulls():
    """Verify null/NaN values are detected and rejected."""
    df_with_null = pd.DataFrame({
        "CustomerID": [1, 2],
        "Gender": ["Male", "Female"],
        "Age": [25, None],
        "Annual_Income": [50, 60],
        "Spending_Score": [50, 70],
    })
    with pytest.raises(ValueError, match="null or NaN"):
        validate_numeric_fields(df_with_null)


def test_validate_numeric_fields_invalid_customer_id():
    """Verify non-positive CustomerID raises ValueError."""
    df_bad_id = pd.DataFrame({
        "CustomerID": [0, 2],
        "Gender": ["Male", "Female"],
        "Age": [25, 30],
        "Annual_Income": [50, 60],
        "Spending_Score": [50, 70],
    })
    with pytest.raises(ValueError, match="CustomerID.*must be positive"):
        validate_numeric_fields(df_bad_id)


def test_validate_numeric_fields_invalid_age():
    """Verify negative or excessive age raises ValueError."""
    df_bad_age = pd.DataFrame({
        "CustomerID": [1, 2],
        "Gender": ["Male", "Female"],
        "Age": [-5, 30],
        "Annual_Income": [50, 60],
        "Spending_Score": [50, 70],
    })
    with pytest.raises(ValueError, match="outside \\[1, 120\\]"):
        validate_numeric_fields(df_bad_age)


def test_validate_numeric_fields_negative_income():
    """Verify negative Annual_Income raises ValueError."""
    df_bad_income = pd.DataFrame({
        "CustomerID": [1, 2],
        "Gender": ["Male", "Female"],
        "Age": [25, 30],
        "Annual_Income": [-10, 60],
        "Spending_Score": [50, 70],
    })
    with pytest.raises(ValueError, match="negative values"):
        validate_numeric_fields(df_bad_income)


def test_validate_numeric_fields_invalid_spending_score():
    """Verify Spending_Score outside [1, 100] raises ValueError."""
    df_bad_ss = pd.DataFrame({
        "CustomerID": [1, 2],
        "Gender": ["Male", "Female"],
        "Age": [25, 30],
        "Annual_Income": [50, 60],
        "Spending_Score": [105, 70],
    })
    with pytest.raises(ValueError, match="outside range \\[1, 100\\]"):
        validate_numeric_fields(df_bad_ss)


def test_validate_numeric_fields_invalid_gender():
    """Verify unknown categorical gender raises ValueError."""
    df_bad_gender = pd.DataFrame({
        "CustomerID": [1, 2],
        "Gender": ["Other", "Female"],
        "Age": [25, 30],
        "Annual_Income": [50, 60],
        "Spending_Score": [50, 70],
    })
    with pytest.raises(ValueError, match="unrecognized categories"):
        validate_numeric_fields(df_bad_gender)


def test_load_data_with_temp_malformed_csv(tmp_path: Path):
    """Verify load_data with a temporary malformed CSV file raises ValueError."""
    temp_csv = tmp_path / "malformed.csv"
    temp_csv.write_text(
        "CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)\n"
        "1,Male,-99,15,50\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside \\[1, 120\\]"):
        load_data(temp_csv)


def test_raw_csv_unmodified():
    """Verify that calling load_data does not modify the raw CSV file."""
    path = resolve_data_path()
    with open(path, "rb") as f:
        hash_before = hashlib.sha256(f.read()).hexdigest()

    _ = load_data(path)

    with open(path, "rb") as f:
        hash_after = hashlib.sha256(f.read()).hexdigest()

    assert hash_before == hash_after
