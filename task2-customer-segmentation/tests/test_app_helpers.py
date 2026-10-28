"""Unit tests for src.app_helpers module."""

import pandas as pd
import pytest

from src.app_helpers import (
    compute_cluster_summary_table,
    create_new_customer_record,
    filter_customer_data,
    get_model_metadata,
)
from src.data_loader import load_data


@pytest.fixture
def sample_segmented_df() -> pd.DataFrame:
    """Fixture providing a deterministic sample segmented customer DataFrame."""
    return pd.DataFrame({
        "CustomerID": [1, 2, 3, 4, 5],
        "Gender": ["Male", "Female", "Female", "Male", "Female"],
        "Age": [20, 30, 40, 50, 60],
        "Annual_Income": [20.0, 40.0, 60.0, 80.0, 100.0],
        "Spending_Score": [10.0, 30.0, 50.0, 70.0, 90.0],
        "Cluster_Classic_k5": [0, 1, 2, 3, 4],
        "Cluster_Extended_k5": [0, 1, 0, 1, 2],
    })


def test_filter_customer_data_defaults(sample_segmented_df):
    """Verify default filtering returns all records without modification."""
    filtered = filter_customer_data(sample_segmented_df)
    assert len(filtered) == 5
    pd.testing.assert_frame_equal(filtered, sample_segmented_df)


def test_filter_customer_data_cluster_selection(sample_segmented_df):
    """Verify filtering by cluster labels retains only specified clusters."""
    filtered = filter_customer_data(
        sample_segmented_df,
        selected_clusters=[0, 2],
        cluster_col="Cluster_Classic_k5",
    )
    assert len(filtered) == 2
    assert set(filtered["Cluster_Classic_k5"].unique()) == {0, 2}


def test_filter_customer_data_income_range(sample_segmented_df):
    """Verify filtering by income range restricts results properly."""
    filtered = filter_customer_data(
        sample_segmented_df,
        income_range=(30.0, 70.0),
    )
    assert len(filtered) == 2
    assert list(filtered["Annual_Income"]) == [40.0, 60.0]


def test_filter_customer_data_spending_range(sample_segmented_df):
    """Verify filtering by spending range restricts results properly."""
    filtered = filter_customer_data(
        sample_segmented_df,
        spending_range=(20.0, 80.0),
    )
    assert len(filtered) == 3
    assert list(filtered["Spending_Score"]) == [30.0, 50.0, 70.0]


def test_filter_customer_data_gender(sample_segmented_df):
    """Verify filtering by gender restricts results properly."""
    filtered_f = filter_customer_data(sample_segmented_df, selected_gender="Female")
    assert len(filtered_f) == 3
    assert (filtered_f["Gender"] == "Female").all()

    filtered_m = filter_customer_data(sample_segmented_df, selected_gender="Male")
    assert len(filtered_m) == 2
    assert (filtered_m["Gender"] == "Male").all()


def test_create_new_customer_record_valid():
    """Verify create_new_customer_record builds a valid 1-row DataFrame."""
    record = create_new_customer_record(
        age=35,
        annual_income=65.5,
        spending_score=50.0,
        gender="Female",
    )
    assert isinstance(record, pd.DataFrame)
    assert len(record) == 1
    assert record["Age"].iloc[0] == 35
    assert record["Annual_Income"].iloc[0] == 65.5
    assert record["Spending_Score"].iloc[0] == 50.0
    assert record["Gender"].iloc[0] == "Female"
    assert "CustomerID" in record.columns


def test_create_new_customer_record_invalid_inputs():
    """Verify invalid domain values raise ValueError with informative messages."""
    # Invalid age
    with pytest.raises(ValueError, match="Age must be between"):
        create_new_customer_record(age=-5, annual_income=50, spending_score=50, gender="Male")
    with pytest.raises(ValueError, match="Age must be between"):
        create_new_customer_record(age=150, annual_income=50, spending_score=50, gender="Male")

    # Negative income
    with pytest.raises(ValueError, match="Annual Income cannot be negative"):
        create_new_customer_record(age=30, annual_income=-10, spending_score=50, gender="Male")

    # Invalid spending score
    with pytest.raises(ValueError, match="Spending Score must be between"):
        create_new_customer_record(age=30, annual_income=50, spending_score=150, gender="Male")

    # Invalid gender
    with pytest.raises(ValueError, match="Gender must be 'Male' or 'Female'"):
        create_new_customer_record(age=30, annual_income=50, spending_score=50, gender="Unknown")


def test_compute_cluster_summary_table(sample_segmented_df):
    """Verify compute_cluster_summary_table computes accurate metrics and sums to 100%."""
    table = compute_cluster_summary_table(sample_segmented_df, cluster_col="Cluster_Classic_k5")
    assert len(table) == 5
    assert table["Customers (n)"].sum() == 5
    assert round(table["Share (%)"].sum(), 0) == 100.0
    assert "Persona" in table.columns
    assert "Mean Income ($k)" in table.columns


def test_get_model_metadata():
    """Verify get_model_metadata returns valid dictionary for supported models."""
    classic_meta = get_model_metadata("classic")
    assert classic_meta["k"] == 5
    assert "Annual_Income" in classic_meta["features"]
    assert classic_meta["silhouette"] == 0.5547

    ext_meta = get_model_metadata("extended")
    assert ext_meta["k"] == 5
    assert "Age" in ext_meta["features"]
    assert ext_meta["silhouette"] == 0.3498

    with pytest.raises(ValueError, match="Unknown model_type"):
        get_model_metadata("super_model")
