"""Unit tests for src.clustering module."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.clustering import CustomerClustering, fit_kmeans
from src.data_loader import load_data


@pytest.fixture
def sample_customer_df() -> pd.DataFrame:
    """Fixture providing a deterministic sample DataFrame for clustering tests."""
    return pd.DataFrame({
        "CustomerID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Gender": ["Male", "Female", "Female", "Male", "Female", "Male", "Female", "Female", "Male", "Female"],
        "Age": [19, 21, 20, 23, 31, 22, 35, 23, 64, 30],
        "Annual_Income": [15, 15, 16, 16, 17, 17, 18, 18, 19, 20],
        "Spending_Score": [39, 81, 6, 77, 40, 76, 6, 94, 3, 72],
    })


def test_deterministic_clustering_with_same_seed(sample_customer_df):
    """Verify identical random seeds produce exactly identical cluster assignments and inertia."""
    model_1 = CustomerClustering(n_clusters=3, random_state=42)
    labels_1 = model_1.fit_predict(sample_customer_df)

    model_2 = CustomerClustering(n_clusters=3, random_state=42)
    labels_2 = model_2.fit_predict(sample_customer_df)

    np.testing.assert_array_equal(labels_1, labels_2)
    assert model_1.inertia_ == model_2.inertia_


def test_invalid_cluster_counts(sample_customer_df):
    """Verify invalid k values (k < 2, k >= n_samples, non-integer) are rejected with ValueError."""
    # k < 2
    with pytest.raises(ValueError, match="n_clusters must be >= 2"):
        model = CustomerClustering(n_clusters=1)
        model.fit(sample_customer_df)

    # k >= n_samples
    with pytest.raises(ValueError, match="must be strictly less than the number of available samples"):
        model = CustomerClustering(n_clusters=10)
        model.fit(sample_customer_df)

    # Non-integer k
    with pytest.raises(ValueError, match="n_clusters must be an integer"):
        model = CustomerClustering(n_clusters="five")  # type: ignore
        model.fit(sample_customer_df)


def test_invalid_feature_set(sample_customer_df):
    """Verify invalid feature set name is rejected."""
    with pytest.raises(ValueError, match="Invalid feature_set"):
        model = CustomerClustering(feature_set="unsupported_set")
        model.fit(sample_customer_df)


def test_customer_id_strictly_excluded_from_clustering():
    """Verify CustomerID is excluded from clustering features."""
    df = load_data()
    model = CustomerClustering(n_clusters=5, feature_set="classic")
    model.fit(df)

    assert model.preprocessor_ is not None
    feature_names = model.preprocessor_.get_feature_names()
    assert "CustomerID" not in feature_names
    assert len(feature_names) == 2


def test_row_count_and_index_preservation(sample_customer_df):
    """Verify assigned clusters retain exact customer row count, custom index, and column names."""
    custom_index = [f"cust_{i}" for i in range(len(sample_customer_df))]
    df_indexed = sample_customer_df.copy()
    df_indexed.index = custom_index

    model = CustomerClustering(n_clusters=3, random_state=42)
    segmented = model.assign_clusters(df_indexed, cluster_col="Segment")

    assert len(segmented) == len(sample_customer_df)
    assert list(segmented.index) == custom_index
    assert "Segment" in segmented.columns
    assert set(segmented["Segment"].unique()).issubset({0, 1, 2})


def test_predict_unfitted_raises_error(sample_customer_df):
    """Verify calling predict on unfitted model raises NotFittedError."""
    model = CustomerClustering(n_clusters=3)
    with pytest.raises(NotFittedError):
        model.predict(sample_customer_df)


def test_new_customer_prediction(sample_customer_df):
    """Verify fitted model can predict cluster label for a new unseen customer."""
    model = CustomerClustering(n_clusters=3, feature_set="classic", random_state=42)
    model.fit(sample_customer_df)

    new_customer = pd.DataFrame({
        "CustomerID": [999],
        "Gender": ["Female"],
        "Age": [28],
        "Annual_Income": [17],
        "Spending_Score": [75],
    })

    pred = model.predict(new_customer)
    assert isinstance(pred, np.ndarray)
    assert len(pred) == 1
    assert pred[0] in [0, 1, 2]


def test_get_cluster_centers_scaled_and_unscaled(sample_customer_df):
    """Verify cluster centers are correctly returned in scaled and unscaled domain units."""
    model = CustomerClustering(n_clusters=3, feature_set="classic", random_state=42)
    model.fit(sample_customer_df)

    # Scaled centers
    scaled_centers = model.get_cluster_centers(scaled=True)
    assert scaled_centers.shape == (3, 2)
    assert list(scaled_centers.columns) == ["Annual_Income", "Spending_Score"]

    # Unscaled centers in original domain ($k, score)
    unscaled_centers = model.get_cluster_centers(scaled=False)
    assert unscaled_centers.shape == (3, 2)
    assert list(unscaled_centers.columns) == ["Annual_Income", "Spending_Score"]
    # Check income is within realistic range of sample data (15-20)
    assert (unscaled_centers["Annual_Income"] >= 14).all()
    assert (unscaled_centers["Annual_Income"] <= 21).all()


def test_extended_cluster_centers_unscaled():
    """Verify extended model unscaled centers include Age, Income, Spending, and Gender_Male_Ratio."""
    df = load_data()
    model = CustomerClustering(n_clusters=5, feature_set="extended", random_state=42)
    model.fit(df)

    centers = model.get_cluster_centers(scaled=False)
    assert centers.shape == (5, 4)
    assert list(centers.columns) == ["Age", "Annual_Income", "Spending_Score", "Gender_Male_Ratio"]
    # Gender ratio must be valid proportion in [0, 1]
    assert (centers["Gender_Male_Ratio"] >= 0.0).all()
    assert (centers["Gender_Male_Ratio"] <= 1.0).all()


def test_joblib_save_and_load_roundtrip(tmp_path: Path, sample_customer_df):
    """Verify saving fitted model and reloading produces identical predictions."""
    model = CustomerClustering(n_clusters=3, feature_set="classic", random_state=42)
    model.fit(sample_customer_df)
    original_preds = model.predict(sample_customer_df)

    save_path = tmp_path / "test_kmeans.joblib"
    model.save(save_path)
    assert save_path.is_file()

    loaded_model = CustomerClustering.load(save_path)
    loaded_preds = loaded_model.predict(sample_customer_df)

    np.testing.assert_array_equal(original_preds, loaded_preds)
    assert loaded_model.inertia_ == model.inertia_


def test_save_unfitted_model_raises_error(tmp_path: Path):
    """Verify attempting to save an unfitted model raises NotFittedError."""
    model = CustomerClustering(n_clusters=3)
    with pytest.raises(NotFittedError):
        model.save(tmp_path / "unfitted.joblib")


def test_load_nonexistent_file_raises_error():
    """Verify loading from nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        CustomerClustering.load("nonexistent_directory/nonexistent_model.joblib")


def test_empty_dataframe_raises_error():
    """Verify empty DataFrame raises ValueError."""
    model = CustomerClustering(n_clusters=3)
    empty_df = pd.DataFrame(columns=["Annual_Income", "Spending_Score"])
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        model.fit(empty_df)


class _UnauthorizedClustering:
    is_fitted_ = True


def test_load_rejects_unauthorized_class(tmp_path: Path):
    """Verify load strictly rejects foreign objects even if attribute structure matches."""
    import joblib

    fake_instance = _UnauthorizedClustering()
    fake_path = tmp_path / "fake_model.joblib"
    joblib.dump(fake_instance, fake_path)

    with pytest.raises(ValueError, match="Loaded object is not a CustomerClustering instance"):
        CustomerClustering.load(fake_path)
