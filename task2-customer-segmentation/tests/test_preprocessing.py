"""Unit tests for src.preprocessing module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.data_loader import load_data
from src.preprocessing import CustomerPreprocessor, prepare_features


@pytest.fixture
def sample_customer_df() -> pd.DataFrame:
    """Fixture providing a deterministic sample customer DataFrame."""
    return pd.DataFrame({
        "CustomerID": [101, 102, 103, 104, 105],
        "Gender": ["Male", "Female", "Female", "Male", "Female"],
        "Age": [20, 30, 40, 50, 60],
        "Annual_Income": [20, 40, 60, 80, 100],
        "Spending_Score": [10, 30, 50, 70, 90],
    })


def test_classic_feature_set(sample_customer_df):
    """Verify classic feature set selects only Annual_Income and Spending_Score."""
    prep = CustomerPreprocessor(feature_set="classic")
    X = prep.fit_transform(sample_customer_df)

    assert X.shape == (5, 2)
    assert prep.get_feature_names() == ["Annual_Income", "Spending_Score"]


def test_extended_feature_set(sample_customer_df):
    """Verify extended feature set includes Age, Annual_Income, Spending_Score, and encoded Gender."""
    prep = CustomerPreprocessor(feature_set="extended")
    X = prep.fit_transform(sample_customer_df)

    assert X.shape == (5, 4)
    assert prep.get_feature_names() == [
        "Age",
        "Annual_Income",
        "Spending_Score",
        "Gender_Male",
    ]


def test_customer_id_strictly_excluded(sample_customer_df):
    """Verify CustomerID is never included in the transformed features."""
    for fset in ["classic", "extended"]:
        prep = CustomerPreprocessor(feature_set=fset)
        _ = prep.fit_transform(sample_customer_df)
        feature_names = prep.get_feature_names()
        assert "CustomerID" not in feature_names
        assert all("customer" not in fn.lower() for fn in feature_names)


def test_scaling_mean_and_unit_variance():
    """Verify StandardScaler normalizes numeric features to approximately zero mean and unit variance."""
    df = load_data()
    prep = CustomerPreprocessor(feature_set="classic")
    X = prep.fit_transform(df)

    # For sample size 200, mean should be ~0 and std ~1
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)

    np.testing.assert_allclose(means, [0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(stds, [1.0, 1.0], atol=1e-7)


def test_gender_encoding_values(sample_customer_df):
    """Verify Male is encoded as 1.0 and Female as 0.0 in Gender_Male indicator."""
    prep = CustomerPreprocessor(feature_set="extended")
    df_transformed = prep.fit_transform(sample_customer_df)
    gender_col_idx = prep.get_feature_names().index("Gender_Male")
    gender_values = df_transformed[:, gender_col_idx]

    # sample_customer_df genders: ["Male", "Female", "Female", "Male", "Female"]
    expected_gender_encoded = np.array([1.0, 0.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(gender_values, expected_gender_encoded)


def test_row_count_and_order_preservation(sample_customer_df):
    """Verify row count and exact order are preserved through transformation."""
    custom_index = ["c_0", "c_1", "c_2", "c_3", "c_4"]
    df_indexed = sample_customer_df.copy()
    df_indexed.index = custom_index

    prep = CustomerPreprocessor(feature_set="extended")
    prep.fit(df_indexed)
    df_result = prep.transform_df(df_indexed)

    assert len(df_result) == 5
    assert list(df_result.index) == custom_index
    assert list(df_result.columns) == prep.get_feature_names()

    # Also verify fit_transform_df
    prep2 = CustomerPreprocessor(feature_set="extended")
    df_result2 = prep2.fit_transform_df(df_indexed)
    assert list(df_result2.index) == custom_index
    pd.testing.assert_frame_equal(df_result, df_result2)


def test_transform_before_fit_raises_error(sample_customer_df):
    """Verify transform raises NotFittedError if fit has not been called."""
    prep = CustomerPreprocessor(feature_set="classic")
    with pytest.raises(NotFittedError):
        prep.transform(sample_customer_df)


def test_get_feature_names_before_fit_raises_error():
    """Verify get_feature_names raises NotFittedError before fit."""
    prep = CustomerPreprocessor(feature_set="classic")
    with pytest.raises(NotFittedError):
        prep.get_feature_names()


def test_invalid_feature_set_name_raises_value_error():
    """Verify unsupported feature set names raise ValueError."""
    with pytest.raises(ValueError, match="Invalid feature_set"):
        CustomerPreprocessor(feature_set="super_segmentation")

    with pytest.raises(ValueError, match="must be a string"):
        CustomerPreprocessor(feature_set=123)  # type: ignore


def test_missing_feature_column_raises_value_error():
    """Verify missing required feature columns raise informative ValueError."""
    prep = CustomerPreprocessor(feature_set="extended")
    # DataFrame missing 'Gender'
    df_missing_gender = pd.DataFrame({
        "Age": [25],
        "Annual_Income": [50],
        "Spending_Score": [60],
    })
    with pytest.raises(ValueError, match="Missing required columns.*Gender"):
        prep.fit(df_missing_gender)


def test_no_data_leakage():
    """Verify strict separation between fit and transform prevents data leakage."""
    # Split deterministic dataset
    train_df = pd.DataFrame({
        "Annual_Income": [20.0, 30.0, 40.0, 50.0],
        "Spending_Score": [10.0, 20.0, 30.0, 40.0],
    })
    test_df = pd.DataFrame({
        "Annual_Income": [100.0, 200.0],
        "Spending_Score": [80.0, 90.0],
    })

    prep = CustomerPreprocessor(feature_set="classic")
    X_train = prep.fit_transform(train_df)
    X_test = prep.transform(test_df)

    # Train mean is 0
    np.testing.assert_allclose(np.mean(X_train, axis=0), [0.0, 0.0], atol=1e-7)

    # Test transformed with train scaler must NOT have zero mean (shows train params were used)
    test_means = np.mean(X_test, axis=0)
    assert not np.allclose(test_means, [0.0, 0.0])
    # Values should be substantially positive since test income/spending is higher than train
    assert (test_means > 0).all()


def test_prepare_features_functional_helper(sample_customer_df):
    """Verify prepare_features functional wrapper works for both fit=True and fit=False."""
    arr, prep, fnames = prepare_features(sample_customer_df, feature_set="classic", fit=True)
    assert arr.shape == (5, 2)
    assert fnames == ["Annual_Income", "Spending_Score"]
    assert prep.is_fitted_

    # Transform new data using existing prep
    arr_new, _, _ = prepare_features(
        sample_customer_df, feature_set="classic", fit=False, preprocessor=prep
    )
    np.testing.assert_array_equal(arr, arr_new)

    with pytest.raises(ValueError, match="Existing preprocessor must be provided"):
        prepare_features(sample_customer_df, feature_set="classic", fit=False, preprocessor=None)


def test_empty_dataframe_raises_value_error():
    """Verify empty DataFrame input raises ValueError."""
    prep = CustomerPreprocessor(feature_set="classic")
    empty_df = pd.DataFrame(columns=["Annual_Income", "Spending_Score"])
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        prep.fit(empty_df)
