"""
Unit tests for preprocessing module.

Run with: python -m pytest tests/test_preprocessing.py -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import (
    FeatureSelector,
    LogTargetTransformer,
    create_baseline_preprocessor,
    identify_feature_types,
    create_full_preprocessor,
    create_full_pipeline,
    get_feature_names_after_preprocessing,
    BASELINE_FEATURES
)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


class TestFeatureSelector:
    """Tests for FeatureSelector transformer."""

    def test_fit_transform(self):
        """Test basic fit and transform."""
        df = pd.DataFrame({
            "GrLivArea": [1500, 2000],
            "BedroomAbvGr": [3, 4],
            "FullBath": [2, 3],
            "ExtraCol": [1, 2]
        })

        selector = FeatureSelector(BASELINE_FEATURES)
        selector.fit(df)
        result = selector.transform(df)

        assert list(result.columns) == BASELINE_FEATURES
        assert result.shape == (2, 3)

    def test_fit_missing_column(self):
        """Test fit fails when column missing."""
        df = pd.DataFrame({"GrLivArea": [1500], "BedroomAbvGr": [3]})
        selector = FeatureSelector(BASELINE_FEATURES)

        with pytest.raises(ValueError, match="not found"):
            selector.fit(df)

    def test_get_feature_names_out(self):
        """Test get_feature_names_out returns correct names."""
        selector = FeatureSelector(BASELINE_FEATURES)
        names = selector.get_feature_names_out()
        assert list(names) == BASELINE_FEATURES


class TestLogTargetTransformer:
    """Tests for LogTargetTransformer."""

    def test_fit_transform_inverse(self):
        """Test log transform and inverse."""
        y = np.array([100000, 200000, 300000, 400000, 500000])

        transformer = LogTargetTransformer()
        transformer.fit(y)
        y_log = transformer.transform(y)
        y_reconstructed = transformer.inverse_transform(y_log)

        np.testing.assert_allclose(y, y_reconstructed, rtol=1e-10)

    def test_fit_negative_values(self):
        """Test fit fails with non-positive values."""
        y = np.array([100000, -50000, 300000])
        transformer = LogTargetTransformer()

        with pytest.raises(ValueError, match="positive"):
            transformer.fit(y)


class TestBaselinePreprocessor:
    """Tests for baseline preprocessing pipeline."""

    def test_create_baseline_preprocessor(self):
        """Test creating baseline preprocessor."""
        preprocessor = create_baseline_preprocessor()
        assert isinstance(preprocessor, Pipeline)
        assert "selector" in preprocessor.named_steps
        assert "imputer" in preprocessor.named_steps
        assert "scaler" in preprocessor.named_steps

    def test_baseline_preprocessor_fit_transform(self):
        """Test baseline preprocessor fit/transform."""
        df = pd.DataFrame({
            "GrLivArea": [1500, 2000, np.nan, 1800],
            "BedroomAbvGr": [3, 4, 2, 3],
            "FullBath": [2, 3, 2, 2],
            "ExtraCol": [1, 2, 3, 4]
        })

        preprocessor = create_baseline_preprocessor()
        X_transformed = preprocessor.fit_transform(df)

        # Should have 3 features (baseline), no NaN
        assert X_transformed.shape == (4, 3)
        assert not np.any(np.isnan(X_transformed))

        # Check scaling (mean ~0, std ~1)
        assert np.allclose(X_transformed.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_transformed.std(axis=0), 1, atol=1e-10)

    def test_baseline_preprocessor_transform_only(self):
        """Test transform on new data after fitting."""
        df_train = pd.DataFrame({
            "GrLivArea": [1500, 2000, 1800],
            "BedroomAbvGr": [3, 4, 3],
            "FullBath": [2, 3, 2],
        })

        df_test = pd.DataFrame({
            "GrLivArea": [1600, 2200],
            "BedroomAbvGr": [3, 4],
            "FullBath": [2, 3],
        })

        preprocessor = create_baseline_preprocessor()
        preprocessor.fit(df_train)

        X_test_transformed = preprocessor.transform(df_test)
        assert X_test_transformed.shape == (2, 3)
        assert not np.any(np.isnan(X_test_transformed))


class TestIdentifyFeatureTypes:
    """Tests for feature type identification."""

    def test_identify_feature_types(self):
        """Test automatic feature type detection."""
        df = pd.DataFrame({
            "GrLivArea": [1500, 2000, 1800],      # numeric
            "LotArea": [8000, 10000, 9000],        # numeric
            "BedroomAbvGr": [3, 4, 3],             # low cardinality numeric -> categorical
            "FullBath": [2, 3, 2],                 # low cardinality numeric -> categorical
            "Neighborhood": ["A", "B", "A"],        # categorical
            "OverallQual": [7, 8, 7],              # low cardinality numeric -> categorical
            "SalePrice": [200000, 300000, 250000], # target
            "Id": [1, 2, 3]                        # ID
        })

        numeric, categorical = identify_feature_types(df)

        # GrLivArea and LotArea should be numeric (high cardinality)
        assert "GrLivArea" in numeric
        assert "LotArea" in numeric

        # BedroomAbvGr, FullBath, OverallQual should be categorical (low cardinality)
        assert "BedroomAbvGr" in categorical
        assert "FullBath" in categorical
        assert "OverallQual" in categorical

        # Neighborhood should be categorical
        assert "Neighborhood" in categorical

        # SalePrice and Id excluded
        assert "SalePrice" not in numeric and "SalePrice" not in categorical
        assert "Id" not in numeric and "Id" not in categorical


class TestFullPreprocessor:
    """Tests for full feature preprocessor."""

    def test_create_full_preprocessor(self):
        """Test creating full preprocessor."""
        df = pd.DataFrame({
            "GrLivArea": [1500, 2000],
            "BedroomAbvGr": [3, 4],
            "Neighborhood": ["A", "B"],
            "SalePrice": [200000, 300000],
            "Id": [1, 2]
        })

        preprocessor = create_full_preprocessor(df)
        # ColumnTransformer created successfully

    def test_full_preprocessor_fit_transform(self):
        """Test full preprocessor on mixed data."""
        df = pd.DataFrame({
            "GrLivArea": [1500, 2000, 1800, np.nan],
            "LotArea": [8000, 10000, 9000, 12000],
            "Neighborhood": ["A", "B", "A", "C"],
            "BedroomAbvGr": [3, 4, 3, 2],
            "SalePrice": [200000, 300000, 250000, 350000],
            "Id": [1, 2, 3, 4]
        })

        preprocessor = create_full_preprocessor(df)
        X_transformed = preprocessor.fit_transform(df.drop(["SalePrice", "Id"], axis=1))

        # Should have numeric + one-hot encoded categorical
        assert X_transformed.shape[0] == 4
        assert X_transformed.shape[1] > 0
        assert not np.any(np.isnan(X_transformed))

    def test_full_preprocessor_handle_unknown(self):
        """Test OneHotEncoder handles unknown categories."""
        df_train = pd.DataFrame({
            "GrLivArea": [1500, 2000],
            "Neighborhood": ["A", "B"],
            "SalePrice": [200000, 300000],
            "Id": [1, 2]
        })

        df_test = pd.DataFrame({
            "GrLivArea": [1800],
            "Neighborhood": ["C"],  # Unknown category
            "SalePrice": [250000],
            "Id": [3]
        })

        preprocessor = create_full_preprocessor(df_train)
        preprocessor.fit(df_train.drop(["SalePrice", "Id"], axis=1))

        # Should not raise error for unknown category
        X_test = preprocessor.transform(df_test.drop(["SalePrice", "Id"], axis=1))
        assert X_test.shape[0] == 1


class TestFullPipeline:
    """Tests for complete pipeline creation."""

    def test_create_full_pipeline(self):
        """Test creating full pipeline with estimator."""
        df = pd.DataFrame({
            "GrLivArea": [1500, 2000, 1800],
            "Neighborhood": ["A", "B", "A"],
            "SalePrice": [200000, 300000, 250000],
            "Id": [1, 2, 3]
        })

        model = LinearRegression()
        pipeline = create_full_pipeline(df, model)

        assert isinstance(pipeline, Pipeline)
        assert "preprocessor" in pipeline.named_steps
        assert "estimator" in pipeline.named_steps

    def test_full_pipeline_fit_predict(self):
        """Test full pipeline fit and predict."""
        df = pd.DataFrame({
            "GrLivArea": [1500, 2000, 1800, 2200, 1600],
            "LotArea": [8000, 10000, 9000, 12000, 8500],
            "Neighborhood": ["A", "B", "A", "C", "B"],
            "BedroomAbvGr": [3, 4, 3, 4, 3],
            "SalePrice": [200000, 300000, 250000, 350000, 220000],
            "Id": [1, 2, 3, 4, 5]
        })

        model = LinearRegression()
        pipeline = create_full_pipeline(df, model)

        X = df.drop(["SalePrice", "Id"], axis=1)
        y = df["SalePrice"]

        pipeline.fit(X, y)
        preds = pipeline.predict(X)

        assert len(preds) == 5
        assert not np.any(np.isnan(preds))


class TestGetFeatureNames:
    """Tests for feature name extraction."""

    def test_get_feature_names_after_preprocessing(self):
        """Test getting feature names from fitted preprocessor."""
        df = pd.DataFrame({
            "GrLivArea": [1500, 2000],
            "LotArea": [8000, 10000],
            "Neighborhood": ["A", "B"],
            "SalePrice": [200000, 300000],
            "Id": [1, 2]
        })

        preprocessor = create_full_preprocessor(df)
        preprocessor.fit(df.drop(["SalePrice", "Id"], axis=1))

        numeric_features = ["GrLivArea", "LotArea"]
        categorical_features = ["Neighborhood"]

        names = get_feature_names_after_preprocessing(
            preprocessor, numeric_features, categorical_features
        )

        # Should have numeric + one-hot encoded categorical
        assert "GrLivArea" in names
        assert "LotArea" in names
        assert any("Neighborhood" in n for n in names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])