"""
Unit tests for training pipeline.

Run with: python -m pytest tests/test_training.py -v
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import joblib
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import create_baseline_preprocessor
from evaluation import compute_regression_metrics, compute_all_metrics
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


class TestMetricsComputation:
    """Tests for evaluation metrics."""

    def test_compute_regression_metrics_basic(self):
        """Test basic metric computation."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])

        metrics = compute_regression_metrics(y_true, y_pred, prefix="Test_")

        assert "Test_MAE" in metrics
        assert "Test_RMSE" in metrics
        assert "Test_R2" in metrics
        assert "Test_MAPE" in metrics
        assert "Test_MedAE" in metrics

        # All values should be finite floats
        for v in metrics.values():
            assert np.isfinite(v)

        # MAE should be 10 for this case
        assert metrics["Test_MAE"] == 10.0

    def test_compute_regression_metrics_perfect(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])

        metrics = compute_regression_metrics(y_true, y_pred)

        assert metrics["MAE"] == 0.0
        assert metrics["RMSE"] == 0.0
        assert metrics["R2"] == 1.0

    def test_compute_regression_metrics_shape_mismatch(self):
        """Test error on shape mismatch."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200])

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_regression_metrics(y_true, y_pred)

    def test_compute_regression_metrics_nan(self):
        """Test error on NaN values."""
        y_true = np.array([100, np.nan, 300])
        y_pred = np.array([100, 200, 300])

        with pytest.raises(ValueError, match="NaN"):
            compute_regression_metrics(y_true, y_pred)

    def test_compute_all_metrics(self):
        """Test combined dollar and log metrics."""
        y_true = np.array([100000, 200000, 300000, 400000, 500000])
        y_pred = np.array([110000, 190000, 310000, 390000, 510000])

        metrics = compute_all_metrics(y_true, y_pred, prefix="Test_")

        # Should have both dollar and log metrics
        dollar_keys = [k for k in metrics if not k.startswith("Test_Log_")]
        log_keys = [k for k in metrics if k.startswith("Test_Log_")]

        assert len(dollar_keys) >= 5  # MAE, RMSE, R2, MAPE, MedAE
        assert len(log_keys) >= 5

    def test_compute_log_metrics(self):
        """Test log-scale metrics."""
        from evaluation import compute_log_metrics
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])

        # compute_log_metrics adds "Log_" prefix internally, so use empty prefix
        metrics = compute_log_metrics(y_true, y_pred, prefix="")

        assert "Log_MAE" in metrics
        assert "Log_RMSE" in metrics
        assert "Log_R2" in metrics


class TestTrainingPipeline:
    """Tests for training pipeline integration."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 100

        df = pd.DataFrame({
            "GrLivArea": np.random.normal(1500, 500, n).astype(int),
            "BedroomAbvGr": np.random.choice([2, 3, 4, 5], n),
            "FullBath": np.random.choice([1, 2, 3], n),
            "SalePrice": np.random.lognormal(12, 0.3, n)
        })
        return df

    def test_pipeline_fit_predict(self, sample_data):
        """Test full pipeline: preprocess -> model -> predict."""
        X = sample_data[["GrLivArea", "BedroomAbvGr", "FullBath"]]
        y = sample_data["SalePrice"]

        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create pipeline
        preprocessor = create_baseline_preprocessor()
        model = LinearRegression()

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # Fit
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_val)

        # Check predictions
        assert len(y_pred) == len(y_val)
        assert not np.any(np.isnan(y_pred))
        assert np.all(y_pred >= 0)  # Prices should be non-negative

    def test_pipeline_save_load(self, sample_data):
        """Test saving and loading pipeline."""
        X = sample_data[["GrLivArea", "BedroomAbvGr", "FullBath"]]
        y = sample_data["SalePrice"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline = Pipeline([
            ("preprocessor", create_baseline_preprocessor()),
            ("model", LinearRegression())
        ])

        pipeline.fit(X_train, y_train)
        preds_before = pipeline.predict(X_val)

        # Save and load
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = f.name

        try:
            joblib.dump(pipeline, temp_path)
            loaded_pipeline = joblib.load(temp_path)

            preds_after = loaded_pipeline.predict(X_val)

            # Predictions should be identical
            np.testing.assert_allclose(preds_before, preds_after)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_preprocessor_fit_only_on_train(self, sample_data):
        """Test that preprocessor is only fit on training data."""
        X = sample_data[["GrLivArea", "BedroomAbvGr", "FullBath"]]
        y = sample_data["SalePrice"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Introduce different statistics in validation
        X_val_modified = X_val.copy()
        X_val_modified["GrLivArea"] = X_val_modified["GrLivArea"] * 10  # Much larger

        preprocessor = create_baseline_preprocessor()
        preprocessor.fit(X_train)

        # Transform both
        X_train_trans = preprocessor.transform(X_train)
        X_val_trans = preprocessor.transform(X_val_modified)

        # Validation should be scaled using TRAIN statistics
        # So mean should not be 0 for validation (since it has different distribution)
        # But no error should occur
        assert X_train_trans.shape[1] == 3
        assert X_val_trans.shape[1] == 3
        assert not np.any(np.isnan(X_train_trans))
        assert not np.any(np.isnan(X_val_trans))

    def test_reproducibility(self, sample_data):
        """Test that same seed gives same results."""
        X = sample_data[["GrLivArea", "BedroomAbvGr", "FullBath"]]
        y = sample_data["SalePrice"]

        # Train twice with same seed
        X_train1, X_val1, y_train1, y_val1 = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train2, X_val2, y_train2, y_val2 = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Splits should be identical
        assert X_train1.index.equals(X_train2.index)
        assert X_val1.index.equals(X_val2.index)

        # Train models
        pipeline1 = Pipeline([
            ("preprocessor", create_baseline_preprocessor()),
            ("model", LinearRegression())
        ])
        pipeline2 = Pipeline([
            ("preprocessor", create_baseline_preprocessor()),
            ("model", LinearRegression())
        ])

        pipeline1.fit(X_train1, y_train1)
        pipeline2.fit(X_train2, y_train2)

        # Coefficients should be identical
        model1 = pipeline1.named_steps["model"]
        model2 = pipeline2.named_steps["model"]

        np.testing.assert_allclose(model1.coef_, model2.coef_)
        np.testing.assert_allclose(model1.intercept_, model2.intercept_)


class TestMetricsFinite:
    """Test that metrics are always finite."""

    def test_metrics_finite_positive(self):
        """Test metrics are finite and positive where expected."""
        np.random.seed(42)
        y_true = np.random.lognormal(12, 0.5, 50)
        y_pred = y_true * np.random.lognormal(0, 0.1, 50)

        metrics = compute_all_metrics(y_true, y_pred)

        # MAE, RMSE, MedAE, MAPE should be positive
        assert metrics["MAE"] > 0
        assert metrics["RMSE"] > 0
        assert metrics["MedAE"] > 0
        assert metrics["MAPE"] > 0

        # R2 can be negative but should be finite
        assert np.isfinite(metrics["R2"])

        # Log metrics
        assert metrics["Log_MAE"] > 0
        assert metrics["Log_RMSE"] > 0
        assert np.isfinite(metrics["Log_R2"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])