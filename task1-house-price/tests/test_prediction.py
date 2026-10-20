"""
Unit tests for prediction module.

Run with: python -m pytest tests/test_prediction.py -v
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

from predict import HousePricePredictor, create_sample_input, predict_from_csv
from preprocessing import create_baseline_preprocessor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


class TestHousePricePredictor:
    """Tests for HousePricePredictor class."""

    @pytest.fixture
    def trained_model_path(self):
        """Create a temporary trained model for testing."""
        # Create sample data and train
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            "GrLivArea": np.random.normal(1500, 300, n).astype(int),
            "BedroomAbvGr": np.random.choice([2, 3, 4], n),
            "FullBath": np.random.choice([1, 2, 3], n),
            "SalePrice": np.random.lognormal(12, 0.2, n)
        })

        X = df[["GrLivArea", "BedroomAbvGr", "FullBath"]]
        y = df["SalePrice"]

        pipeline = Pipeline([
            ("preprocessor", create_baseline_preprocessor()),
            ("model", LinearRegression())
        ])
        pipeline.fit(X, y)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = f.name

        joblib.dump(pipeline, temp_path)
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)

    def test_init_loads_model(self, trained_model_path):
        """Test predictor initializes and loads model."""
        predictor = HousePricePredictor(trained_model_path)
        assert predictor.pipeline is not None
        assert predictor.preprocessor is not None
        assert predictor.model is not None

    def test_init_missing_file(self):
        """Test predictor fails on missing model file."""
        with pytest.raises(FileNotFoundError):
            HousePricePredictor("nonexistent_model.joblib")

    def test_validate_input_dict(self, trained_model_path):
        """Test input validation with dictionary."""
        predictor = HousePricePredictor(trained_model_path)

        valid_input = {"GrLivArea": 1500, "BedroomAbvGr": 3, "FullBath": 2}
        df = predictor.validate_input(valid_input)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["GrLivArea", "BedroomAbvGr", "FullBath"]
        assert len(df) == 1

    def test_validate_input_dataframe(self, trained_model_path):
        """Test input validation with DataFrame."""
        predictor = HousePricePredictor(trained_model_path)

        valid_input = pd.DataFrame([{"GrLivArea": 1500, "BedroomAbvGr": 3, "FullBath": 2}])
        df = predictor.validate_input(valid_input)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_validate_input_missing_feature(self, trained_model_path):
        """Test validation fails on missing feature."""
        predictor = HousePricePredictor(trained_model_path)

        invalid_input = {"GrLivArea": 1500, "BedroomAbvGr": 3}  # Missing FullBath

        with pytest.raises(ValueError, match="Missing required features"):
            predictor.validate_input(invalid_input)

    def test_validate_input_nan(self, trained_model_path):
        """Test validation fails on NaN values."""
        predictor = HousePricePredictor(trained_model_path)

        invalid_input = {"GrLivArea": 1500, "BedroomAbvGr": np.nan, "FullBath": 2}

        with pytest.raises(ValueError, match="cannot be NaN"):
            predictor.validate_input(invalid_input)

    def test_validate_input_negative(self, trained_model_path):
        """Test validation fails on negative values."""
        predictor = HousePricePredictor(trained_model_path)

        invalid_input = {"GrLivArea": -100, "BedroomAbvGr": 3, "FullBath": 2}

        with pytest.raises(ValueError, match="non-negative"):
            predictor.validate_input(invalid_input)

    def test_validate_input_unreasonable(self, trained_model_path):
        """Test validation fails on unreasonable values."""
        predictor = HousePricePredictor(trained_model_path)

        invalid_input = {"GrLivArea": 50000, "BedroomAbvGr": 3, "FullBath": 2}

        with pytest.raises(ValueError, match="unreasonably large"):
            predictor.validate_input(invalid_input)

    def test_predict_returns_float(self, trained_model_path):
        """Test predict returns a float."""
        predictor = HousePricePredictor(trained_model_path)

        sample = create_sample_input()
        price = predictor.predict(sample)

        assert isinstance(price, float)
        assert price >= 0

    def test_predict_batch(self, trained_model_path):
        """Test batch prediction."""
        predictor = HousePricePredictor(trained_model_path)

        samples = [
            {"GrLivArea": 1500, "BedroomAbvGr": 3, "FullBath": 2},
            {"GrLivArea": 2000, "BedroomAbvGr": 4, "FullBath": 3},
            {"GrLivArea": 1800, "BedroomAbvGr": 3, "FullBath": 2}
        ]

        prices = predictor.predict_batch(samples)

        assert len(prices) == 3
        assert all(isinstance(p, float) for p in prices)
        assert all(p >= 0 for p in prices)

    def test_get_model_info(self, trained_model_path):
        """Test getting model info."""
        predictor = HousePricePredictor(trained_model_path)
        info = predictor.get_model_info()

        assert "model_type" in info
        assert "features" in info
        assert "target" in info
        assert "model_path" in info
        assert info["features"] == ["GrLivArea", "BedroomAbvGr", "FullBath"]


class TestCreateSampleInput:
    """Tests for create_sample_input function."""

    def test_create_sample_input(self):
        """Test sample input creation."""
        sample = create_sample_input()

        assert isinstance(sample, dict)
        assert "GrLivArea" in sample
        assert "BedroomAbvGr" in sample
        assert "FullBath" in sample
        assert all(isinstance(v, (int, float)) for v in sample.values())
        assert all(v > 0 for v in sample.values())


class TestPredictFromCSV:
    """Tests for predict_from_csv function."""

    @pytest.fixture
    def trained_model_path(self):
        """Create a temporary trained model for testing."""
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            "GrLivArea": np.random.normal(1500, 300, n).astype(int),
            "BedroomAbvGr": np.random.choice([2, 3, 4], n),
            "FullBath": np.random.choice([1, 2, 3], n),
            "SalePrice": np.random.lognormal(12, 0.2, n)
        })

        X = df[["GrLivArea", "BedroomAbvGr", "FullBath"]]
        y = df["SalePrice"]

        pipeline = Pipeline([
            ("preprocessor", create_baseline_preprocessor()),
            ("model", LinearRegression())
        ])
        pipeline.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = f.name

        joblib.dump(pipeline, temp_path)
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)

    def test_predict_from_csv(self, trained_model_path):
        """Test CSV prediction."""
        # Create test CSV
        test_df = pd.DataFrame({
            "GrLivArea": [1500, 2000, 1800],
            "BedroomAbvGr": [3, 4, 3],
            "FullBath": [2, 3, 2]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
            test_df.to_csv(csv_path, index=False)

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            output_path = f.name

        try:
            result = predict_from_csv(trained_model_path, csv_path, output_path)

            assert isinstance(result, pd.DataFrame)
            assert "Predicted_SalePrice" in result.columns
            assert len(result) == 3
            assert not result["Predicted_SalePrice"].isna().any()
            assert Path(output_path).exists()
        finally:
            Path(csv_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def test_predict_from_csv_missing_columns(self, trained_model_path):
        """Test CSV prediction fails on missing columns."""
        test_df = pd.DataFrame({
            "GrLivArea": [1500],
            "BedroomAbvGr": [3]  # Missing FullBath
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
            test_df.to_csv(csv_path, index=False)

        try:
            with pytest.raises(ValueError, match="missing required columns"):
                predict_from_csv(trained_model_path, csv_path)
        finally:
            Path(csv_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])