"""
Prediction utilities for Task 1: House Price Prediction.

This module provides functions to load trained models and make
predictions on new data, with input validation and error handling.
"""

from typing import Dict, List, Union, Optional
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline

from data_loader import BASELINE_FEATURES, TARGET_COLUMN


class HousePricePredictor:
    """
    Wrapper for loading and using trained house price prediction models.
    """

    def __init__(self, model_path: str):
        """
        Initialize predictor with saved model.

        Args:
            model_path: Path to saved joblib pipeline

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If loaded object is not a valid pipeline
        """
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.pipeline = joblib.load(model_path)

        if not isinstance(self.pipeline, Pipeline):
            raise ValueError("Loaded model is not a sklearn Pipeline")

        # Check pipeline has expected steps
        if "preprocessor" not in self.pipeline.named_steps:
            raise ValueError("Pipeline missing 'preprocessor' step")
        if "model" not in self.pipeline.named_steps:
            raise ValueError("Pipeline missing 'model' step")

        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.model = self.pipeline.named_steps["model"]

        print(f"✓ Loaded model from {model_path}")
        print(f"  Model type: {type(self.model).__name__}")

    def validate_input(self, input_data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Validate and convert input to DataFrame with correct columns.

        Args:
            input_data: Dictionary or DataFrame with feature values

        Returns:
            Validated DataFrame

        Raises:
            ValueError: If required features missing or invalid values
        """
        # Convert to DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise ValueError("Input must be dict or DataFrame")

        # Check required features
        missing = [f for f in BASELINE_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # Select only baseline features in correct order
        df = df[BASELINE_FEATURES].copy()

        # Validate values
        for feat in BASELINE_FEATURES:
            val = df[feat].iloc[0]
            if pd.isna(val):
                raise ValueError(f"Feature '{feat}' cannot be NaN")
            if not isinstance(val, (int, float, np.number)):
                raise ValueError(f"Feature '{feat}' must be numeric, got {type(val)}")
            if val < 0:
                raise ValueError(f"Feature '{feat}' must be non-negative, got {val}")

        # Reasonable range checks (based on Ames Housing data)
        if df["GrLivArea"].iloc[0] > 10000:
            raise ValueError(f"GrLivArea seems unreasonably large: {df['GrLivArea'].iloc[0]}")
        if df["BedroomAbvGr"].iloc[0] > 20:
            raise ValueError(f"BedroomAbvGr seems unreasonably large: {df['BedroomAbvGr'].iloc[0]}")
        if df["FullBath"].iloc[0] > 10:
            raise ValueError(f"FullBath seems unreasonably large: {df['FullBath'].iloc[0]}")

        return df

    def predict(self, input_data: Union[Dict, pd.DataFrame]) -> float:
        """
        Predict house price for given features.

        Args:
            input_data: Dictionary or DataFrame with GrLivArea, BedroomAbvGr, FullBath

        Returns:
            Predicted sale price in dollars

        Raises:
            ValueError: If input validation fails
        """
        # Validate and prepare input
        df = self.validate_input(input_data)

        # Preprocess
        X_processed = self.preprocessor.transform(df)

        # Predict
        prediction = self.model.predict(X_processed)[0]

        # Ensure non-negative
        prediction = max(0, float(prediction))

        return prediction

    def predict_batch(self, input_data: Union[List[Dict], pd.DataFrame]) -> List[float]:
        """
        Predict prices for multiple houses using vectorized inference.

        Args:
            input_data: List of dictionaries or DataFrame with features

        Returns:
            List of predicted prices
        """
        # Handle empty input
        if isinstance(input_data, list):
            if len(input_data) == 0:
                return []
        elif isinstance(input_data, pd.DataFrame):
            if input_data.empty:
                return []
        else:
            raise ValueError("Input must be list of dicts or DataFrame")

        # Convert to DataFrame for vectorized processing
        df = pd.DataFrame(input_data)

        # Validate all rows at once
        missing = [f for f in BASELINE_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # Select only baseline features in correct order
        df = df[BASELINE_FEATURES].copy()

        # Validate values (vectorized checks)
        for feat in BASELINE_FEATURES:
            if df[feat].isna().any():
                raise ValueError(f"Feature '{feat}' cannot contain NaN values")
            if not pd.api.types.is_numeric_dtype(df[feat]):
                raise ValueError(f"Feature '{feat}' must be numeric")
            if (df[feat] < 0).any():
                raise ValueError(f"Feature '{feat}' must be non-negative")

        # Reasonable range checks
        if (df["GrLivArea"] > 10000).any():
            raise ValueError("GrLivArea contains unreasonably large values (>10000)")
        if (df["BedroomAbvGr"] > 20).any():
            raise ValueError("BedroomAbvGr contains unreasonably large values (>20)")
        if (df["FullBath"] > 10).any():
            raise ValueError("FullBath contains unreasonably large values (>10)")

        # Vectorized preprocessing and prediction
        X_processed = self.preprocessor.transform(df)
        predictions = self.model.predict(X_processed)

        # Ensure non-negative
        predictions = np.maximum(predictions, 0)

        return predictions.tolist()

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model details
        """
        info = {
            "model_type": type(self.model).__name__,
            "features": BASELINE_FEATURES,
            "target": TARGET_COLUMN,
            "model_path": str(self.model_path)
        }

        # Add coefficients if linear model
        if hasattr(self.model, "coef_"):
            info["coefficients"] = dict(zip(BASELINE_FEATURES, self.model.coef_))
            info["intercept"] = float(self.model.intercept_)

        return info


def create_sample_input() -> Dict:
    """
    Create a sample input for testing.

    Returns:
        Dictionary with typical values
    """
    return {
        "GrLivArea": 1500,
        "BedroomAbvGr": 3,
        "FullBath": 2
    }


def predict_from_csv(
    model_path: str,
    csv_path: str,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Make predictions from a CSV file.

    Args:
        model_path: Path to saved model
        csv_path: Path to input CSV with columns GrLivArea, BedroomAbvGr, FullBath
        output_path: Optional path to save predictions CSV

    Returns:
        DataFrame with original data + predictions
    """
    predictor = HousePricePredictor(model_path)
    df = pd.read_csv(csv_path)

    # Validate required columns
    missing = [f for f in BASELINE_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    # Make predictions
    predictions = []
    for _, row in df.iterrows():
        try:
            pred = predictor.predict(row.to_dict())
            predictions.append(pred)
        except ValueError as e:
            print(f"Warning: Skipping row {_} due to error: {e}")
            predictions.append(np.nan)

    df["Predicted_SalePrice"] = predictions

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"✓ Predictions saved to {output_path}")

    return df


if __name__ == "__main__":
    # Quick test - will fail if no model exists, that's expected
    import sys

    model_path = "models/baseline_pipeline.joblib"

    if Path(model_path).exists():
        predictor = HousePricePredictor(model_path)
        sample = create_sample_input()
        price = predictor.predict(sample)
        print(f"\nSample prediction: ${price:,.2f}")
        print(f"Model info: {predictor.get_model_info()}")
    else:
        print(f"Model not found at {model_path}. Run train.py first.")
        sys.exit(1)