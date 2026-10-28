"""K-Means clustering module for Mall Customer Segmentation.

This module provides a reproducible, robust clustering engine that:
- Integrates seamlessly with CustomerPreprocessor for 'classic' and 'extended' feature sets.
- Strictly excludes CustomerID and maintains exact row-order and index alignment.
- Employs deterministic random seeds (default random_state=42) and explicit hyperparameter settings.
- Validates inputs, ensuring cluster count k satisfies 2 <= k < n_samples.
- Calculates cluster centroids in both standardized feature space and interpretable original units.
- Supports model persistence (joblib save/load) and reproducible new-customer inference.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError

try:
    from src.preprocessing import CustomerPreprocessor
except ImportError:
    from preprocessing import CustomerPreprocessor


class CustomerClustering(BaseEstimator, ClusterMixin):
    """K-Means clustering pipeline wrapper for Mall Customer Segmentation.

    Attributes:
        n_clusters: Number of clusters (k).
        feature_set: 'classic' or 'extended'.
        random_state: Random state for deterministic centroid initialization.
        n_init: Number of time K-Means algorithm will run with different centroid seeds.
        max_iter: Maximum number of iterations for a single run.
        algorithm: K-Means algorithm implementation (default: 'lloyd').
        preprocessor_: Fitted CustomerPreprocessor instance.
        kmeans_: Fitted scikit-learn KMeans estimator.
        labels_: Array of integer cluster labels assigned to training samples.
        inertia_: Sum of squared distances of samples to their closest cluster center.
        is_fitted_: Boolean indicating whether model has been trained.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        feature_set: str = "classic",
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300,
        algorithm: str = "lloyd",
    ) -> None:
        self.n_clusters = n_clusters
        self.feature_set = feature_set
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.algorithm = algorithm

        self.preprocessor_: Optional[CustomerPreprocessor] = None
        self.kmeans_: Optional[KMeans] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.is_fitted_: bool = False

    def _validate_inputs(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame and cluster count parameter."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Input data must be a pandas DataFrame, got {type(df).__name__}.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        n_samples = len(df)
        if not isinstance(self.n_clusters, int) or isinstance(self.n_clusters, bool):
            raise ValueError(
                f"n_clusters must be an integer, got {type(self.n_clusters).__name__}."
            )
        if self.n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {self.n_clusters}.")
        if self.n_clusters >= n_samples:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) must be strictly less than the number of "
                f"available samples ({n_samples})."
            )

    def fit(self, df: pd.DataFrame, y=None) -> "CustomerClustering":
        """Fit preprocessing pipeline and K-Means clustering model on input data.

        Args:
            df: Customer DataFrame.
            y: Ignored (unsupervised clustering).

        Returns:
            self: Fitted CustomerClustering instance.
        """
        self._validate_inputs(df)

        # 1. Fit preprocessor strictly on the provided training partition
        self.preprocessor_ = CustomerPreprocessor(feature_set=self.feature_set)
        X = self.preprocessor_.fit_transform(df)

        # 2. Fit K-Means estimator with explicit, reproducible hyperparameters
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter,
            algorithm=self.algorithm,
        )
        self.kmeans_.fit(X)

        self.labels_ = self.kmeans_.labels_
        self.inertia_ = float(self.kmeans_.inertia_)
        self.is_fitted_ = True

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict cluster index for new or existing customer records.

        Args:
            df: Customer DataFrame to segment.

        Returns:
            1D numpy array of assigned cluster labels.

        Raises:
            NotFittedError: If model is not yet fitted.
        """
        if not self.is_fitted_ or self.preprocessor_ is None or self.kmeans_ is None:
            raise NotFittedError("CustomerClustering is not fitted. Call 'fit' before 'predict'.")

        X = self.preprocessor_.transform(df)
        return self.kmeans_.predict(X)

    def fit_predict(self, df: pd.DataFrame, y=None) -> np.ndarray:
        """Fit model and return assigned cluster labels for training samples."""
        self.fit(df, y=y)
        assert self.labels_ is not None
        return self.labels_

    def assign_clusters(
        self,
        df: pd.DataFrame,
        cluster_col: str = "Cluster",
        fit: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Assign cluster labels to a DataFrame while preserving IDs and row order.

        Args:
            df: Input customer DataFrame.
            cluster_col: Name for the added cluster label column (default: 'Cluster').
            fit: If True, explicitly fit model on df. If None, auto-fits if model is
                 not yet fitted. If False, requires model to already be fitted.

        Returns:
            Copy of df with cluster column appended.
        """
        if fit is True or (fit is None and not self.is_fitted_):
            labels = self.fit_predict(df)
        else:
            labels = self.predict(df)

        result = df.copy()
        result[cluster_col] = labels
        return result

    def get_cluster_centers(self, scaled: bool = False) -> pd.DataFrame:
        """Return cluster centroids in either standardized or original interpretable units.

        Args:
            scaled: If True, returns coordinates in standardized z-score space.
                    If False, inverts scaling to provide original domain units
                    (e.g. Annual Income in $k, Spending Score in 1-100, Age in years).

        Returns:
            pd.DataFrame containing cluster centers indexed by cluster label.

        Raises:
            NotFittedError: If model has not been fitted.
        """
        if not self.is_fitted_ or self.preprocessor_ is None or self.kmeans_ is None:
            raise NotFittedError(
                "CustomerClustering is not fitted. Call 'fit' before 'get_cluster_centers'."
            )

        centers_scaled = self.kmeans_.cluster_centers_
        cluster_index = [f"Cluster_{i}" for i in range(self.n_clusters)]

        if scaled:
            feature_names = self.preprocessor_.get_feature_names()
            return pd.DataFrame(
                centers_scaled,
                columns=feature_names,
                index=cluster_index,
            )

        # Invert scaling to return original interpretable domain units
        transformer = self.preprocessor_.transformer_
        assert transformer is not None

        if self.feature_set == "classic":
            scaler = transformer.named_transformers_["num"]
            centers_orig = scaler.inverse_transform(centers_scaled)
            columns = ["Annual_Income", "Spending_Score"]
            return pd.DataFrame(
                centers_orig,
                columns=columns,
                index=cluster_index,
            ).round(2)

        elif self.feature_set == "extended":
            scaler = transformer.named_transformers_["num"]
            num_centers = scaler.inverse_transform(centers_scaled[:, :3])
            # The 4th column is OneHotEncoded Gender_Male proportion in [0, 1]
            gender_ratio = centers_scaled[:, 3:]
            centers_combined = np.hstack([num_centers, gender_ratio])
            columns = ["Age", "Annual_Income", "Spending_Score", "Gender_Male_Ratio"]
            return pd.DataFrame(
                centers_combined,
                columns=columns,
                index=cluster_index,
            ).round(2)

        raise ValueError(f"Unrecognized feature_set '{self.feature_set}'.")

    def save(self, filepath: Union[str, Path]) -> Path:
        """Persist fitted clustering pipeline and preprocessor to disk.

        Args:
            filepath: Destination file path (.joblib).

        Returns:
            Resolved Path object of saved artifact.

        Raises:
            NotFittedError: If attempting to save an unfitted model.
        """
        if not self.is_fitted_:
            raise NotFittedError("Cannot save an unfitted CustomerClustering instance.")

        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target_path)
        return target_path.resolve()

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CustomerClustering":
        """Load a persisted CustomerClustering instance from disk.

        Args:
            filepath: Path to saved .joblib file.

        Returns:
            Reconstructed, fitted CustomerClustering instance.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If loaded object is not a CustomerClustering instance.
        """
        p = Path(filepath)
        if not p.is_file():
            raise FileNotFoundError(f"Clustering model file not found at '{filepath}'.")

        instance = joblib.load(p)
        if not isinstance(instance, cls):
            # Explicit, safely validated compatibility strategy for legacy artifacts:
            # Only accept CustomerClustering from authorized project module namespaces
            valid_modules = {"src.clustering", "clustering"}
            if (
                type(instance).__name__ == cls.__name__
                and type(instance).__module__ in valid_modules
            ):
                pass
            else:
                raise ValueError(
                    f"Loaded object is not a {cls.__name__} instance "
                    f"(got {type(instance).__module__}.{type(instance).__name__})."
                )
        if not instance.is_fitted_:
            raise ValueError("Loaded CustomerClustering instance was not fitted.")

        return instance


def fit_kmeans(
    df: pd.DataFrame,
    n_clusters: int = 5,
    feature_set: str = "classic",
    random_state: int = 42,
) -> CustomerClustering:
    """Convenience functional helper to train a CustomerClustering model.

    Args:
        df: Input customer DataFrame.
        n_clusters: Number of clusters (k).
        feature_set: 'classic' or 'extended'.
        random_state: Deterministic random seed.

    Returns:
        Fitted CustomerClustering instance.
    """
    model = CustomerClustering(
        n_clusters=n_clusters,
        feature_set=feature_set,
        random_state=random_state,
    )
    return model.fit(df)
