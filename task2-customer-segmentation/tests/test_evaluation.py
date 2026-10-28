"""Unit tests for src.evaluation module."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_data
from src.evaluation import (
    assess_cluster_stability,
    compute_clustering_metrics,
    generate_cluster_profiles,
    plot_cluster_scatter_2d,
    plot_elbow_and_silhouette,
)


@pytest.fixture
def real_data() -> pd.DataFrame:
    """Fixture providing the validated Mall Customer dataset."""
    return load_data()


def test_compute_clustering_metrics_schema(real_data):
    """Verify compute_clustering_metrics returns expected schema and non-empty rows."""
    metrics_df = compute_clustering_metrics(
        real_data,
        feature_set="classic",
        k_range=[2, 3, 4, 5],
        include_k1_inertia=True,
    )
    expected_cols = [
        "k",
        "feature_set",
        "inertia",
        "silhouette_score",
        "davies_bouldin_index",
        "calinski_harabasz_score",
        "n_samples",
    ]
    assert list(metrics_df.columns) == expected_cols
    assert len(metrics_df) == 5  # k=1 plus k=2,3,4,5
    assert (metrics_df["n_samples"] == 200).all()


def test_compute_clustering_metrics_k1_inertia(real_data):
    """Verify k=1 provides baseline inertia and NaN for silhouette and Davies-Bouldin."""
    metrics_df = compute_clustering_metrics(
        real_data,
        feature_set="classic",
        k_range=[2, 3],
        include_k1_inertia=True,
    )
    row_k1 = metrics_df[metrics_df["k"] == 1].iloc[0]
    assert row_k1["inertia"] > 0
    assert np.isnan(row_k1["silhouette_score"])
    assert np.isnan(row_k1["davies_bouldin_index"])


def test_compute_clustering_metrics_extended(real_data):
    """Verify metrics computation works for the extended feature set."""
    metrics_ext = compute_clustering_metrics(
        real_data,
        feature_set="extended",
        k_range=[3, 4],
        include_k1_inertia=False,
    )
    assert len(metrics_ext) == 2
    assert (metrics_ext["feature_set"] == "extended").all()
    assert (metrics_ext["silhouette_score"] > 0).all()


def test_assess_cluster_stability(real_data):
    """Verify cluster stability across multiple random seeds produces high ARI."""
    stability = assess_cluster_stability(
        real_data,
        feature_set="classic",
        k=5,
        seeds=[42, 100, 2024],
    )
    assert stability["feature_set"] == "classic"
    assert stability["k"] == 5
    assert stability["mean_pairwise_ari"] >= 0.95
    assert stability["inertia_std"] < 1.0


def test_generate_cluster_profiles(real_data):
    """Verify cluster profiles correctly calculate cluster counts, proportions, and statistics."""
    from src.clustering import fit_kmeans
    model = fit_kmeans(real_data, n_clusters=5, feature_set="classic", random_state=42)
    segmented = model.assign_clusters(real_data)

    profiles = generate_cluster_profiles(segmented)
    assert len(profiles) == 5
    assert profiles["Size"].sum() == 200
    assert round(profiles["Percentage"].sum(), 0) == 100.0

    required_cols = [
        "Cluster",
        "Descriptive_Name",
        "Size",
        "Percentage",
        "Mean_Age",
        "Median_Age",
        "Mean_Income_k$",
        "Median_Income_k$",
        "Mean_Spending_Score",
        "Median_Spending_Score",
        "Female_Count",
        "Male_Count",
        "Female_Pct",
        "Male_Pct",
        "Retail_Interpretation",
    ]
    for col in required_cols:
        assert col in profiles.columns

    # Verify no NaN values in numerical stats
    assert profiles[["Mean_Income_k$", "Mean_Spending_Score", "Mean_Age"]].isnull().sum().sum() == 0


def test_plot_elbow_and_silhouette(tmp_path: Path, real_data):
    """Verify plot_elbow_and_silhouette creates valid non-empty PNG files."""
    metrics_df = compute_clustering_metrics(
        real_data,
        feature_set="classic",
        k_range=range(2, 6),
        include_k1_inertia=True,
    )
    elbow_file, sil_file = plot_elbow_and_silhouette(metrics_df, "classic", tmp_path)

    assert elbow_file.is_file()
    assert sil_file.is_file()
    assert elbow_file.stat().st_size > 10000
    assert sil_file.stat().st_size > 10000


def test_plot_cluster_scatter_2d(tmp_path: Path, real_data):
    """Verify plot_cluster_scatter_2d creates a valid non-empty scatter plot."""
    from src.clustering import fit_kmeans
    model = fit_kmeans(real_data, n_clusters=5, feature_set="classic", random_state=42)
    segmented = model.assign_clusters(real_data)
    centers = model.get_cluster_centers(scaled=False)

    scatter_file = plot_cluster_scatter_2d(segmented, centers, "classic", 5, tmp_path)
    assert scatter_file.is_file()
    assert scatter_file.stat().st_size > 20000
