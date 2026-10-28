"""Helper functions for Task 2 Streamlit dashboard.

Provides testable, modular utility functions for:
- Data filtering across cluster, income, spending, and gender selections.
- Construction of formatted new-customer prediction DataFrames.
- Cluster profile summarization for UI tables.
- Model metadata and feature definitions.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


CLUSTER_DESCRIPTIONS: Dict[int, Dict[str, str]] = {
    0: {
        "name": "Moderate Income / Moderate Spending",
        "persona": "Middle-Market Majority",
        "description": "Core baseline customer segment with average income and balanced center spending.",
        "strategy": "Broad promotional campaigns, department staples, and family-oriented loyalty programs.",
    },
    1: {
        "name": "High Income / High Spending",
        "persona": "Affluent Trendsetters",
        "description": "High-earning customers actively investing in premium lifestyle and retail offerings.",
        "strategy": "Exclusive product previews, VIP perks, premium concierge services, and luxury events.",
    },
    2: {
        "name": "Low Income / High Spending",
        "persona": "Young Impulsive Shoppers",
        "description": "Budget-constrained younger shoppers highly engaged with lifestyle, dining, and trends.",
        "strategy": "Fast fashion, dynamic social marketing, trend-driven discounts, and event-based activations.",
    },
    3: {
        "name": "High Income / Low Spending",
        "persona": "Affluent Conservative Savers",
        "description": "High purchasing power with low center engagement; cautious and value-driven spenders.",
        "strategy": "Specialty electronics, luxury wellness, curated value propositions, and targeted incentives.",
    },
    4: {
        "name": "Low Income / Low Spending",
        "persona": "Frugal / Budget-Constrained",
        "description": "Price-sensitive customers purchasing essential goods with minimal discretionary spend.",
        "strategy": "Clear discounts, essential household promotions, value bundles, and clearance sales.",
    },
}


def filter_customer_data(
    df: pd.DataFrame,
    selected_clusters: Optional[List[int]] = None,
    income_range: Optional[Tuple[float, float]] = None,
    spending_range: Optional[Tuple[float, float]] = None,
    selected_gender: str = "All",
    cluster_col: str = "Cluster_Classic_k5",
) -> pd.DataFrame:
    """Filter customer records based on user-selected criteria without modifying fitted models.

    Args:
        df: Input segmented customer DataFrame.
        selected_clusters: List of integer cluster labels to include.
        income_range: Tuple of (min_income, max_income) in thousands.
        spending_range: Tuple of (min_spending, max_spending).
        selected_gender: 'All', 'Female', or 'Male'.
        cluster_col: Name of active cluster label column.

    Returns:
        Filtered copy of the customer DataFrame.
    """
    filtered = df.copy()

    if selected_clusters is not None and len(selected_clusters) > 0:
        filtered = filtered[filtered[cluster_col].isin(selected_clusters)]

    if income_range is not None:
        min_inc, max_inc = income_range
        filtered = filtered[(filtered["Annual_Income"] >= min_inc) & (filtered["Annual_Income"] <= max_inc)]

    if spending_range is not None:
        min_ss, max_ss = spending_range
        filtered = filtered[(filtered["Spending_Score"] >= min_ss) & (filtered["Spending_Score"] <= max_ss)]

    if selected_gender != "All":
        filtered = filtered[filtered["Gender"] == selected_gender]

    return filtered


def create_new_customer_record(
    age: int,
    annual_income: float,
    spending_score: float,
    gender: str,
) -> pd.DataFrame:
    """Construct a single-row DataFrame formatted for model prediction.

    Args:
        age: Customer age (18-100).
        annual_income: Customer annual income in thousands (1-300).
        spending_score: Customer spending score (1-100).
        gender: 'Male' or 'Female'.

    Returns:
        1-row pd.DataFrame matching load_data() normalized schema.

    Raises:
        ValueError: If input values fall outside realistic physical domains.
    """
    if not (1 <= age <= 120):
        raise ValueError(f"Age must be between 1 and 120, got {age}.")
    if annual_income < 0:
        raise ValueError(f"Annual Income cannot be negative, got {annual_income}.")
    if not (1 <= spending_score <= 100):
        raise ValueError(f"Spending Score must be between 1 and 100, got {spending_score}.")
    if gender not in {"Male", "Female"}:
        raise ValueError(f"Gender must be 'Male' or 'Female', got '{gender}'.")

    return pd.DataFrame({
        "CustomerID": [999],  # Synthetic identifier, excluded by preprocessor
        "Gender": [gender],
        "Age": [int(age)],
        "Annual_Income": [float(annual_income)],
        "Spending_Score": [float(spending_score)],
    })


def compute_cluster_summary_table(
    df: pd.DataFrame,
    cluster_col: str = "Cluster_Classic_k5",
) -> pd.DataFrame:
    """Compute cluster-level demographic and spending metrics for UI display.

    Args:
        df: Customer DataFrame containing cluster_col.
        cluster_col: Name of cluster column.

    Returns:
        pd.DataFrame formatted for Streamlit table display.
    """
    total = len(df)
    rows = []

    for c in sorted(df[cluster_col].unique()):
        group = df[df[cluster_col] == c]
        count = len(group)
        pct = round(count / total * 100, 1)

        info = CLUSTER_DESCRIPTIONS.get(c, {
            "name": f"Cluster {c}",
            "persona": f"Segment {c}",
            "description": "Empirical cluster group.",
            "strategy": "Targeted marketing.",
        })

        rows.append({
            "Cluster": c,
            "Persona": info["persona"],
            "Descriptive Name": info["name"],
            "Customers (n)": count,
            "Share (%)": pct,
            "Mean Income ($k)": round(group["Annual_Income"].mean(), 1),
            "Mean Spending Score": round(group["Spending_Score"].mean(), 1),
            "Median Age": round(group["Age"].median(), 0),
            "Female (%)": round((group["Gender"] == "Female").mean() * 100, 1),
            "Male (%)": round((group["Gender"] == "Male").mean() * 100, 1),
        })

    return pd.DataFrame(rows)


def get_model_metadata(model_type: str = "classic") -> Dict[str, Any]:
    """Retrieve metadata and feature definitions for the selected clustering model."""
    if model_type.lower() == "classic":
        return {
            "name": "Classic Segmentation (2D Benchmark)",
            "features": ["Annual_Income", "Spending_Score"],
            "k": 5,
            "silhouette": 0.5547,
            "davies_bouldin": 0.5722,
            "inertia": 65.57,
            "stability_ari": 1.0,
            "cluster_col": "Cluster_Classic_k5",
            "model_file": "kmeans_classic_k5.joblib",
            "summary": "Segments customers strictly on Annual Income and Spending Score. Serves as the primary behavioral benchmark.",
        }
    elif model_type.lower() == "extended":
        return {
            "name": "Extended Demographic Segmentation (4D)",
            "features": ["Age", "Annual_Income", "Spending_Score", "Gender_Male"],
            "k": 5,
            "silhouette": 0.3498,
            "davies_bouldin": 1.0245,
            "inertia": 216.78,
            "stability_ari": 1.0,
            "cluster_col": "Cluster_Extended_k5",
            "model_file": "kmeans_extended_k5.joblib",
            "summary": "Incorporates Age and binary Gender encoding alongside Income and Spending Score for richer persona profiling.",
        }
    raise ValueError(f"Unknown model_type '{model_type}'. Expected 'classic' or 'extended'.")
