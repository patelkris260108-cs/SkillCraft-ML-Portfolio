"""
3D Visualization utilities for EstateIQ using PyDeck.

This module provides functions to create interactive 3D charts for the
Ames Housing dataset, including 3D scatter plots and other analytical views.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import pydeck as pdk
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DataLoader, BASELINE_FEATURES, TARGET_COLUMN


# Color scales for consistent theming (light theme)
COLOR_SCALES = {
    "sale_price": [
        [0, "#E8DECF"],      # Warm light (min)
        [0.25, "#0D9488"],   # Teal/dark green (accent)
        [0.5, "#14B8A6"],    # Emerald (mid)
        [0.75, "#57C7E8"],   # Cyan (secondary accent)
        [1, "#1A1A1A"],      # Dark charcoal (max)
    ],
    "quality": [
        [1, "#F1ECE1"],      # Very light warm (quality 1)
        [3, "#0D9488"],      # Teal (quality 3-4)
        [5, "#14B8A6"],      # Emerald (quality 5-6)
        [7, "#57C7E8"],      # Cyan (quality 7-8)
        [10, "#1A1A1A"],     # Dark charcoal (quality 10)
    ],
}


def load_training_data(sample_size: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
    """
    Load training data with optional sampling for performance.

    Args:
        sample_size: If set, randomly sample this many rows
        random_state: Random seed for reproducible sampling

    Returns:
        DataFrame with training data
    """
    loader = DataLoader("data/raw", strict_validation=False)
    df = loader.load_train(validate=False)

    # Select relevant columns for 3D visualization
    cols = ["GrLivArea", "OverallQual", "SalePrice", "Neighborhood"]
    df = df[cols].copy()

    # Drop rows with missing values in key columns
    df = df.dropna(subset=cols)

    # Ensure numeric types
    df["GrLivArea"] = pd.to_numeric(df["GrLivArea"], errors="coerce")
    df["OverallQual"] = pd.to_numeric(df["OverallQual"], errors="coerce")
    df["SalePrice"] = pd.to_numeric(df["SalePrice"], errors="coerce")
    df = df.dropna(subset=cols)

    # Sample if requested
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)

    return df


def load_validation_predictions() -> Optional[pd.DataFrame]:
    """
    Load or regenerate validation set predictions for actual vs predicted views.

    Returns:
        DataFrame with actual and predicted values, or None if not available
    """
    try:
        # Try to load split indices
        import json
        split_path = Path(__file__).parent.parent / "reports" / "split_indices.json"
        if not split_path.exists():
            return None

        with open(split_path, "r") as f:
            split_info = json.load(f)

        val_indices = split_info["val_indices"]

        # Load training data and model
        loader = DataLoader("data/raw", strict_validation=False)
        train_df = loader.load_train(validate=False)

        # Get validation set
        val_df = train_df.iloc[val_indices].copy()

        # Load model and make predictions
        from predict import HousePricePredictor
        model_path = Path(__file__).parent.parent / "models" / "baseline_pipeline.joblib"

        if not model_path.exists():
            return None

        predictor = HousePricePredictor(str(model_path))

        # Prepare features
        X_val = val_df[BASELINE_FEATURES].copy()
        y_val = val_df[TARGET_COLUMN].copy()

        # Make predictions
        predictions = predictor.predict_batch(X_val)

        # Create result DataFrame
        result_df = val_df[["GrLivArea", "OverallQual", "SalePrice", "Neighborhood"]].copy()
        result_df["Predicted_SalePrice"] = predictions
        result_df["Residual"] = result_df["SalePrice"] - result_df["Predicted_SalePrice"]
        result_df["Set"] = "Validation"

        return result_df

    except Exception:
        return None


def create_3d_scatter_chart(
    data: pd.DataFrame,
    x_col: str = "GrLivArea",
    y_col: str = "OverallQual",
    z_col: str = "SalePrice",
    color_col: Optional[str] = None,
    tooltip_cols: Optional[List[str]] = None,
    title: str = "3D Scatter: Living Area × Quality × Sale Price",
    sample_size: Optional[int] = 1000,
    random_state: int = 42,
    point_radius_min: int = 3,
    point_radius_max: int = 8,
) -> pdk.Deck:
    """
    Create an interactive 3D scatter plot using PyDeck.

    Args:
        data: DataFrame with the data
        x_col: Column for X axis (living area)
        y_col: Column for Y axis (quality)
        z_col: Column for Z axis (sale price)
        color_col: Column for color encoding (defaults to z_col)
        tooltip_cols: Columns to show in tooltip
        title: Chart title
        sample_size: Max points to render (for performance)
        random_state: Random seed for sampling
        point_radius_min: Minimum point radius in pixels
        point_radius_max: Maximum point radius in pixels

    Returns:
        PyDeck Deck object
    """
    # Prepare data
    plot_data = data.copy()

    # Sample for performance if needed
    if sample_size and len(plot_data) > sample_size:
        plot_data = plot_data.sample(n=sample_size, random_state=random_state)

    # Normalize coordinates for PyDeck (which expects [lng, lat] for position)
    # We'll use a custom coordinate system
    x_vals = plot_data[x_col].values
    y_vals = plot_data[y_col].values
    z_vals = plot_data[z_col].values

    # Normalize to 0-1 range for better PyDeck rendering
    x_norm = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min())
    y_norm = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min())
    z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min())

    # Scale for better visual separation (must stay within valid geographic bounds:
    # longitude -180..180, latitude -85..85 for deck.gl with map_style=None)
    scale_factor = 50
    plot_data["x_pos"] = x_norm * scale_factor
    plot_data["y_pos"] = y_norm * scale_factor
    plot_data["z_pos"] = z_norm * scale_factor

    # Determine color column
    if color_col is None:
        color_col = z_col

    color_vals = plot_data[color_col].values
    color_norm = (color_vals - color_vals.min()) / (color_vals.max() - color_vals.min())

    # Create color mapping (teal -> cyan -> dark charcoal)
    def get_color(val):
        if val < 0.25:
            return [13, 148, 136, 200]    # Teal/dark green
        elif val < 0.5:
            return [20, 184, 166, 200]    # Emerald
        elif val < 0.75:
            return [87, 199, 232, 200]    # Cyan
        else:
            return [26, 26, 26, 200]      # Dark charcoal

    plot_data["color"] = [get_color(v) for v in color_norm]

    # Point radius based on z value (larger = higher price)
    plot_data["radius"] = point_radius_min + (color_norm * (point_radius_max - point_radius_min))

    # Build tooltip data
    if tooltip_cols is None:
        tooltip_cols = ["GrLivArea", "OverallQual", "SalePrice", "Neighborhood"]

    # Ensure tooltip cols exist
    tooltip_cols = [c for c in tooltip_cols if c in plot_data.columns]

    # Create ScatterplotLayer
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=plot_data,
        get_position=["x_pos", "y_pos", "z_pos"],
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        auto_highlight=True,
        tooltip=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        radius_scale=1,
        radius_min_pixels=point_radius_min,
        radius_max_pixels=point_radius_max,
    )

    # Define view state for 3D
    view_state = pdk.ViewState(
        latitude=25,
        longitude=25,
        zoom=1,
        pitch=45,
        bearing=0,
        height=600,
    )

    # Tooltip
    tooltip_html = "<b>{Neighborhood}</b><br/>"
    tooltip_html += "Living Area: {GrLivArea} sq ft<br/>"
    tooltip_html += "Quality: {OverallQual}/10<br/>"
    tooltip_html += "Sale Price: ${SalePrice}"

    # Create deck
    deck = pdk.Deck(
        layers=[scatter_layer],
        initial_view_state=view_state,
        tooltip={"html": tooltip_html, "style": {"backgroundColor": "#F1ECE1", "color": "#1A1A1A", "border": "1px solid #E5E2DD"}},
        map_style=None,  # No base map
    )

    return deck


def create_sale_price_distribution_chart(data: pd.DataFrame):
    """
    Create a sale price distribution chart using Altair.

    Returns:
        Altair chart object
    """
    import altair as alt

    chart = alt.Chart(data).transform_density(
        "SalePrice",
        as_=["SalePrice", "density"],
        extent=[data["SalePrice"].min(), data["SalePrice"].max()],
    ).mark_area(
        opacity=0.6,
        color="#13B8A6",
        line={"color": "#57C7E8"}
    ).encode(
        x=alt.X("SalePrice:Q", title="Sale Price ($)", axis=alt.Axis(format="$,.0f")),
        y=alt.Y("density:Q", title="Density"),
        tooltip=[
            alt.Tooltip("SalePrice:Q", title="Sale Price", format="$,.0f"),
            alt.Tooltip("density:Q", title="Density", format=".4f"),
        ],
    ).properties(
        width=600,
        height=300,
        title="Sale Price Distribution"
    ).configure(
        background="#FAF7F3",
        axis={"labelColor": "#1A1A1A", "titleColor": "#1A1A1A", "gridColor": "#E5E2DD"},
        title={"color": "#1A1A1A"},
        view={"stroke": "transparent"},
    )

    return chart


def create_living_area_vs_price_chart(data: pd.DataFrame):
    """
    Create a Living Area vs Sale Price scatter chart using Altair.

    Returns:
        Altair chart object
    """
    import altair as alt

    chart = alt.Chart(data).mark_circle(
        opacity=0.6,
        size=60,
        color="#13B8A6"
    ).encode(
        x=alt.X("GrLivArea:Q", title="Living Area (sq ft)", axis=alt.Axis(format=",.0f")),
        y=alt.Y("SalePrice:Q", title="Sale Price ($)", axis=alt.Axis(format="$,.0f")),
        color=alt.Color("OverallQual:O", title="Overall Quality",
                       scale=alt.Scale(scheme="viridis")),
        tooltip=[
            alt.Tooltip("GrLivArea:Q", title="Living Area", format=",.0f"),
            alt.Tooltip("SalePrice:Q", title="Sale Price", format="$,.0f"),
            alt.Tooltip("OverallQual:O", title="Quality"),
            alt.Tooltip("Neighborhood:N", title="Neighborhood"),
        ],
    ).properties(
        width=600,
        height=400,
        title="Living Area vs Sale Price"
    ).configure(
        background="#FAF7F3",
        axis={"labelColor": "#1A1A1A", "titleColor": "#1A1A1A", "gridColor": "#E5E2DD"},
        title={"color": "#1A1A1A"},
        legend={"labelColor": "#1A1A1A", "titleColor": "#1A1A1A"},
        view={"stroke": "transparent"},
    )

    return chart


def create_quality_comparison_chart(data: pd.DataFrame):
    """
    Create a box plot comparing Sale Price by Overall Quality using Altair.

    Returns:
        Altair chart object
    """
    import altair as alt

    chart = alt.Chart(data).mark_boxplot(
        extent="min-max",
        color="#13B8A6",
        size=40,
    ).encode(
        x=alt.X("OverallQual:O", title="Overall Quality (1-10)", sort=list(range(1, 11))),
        y=alt.Y("SalePrice:Q", title="Sale Price ($)", axis=alt.Axis(format="$,.0f")),
        tooltip=[
            alt.Tooltip("OverallQual:O", title="Quality"),
            alt.Tooltip("SalePrice:Q", title="Median Price", format="$,.0f"),
        ],
    ).properties(
        width=600,
        height=400,
        title="Sale Price Distribution by Overall Quality"
    ).configure(
        background="#FAF7F3",
        axis={"labelColor": "#1A1A1A", "titleColor": "#1A1A1A", "gridColor": "#E5E2DD"},
        title={"color": "#1A1A1A"},
        view={"stroke": "transparent"},
    )

    return chart


def create_neighborhood_comparison_chart(data: pd.DataFrame, top_n: int = 15):
    """
    Create a bar chart comparing median sale price by neighborhood using Altair.

    Returns:
        Altair chart object
    """
    import altair as alt

    # Calculate median price by neighborhood
    neighborhood_stats = data.groupby("Neighborhood").agg(
        median_price=("SalePrice", "median"),
        count=("SalePrice", "count"),
    ).reset_index().sort_values("median_price", ascending=False).head(top_n)

    chart = alt.Chart(neighborhood_stats).mark_bar(
        color="#13B8A6",
        cornerRadiusTopLeft=4,
        cornerRadiusTopRight=4,
    ).encode(
        x=alt.X("median_price:Q", title="Median Sale Price ($)", axis=alt.Axis(format="$,.0f")),
        y=alt.Y("Neighborhood:N", title="Neighborhood", sort="-x"),
        tooltip=[
            alt.Tooltip("Neighborhood:N", title="Neighborhood"),
            alt.Tooltip("median_price:Q", title="Median Price", format="$,.0f"),
            alt.Tooltip("count:Q", title="Sample Count"),
        ],
    ).properties(
        width=600,
        height=500,
        title=f"Top {top_n} Neighborhoods by Median Sale Price"
    ).configure(
        background="#FAF7F3",
        axis={"labelColor": "#1A1A1A", "titleColor": "#1A1A1A", "gridColor": "#E5E2DD"},
        title={"color": "#1A1A1A"},
        view={"stroke": "transparent"},
    )

    return chart


def create_actual_vs_predicted_chart(pred_data: pd.DataFrame):
    """
    Create an actual vs predicted scatter chart using Altair.

    Returns:
        Altair chart object
    """
    import altair as alt

    # Filter out rows with NaN predictions
    plot_data = pred_data.dropna(subset=["Predicted_SalePrice"]).copy()

    if len(plot_data) == 0:
        return None

    # Perfect prediction line data
    min_val = min(plot_data["SalePrice"].min(), plot_data["Predicted_SalePrice"].min())
    max_val = max(plot_data["SalePrice"].max(), plot_data["Predicted_SalePrice"].max())
    line_data = pd.DataFrame({"x": [min_val, max_val], "y": [min_val, max_val]})

    points = alt.Chart(plot_data).mark_circle(
        opacity=0.6,
        size=80,
        color="#13B8A6"
    ).encode(
        x=alt.X("SalePrice:Q", title="Actual Sale Price ($)", axis=alt.Axis(format="$,.0f")),
        y=alt.Y("Predicted_SalePrice:Q", title="Predicted Sale Price ($)", axis=alt.Axis(format="$,.0f")),
        color=alt.Color("Residual:Q", title="Residual ($)",
                       scale=alt.Scale(scheme="redblue", domainMid=0)),
        tooltip=[
            alt.Tooltip("SalePrice:Q", title="Actual", format="$,.0f"),
            alt.Tooltip("Predicted_SalePrice:Q", title="Predicted", format="$,.0f"),
            alt.Tooltip("Residual:Q", title="Residual", format="$,.0f"),
            alt.Tooltip("Neighborhood:N", title="Neighborhood"),
        ],
    )

    line = alt.Chart(line_data).mark_line(
        color="#EF4444",
        strokeDash=[5, 5],
        strokeWidth=2,
    ).encode(
        x="x:Q",
        y="y:Q",
    )

    chart = (points + line).properties(
        width=500,
        height=500,
        title="Actual vs Predicted Sale Price (Validation Set)"
    ).configure(
        background="#FAF7F3",
        axis={"labelColor": "#1A1A1A", "titleColor": "#1A1A1A", "gridColor": "#E5E2DD"},
        title={"color": "#1A1A1A"},
        legend={"labelColor": "#1A1A1A", "titleColor": "#1A1A1A"},
        view={"stroke": "transparent"},
    )

    return chart


def create_residual_chart(pred_data: pd.DataFrame):
    """
    Create a residual plot using Altair.

    Returns:
        Altair chart object
    """
    import altair as alt

    plot_data = pred_data.dropna(subset=["Predicted_SalePrice"]).copy()

    if len(plot_data) == 0:
        return None

    chart = alt.Chart(plot_data).mark_circle(
        opacity=0.6,
        size=60,
        color="#57C7E8"
    ).encode(
        x=alt.X("Predicted_SalePrice:Q", title="Predicted Sale Price ($)", axis=alt.Axis(format="$,.0f")),
        y=alt.Y("Residual:Q", title="Residual ($)", axis=alt.Axis(format="$,.0f")),
        color=alt.Color("Residual:Q", title="Residual",
                       scale=alt.Scale(scheme="redblue", domainMid=0)),
        tooltip=[
            alt.Tooltip("Predicted_SalePrice:Q", title="Predicted", format="$,.0f"),
            alt.Tooltip("Residual:Q", title="Residual", format="$,.0f"),
            alt.Tooltip("Neighborhood:N", title="Neighborhood"),
        ],
    )

    # Add horizontal line at y=0
    hline = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color="#EF4444",
        strokeDash=[5, 5],
        strokeWidth=2,
    ).encode(y="y:Q")

    return (chart + hline).properties(
        width=600,
        height=400,
        title="Residuals vs Predicted Price"
    ).configure(
        background="#FAF7F3",
        axis={"labelColor": "#1A1A1A", "titleColor": "#1A1A1A", "gridColor": "#E5E2DD"},
        title={"color": "#1A1A1A"},
        legend={"labelColor": "#1A1A1A", "titleColor": "#1A1A1A"},
        view={"stroke": "transparent"},
    )


if __name__ == "__main__":
    # Quick test
    print("Testing 3D visualization module...")

    # Test loading data
    df = load_training_data(sample_size=100)
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    print(f"GrLivArea range: {df['GrLivArea'].min()} - {df['GrLivArea'].max()}")
    print(f"OverallQual range: {df['OverallQual'].min()} - {df['OverallQual'].max()}")
    print(f"SalePrice range: ${df['SalePrice'].min():,.0f} - ${df['SalePrice'].max():,.0f}")

    # Test 3D chart creation
    deck = create_3d_scatter_chart(df, sample_size=50)
    print("3D chart created successfully")

    print("\nAll tests passed!")