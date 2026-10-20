"""
Visualization utilities for Task 1: House Price Prediction.

This module provides functions to create diagnostic plots for
regression model evaluation, including residual plots, actual vs
predicted scatter plots, and target distribution histograms.
"""

from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Default figure settings
DEFAULT_FIGSIZE = (10, 6)
DEFAULT_DPI = 150
PLOT_DIR = Path("reports/figures")


def ensure_plot_dir() -> Path:
    """Ensure plot directory exists and return Path."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    return PLOT_DIR


def save_figure(fig: plt.Figure, filename: str, dpi: int = DEFAULT_DPI) -> Path:
    """
    Save figure to reports/figures directory.

    Args:
        fig: Matplotlib figure
        filename: Output filename (e.g., "residuals.png")
        dpi: Resolution

    Returns:
        Path to saved file
    """
    ensure_plot_dir()
    filepath = PLOT_DIR / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved plot: {filepath}")
    return filepath


def plot_target_distribution(
    y: np.ndarray,
    title: str = "SalePrice Distribution",
    filename: Optional[str] = "price_distribution.png",
    bins: int = 50,
    log_scale: bool = False
) -> plt.Figure:
    """
    Plot histogram of target variable with statistics.

    Args:
        y: Target values
        title: Plot title
        filename: Output filename (None to skip saving)
        bins: Number of histogram bins
        log_scale: If True, also show log-transformed distribution

    Returns:
        Matplotlib figure
    """
    y = np.asarray(y).ravel()

    if log_scale:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=DEFAULT_FIGSIZE)
        axes = [axes]

    # Original scale
    ax = axes[0]
    ax.hist(y, bins=bins, edgecolor="white", alpha=0.7, color="#2E86AB", density=True)

    # Add KDE
    from scipy import stats
    kde = stats.gaussian_kde(y)
    x_range = np.linspace(y.min(), y.max(), 200)
    ax.plot(x_range, kde(x_range), color="#A23B72", linewidth=2, label="KDE")

    # Statistics lines
    ax.axvline(y.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: ${y.mean():,.0f}")
    ax.axvline(np.median(y), color="green", linestyle="--", linewidth=1.5, label=f"Median: ${np.median(y):,.0f}")

    ax.set_xlabel("Sale Price ($)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log scale subplot
    if log_scale:
        ax2 = axes[1]
        y_log = np.log1p(y)
        ax2.hist(y_log, bins=bins, edgecolor="white", alpha=0.7, color="#F18F01", density=True)

        kde_log = stats.gaussian_kde(y_log)
        x_range_log = np.linspace(y_log.min(), y_log.max(), 200)
        ax2.plot(x_range_log, kde_log(x_range_log), color="#A23B72", linewidth=2, label="KDE")

        ax2.axvline(y_log.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {y_log.mean():.2f}")
        ax2.axvline(np.median(y_log), color="green", linestyle="--", linewidth=1.5, label=f"Median: {np.median(y_log):.2f}")

        ax2.set_xlabel("Log(Sale Price + 1)")
        ax2.set_ylabel("Density")
        ax2.set_title(f"{title} (Log Scale)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename:
        save_figure(fig, filename)

    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted",
    filename: Optional[str] = "actual_vs_predicted.png",
    alpha: float = 0.6,
    sample_size: Optional[int] = None
) -> plt.Figure:
    """
    Scatter plot of actual vs predicted values.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        title: Plot title
        filename: Output filename (None to skip saving)
        alpha: Point transparency
        sample_size: If set, randomly sample this many points for clarity

    Returns:
        Matplotlib figure
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Optionally sample for large datasets
    if sample_size and len(y_true) > sample_size:
        idx = np.random.choice(len(y_true), sample_size, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=alpha, s=20, color="#2E86AB", edgecolors="white", linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")

    # Add R² annotation
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f"R² = {r2:.4f}", transform=ax.transAxes,
            fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Actual Sale Price ($)")
    ax.set_ylabel("Predicted Sale Price ($)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Equal aspect for visual accuracy
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    if filename:
        save_figure(fig, filename)

    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Plot",
    filename: Optional[str] = "residuals.png",
    alpha: float = 0.6,
    sample_size: Optional[int] = None
) -> plt.Figure:
    """
    Plot residuals vs predicted values.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        title: Plot title
        filename: Output filename (None to skip saving)
        alpha: Point transparency
        sample_size: If set, randomly sample this many points

    Returns:
        Matplotlib figure
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    residuals = y_true - y_pred

    # Optionally sample
    if sample_size and len(y_true) > sample_size:
        idx = np.random.choice(len(y_true), sample_size, replace=False)
        y_pred = y_pred[idx]
        residuals = residuals[idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    ax1 = axes[0]
    ax1.scatter(y_pred, residuals, alpha=alpha, s=20, color="#2E86AB", edgecolors="white", linewidth=0.5)
    ax1.axhline(y=0, color="red", linestyle="--", linewidth=2)
    ax1.set_xlabel("Predicted Sale Price ($)")
    ax1.set_ylabel("Residual ($)")
    ax1.set_title("Residuals vs Predicted")
    ax1.grid(True, alpha=0.3)

    # Residuals histogram
    ax2 = axes[1]
    ax2.hist(residuals, bins=30, edgecolor="white", alpha=0.7, color="#F18F01", density=True)

    # Add normal curve for comparison
    from scipy import stats
    mu, sigma = residuals.mean(), residuals.std()
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    ax2.plot(x_range, stats.norm.pdf(x_range, mu, sigma), "r--", linewidth=2, label=f"Normal(μ={mu:.0f}, σ={sigma:.0f})")

    ax2.axvline(0, color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("Residual ($)")
    ax2.set_ylabel("Density")
    ax2.set_title("Residual Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if filename:
        save_figure(fig, filename)

    return fig


def plot_residuals_vs_feature(
    X: np.ndarray,
    feature_names: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residuals vs Features",
    filename: Optional[str] = "residuals_vs_features.png"
) -> plt.Figure:
    """
    Plot residuals vs each feature to check for patterns.

    Args:
        X: Feature matrix
        feature_names: List of feature names
        y_true: Ground truth values
        y_pred: Predicted values
        title: Plot title
        filename: Output filename (None to skip saving)

    Returns:
        Matplotlib figure
    """
    residuals = y_true - y_pred
    n_features = X.shape[1]

    # Calculate grid layout
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.ravel() if n_features > 1 else [axes]

    for i, (feat_name, ax) in enumerate(zip(feature_names, axes)):
        ax.scatter(X[:, i], residuals, alpha=0.6, s=15, color="#2E86AB", edgecolors="white", linewidth=0.3)
        ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel(feat_name)
        ax.set_ylabel("Residual ($)")
        ax.set_title(f"Residuals vs {feat_name}")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if filename:
        save_figure(fig, filename)

    return fig


def plot_feature_importance(
    feature_names: List[str],
    coefficients: np.ndarray,
    title: str = "Feature Coefficients (Linear Regression)",
    filename: Optional[str] = "feature_importance.png",
    top_n: Optional[int] = None
) -> plt.Figure:
    """
    Plot feature coefficients for linear models.

    Args:
        feature_names: List of feature names
        coefficients: Model coefficients
        title: Plot title
        filename: Output filename (None to skip saving)
        top_n: If set, show only top N by absolute magnitude

    Returns:
        Matplotlib figure
    """
    # Create DataFrame for sorting
    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients
    })
    df["abs_coef"] = df["coefficient"].abs()
    df = df.sort_values("abs_coef", ascending=True)

    if top_n:
        df = df.tail(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.3)))

    colors = ["#A23B72" if c < 0 else "#2E86AB" for c in df["coefficient"]]
    bars = ax.barh(range(len(df)), df["coefficient"], color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"])
    ax.set_xlabel("Coefficient Value")
    ax.set_title(title)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.text(row["coefficient"] + (0.01 * df["coefficient"].max()),
                i, f"{row['coefficient']:.4f}", va="center", fontsize=9)

    plt.tight_layout()

    if filename:
        save_figure(fig, filename)

    return fig


def plot_model_comparison(
    model_names: List[str],
    metrics: List[Dict[str, float]],
    metric_key: str = "Val_RMSE",
    title: str = "Model Comparison",
    filename: Optional[str] = "model_comparison.png"
) -> plt.Figure:
    """
    Bar chart comparing models on a specific metric.

    Args:
        model_names: List of model names
        metrics: List of metric dictionaries (one per model)
        metric_key: Metric to compare
        title: Plot title
        filename: Output filename (None to skip saving)

    Returns:
        Matplotlib figure
    """
    values = [m.get(metric_key, 0) for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.barh(range(len(model_names)), values, color="#2E86AB", edgecolor="white", height=0.6)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlabel(metric_key)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + (0.01 * max(values)), bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=10)

    plt.tight_layout()

    if filename:
        save_figure(fig, filename)

    return fig


def plot_cross_validation_scores(
    cv_results: Dict,
    title: str = "Cross-Validation Scores",
    filename: Optional[str] = "cv_scores.png"
) -> plt.Figure:
    """
    Plot cross-validation scores with mean and std.

    Args:
        cv_results: Dictionary from cross_validate_model
        title: Plot title
        filename: Output filename (None to skip saving)

    Returns:
        Matplotlib figure
    """
    metrics_to_plot = ["CV_MAE_mean", "CV_RMSE_mean", "CV_R2_mean"]
    metric_labels = ["MAE", "RMSE", "R²"]

    means = [cv_results.get(m, 0) for m in metrics_to_plot]
    stds = [cv_results.get(m.replace("_mean", "_std"), 0) for m in metrics_to_plot]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(metric_labels, means, yerr=stds, capsize=10,
                   color=["#2E86AB", "#F18F01", "#A23B72"],
                   edgecolor="white", alpha=0.8)

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01 * max(means),
                f"{mean:.4f}±{std:.4f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()

    if filename:
        save_figure(fig, filename)

    return fig


def generate_all_diagnostic_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    model_name: str = "Model"
) -> List[Path]:
    """
    Generate all standard diagnostic plots for a regression model.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        X: Feature matrix (optional, for residuals vs features)
        feature_names: Feature names (optional)
        model_name: Model name for titles

    Returns:
        List of saved file paths
    """
    saved_files = []

    # Target distribution
    fig = plot_target_distribution(y_true, title=f"{model_name} - SalePrice Distribution",
                                    filename=f"{model_name.lower().replace(' ', '_')}_price_distribution.png",
                                    log_scale=True)
    saved_files.append(PLOT_DIR / f"{model_name.lower().replace(' ', '_')}_price_distribution.png")

    # Actual vs Predicted
    fig = plot_actual_vs_predicted(y_true, y_pred,
                                    title=f"{model_name} - Actual vs Predicted",
                                    filename=f"{model_name.lower().replace(' ', '_')}_actual_vs_predicted.png")
    saved_files.append(PLOT_DIR / f"{model_name.lower().replace(' ', '_')}_actual_vs_predicted.png")

    # Residuals
    fig = plot_residuals(y_true, y_pred,
                          title=f"{model_name} - Residual Analysis",
                          filename=f"{model_name.lower().replace(' ', '_')}_residuals.png")
    saved_files.append(PLOT_DIR / f"{model_name.lower().replace(' ', '_')}_residuals.png")

    # Residuals vs features (if provided)
    if X is not None and feature_names is not None:
        fig = plot_residuals_vs_feature(X, feature_names, y_true, y_pred,
                                         title=f"{model_name} - Residuals vs Features",
                                         filename=f"{model_name.lower().replace(' ', '_')}_residuals_vs_features.png")
        saved_files.append(PLOT_DIR / f"{model_name.lower().replace(' ', '_')}_residuals_vs_features.png")

    print(f"\nGenerated {len(saved_files)} diagnostic plots for {model_name}")
    return saved_files


if __name__ == "__main__":
    # Quick test
    print("Testing visualization module...")

    # Generate sample data
    np.random.seed(42)
    y_true = np.random.lognormal(mean=12, sigma=0.5, size=200)
    y_pred = y_true * np.random.lognormal(mean=0, sigma=0.15, size=200)

    X_sample = np.random.randn(200, 3)
    feat_names = ["GrLivArea", "BedroomAbvGr", "FullBath"]

    # Test plots
    plot_target_distribution(y_true, log_scale=True)
    plot_actual_vs_predicted(y_true, y_pred)
    plot_residuals(y_true, y_pred)
    plot_residuals_vs_feature(X_sample, feat_names, y_true, y_pred)
    plot_feature_importance(feat_names, np.array([150, 5000, 10000]))
    plot_model_comparison(["Model A", "Model B"], [{"Val_RMSE": 25000}, {"Val_RMSE": 30000}])

    print("\n✓ Visualization module test passed!")