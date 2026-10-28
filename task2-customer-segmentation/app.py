"""Mall Customer Segmentation Intelligence — Streamlit Web Application.

SkillCraft Technology — Machine Learning Internship (Task 2 Phase 4)
Interactive unsupervised customer clustering dashboard powered by scikit-learn K-Means.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure task2 root directory is on sys.path for robust package imports
TASK2_DIR = Path(__file__).resolve().parent
if str(TASK2_DIR) not in sys.path:
    sys.path.insert(0, str(TASK2_DIR))

from src.app_helpers import (
    CLUSTER_DESCRIPTIONS,
    compute_cluster_summary_table,
    create_new_customer_record,
    filter_customer_data,
    get_model_metadata,
)
from src.clustering import CustomerClustering
from src.data_loader import load_data


# Color constants for distinct, accessible cluster palettes
CLUSTER_PALETTE = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#06B6D4"]


# --- Cached Data and Model Loaders ---
@st.cache_data
def get_raw_data() -> pd.DataFrame:
    """Load normalized raw Mall Customers dataset."""
    return load_data()


@st.cache_data
def get_segmented_data() -> pd.DataFrame:
    """Load segmented customer dataset containing pre-computed cluster assignments."""
    seg_path = TASK2_DIR / "data" / "processed" / "mall_customers_segmented.csv"
    if seg_path.is_file():
        return pd.read_csv(seg_path)
    # Fallback if processed file is missing
    df = get_raw_data()
    m_classic = get_model("classic")
    m_ext = get_model("extended")
    df["Cluster_Classic_k5"] = m_classic.predict(df)
    df["Cluster_Extended_k5"] = m_ext.predict(df)
    return df


@st.cache_data
def get_evaluation_metrics() -> pd.DataFrame:
    """Load pre-computed model selection evaluation metrics."""
    metrics_path = TASK2_DIR / "reports" / "cluster_evaluation.csv"
    if metrics_path.is_file():
        return pd.read_csv(metrics_path)
    return pd.DataFrame()


@st.cache_resource
def get_model(model_type: str = "classic") -> CustomerClustering:
    """Load serialized K-Means clustering pipeline."""
    model_file = "kmeans_classic_k5.joblib" if model_type == "classic" else "kmeans_extended_k5.joblib"
    model_path = TASK2_DIR / "models" / model_file
    if not model_path.is_file():
        # Fallback fitting if model artifact is absent
        df = get_raw_data()
        model = CustomerClustering(n_clusters=5, feature_set=model_type, random_state=42)
        model.fit(df)
        return model
    return CustomerClustering.load(model_path)


# --- Custom UI Styles ---
def apply_custom_styles() -> None:
    st.markdown(
        """
        <style>
        /* Modern Typography and Spacing */
        .main-header {
            font-size: 2.1rem;
            font-weight: 800;
            color: #1E293B;
            margin-bottom: 0.2rem;
            letter-spacing: -0.02em;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748B;
            margin-bottom: 1.5rem;
            font-weight: 400;
        }
        .badge {
            display: inline-block;
            background-color: #EFF6FF;
            color: #2563EB;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            border: 1px solid #BFDBFE;
            margin-bottom: 0.5rem;
        }
        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 0.65rem;
            padding: 1rem 1.25rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            margin-bottom: 1rem;
        }
        .metric-title {
            font-size: 0.82rem;
            font-weight: 600;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-val {
            font-size: 1.7rem;
            font-weight: 700;
            color: #0F172A;
            margin-top: 0.2rem;
        }
        .metric-sub {
            font-size: 0.8rem;
            color: #059669;
            margin-top: 0.15rem;
        }
        .info-callout {
            background-color: #F8FAFC;
            border-left: 4px solid #2563EB;
            padding: 0.85rem 1.1rem;
            border-radius: 0 0.5rem 0.5rem 0;
            margin: 1rem 0;
            font-size: 0.92rem;
            color: #334155;
        }
        .warning-callout {
            background-color: #FFFBEB;
            border-left: 4px solid #F59E0B;
            padding: 0.85rem 1.1rem;
            border-radius: 0 0.5rem 0.5rem 0;
            margin: 1rem 0;
            font-size: 0.92rem;
            color: #92400E;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --- Section Renderers ---
def render_header() -> None:
    st.markdown('<span class="badge">SkillCraft ML Internship — Task 2</span>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">Mall Customer Segmentation Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Unsupervised Behavioral & Demographic Clustering via K-Means | Empirical Persona Profiling</div>',
        unsafe_allow_html=True,
    )


def render_sidebar(df_segmented: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    st.sidebar.markdown("### ⚙️ Segmentation Settings")

    model_option = st.sidebar.radio(
        "Active Segmentation Model",
        options=["Classic (2D)", "Extended (4D)"],
        index=0,
        help="Classic clusters on Annual Income + Spending Score. Extended also incorporates Age and Gender.",
    )
    model_type = "classic" if "Classic" in model_option else "extended"
    meta = get_model_metadata(model_type)

    st.sidebar.info(
        f"**{meta['name']}**\n\n"
        f"• **Features**: `{', '.join(meta['features'])}`\n"
        f"• **Optimal Clusters**: `{meta['k']}`\n"
        f"• **Silhouette Score**: `{meta['silhouette']}`"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Explorer Filters")
    st.sidebar.caption("Filters affect displayed data only. Fitted model weights are never modified.")

    cluster_col = meta["cluster_col"]
    all_clusters = sorted(df_segmented[cluster_col].unique().tolist())
    selected_clusters = st.sidebar.multiselect(
        "Filter by Cluster",
        options=all_clusters,
        default=all_clusters,
        format_func=lambda c: f"Cluster {c}: {CLUSTER_DESCRIPTIONS.get(c, {}).get('name', '')}",
    )

    min_income = float(df_segmented["Annual_Income"].min())
    max_income = float(df_segmented["Annual_Income"].max())
    income_range = st.sidebar.slider(
        "Annual Income ($k)",
        min_value=min_income,
        max_value=max_income,
        value=(min_income, max_income),
        step=1.0,
    )

    min_spending = float(df_segmented["Spending_Score"].min())
    max_spending = float(df_segmented["Spending_Score"].max())
    spending_range = st.sidebar.slider(
        "Spending Score (1-100)",
        min_value=min_spending,
        max_value=max_spending,
        value=(min_spending, max_spending),
        step=1.0,
    )

    gender_filter = st.sidebar.selectbox("Gender", options=["All", "Female", "Male"], index=0)

    filtered_df = filter_customer_data(
        df_segmented,
        selected_clusters=selected_clusters,
        income_range=income_range,
        spending_range=spending_range,
        selected_gender=gender_filter,
        cluster_col=cluster_col,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Displaying **{len(filtered_df)}** of **{len(df_segmented)}** customers "
        f"({len(filtered_df)/len(df_segmented)*100:.1f}%)"
    )

    return model_type, filtered_df


def render_overview_tab(df_segmented: pd.DataFrame, model_type: str) -> None:
    meta = get_model_metadata(model_type)
    cluster_col = meta["cluster_col"]

    st.markdown("### 📊 Executive Portfolio Overview")

    # Metric Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Total Customers</div>'
            f'<div class="metric-val">{len(df_segmented)}</div>'
            f'<div class="metric-sub">Complete Audit Sample</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        mean_inc = df_segmented["Annual_Income"].mean()
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Average Income</div>'
            f'<div class="metric-val">${mean_inc:.1f}k</div>'
            f'<div class="metric-sub">Range: ${df_segmented["Annual_Income"].min():.0f}k – ${df_segmented["Annual_Income"].max():.0f}k</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        mean_ss = df_segmented["Spending_Score"].mean()
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Avg Spending Score</div>'
            f'<div class="metric-val">{mean_ss:.1f}</div>'
            f'<div class="metric-sub">Scale: 1 – 100</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Behavioral Segments</div>'
            f'<div class="metric-val">{meta["k"]}</div>'
            f'<div class="metric-sub">Silhouette: {meta["silhouette"]}</div></div>',
            unsafe_allow_html=True,
        )

    # Executive Summary Card
    st.markdown(
        f"""
        <div class="info-callout">
            <strong>Dataset Synopsis:</strong> This intelligence dashboard analyzes 200 surveyed mall customer profiles
            across financial (Annual Income) and behavioral (Spending Score) dimensions. 
            Audited data contains <strong>0 missing values</strong> and <strong>0 duplicate records</strong>,
            comprising 56.0% Female and 44.0% Male customers.
            <br><br>
            <strong>Active Model:</strong> <em>{meta['name']}</em> segments customers into 
            <strong>{meta['k']} clusters</strong>. 
            <em>Notice:</em> Unsupervised clustering identifies empirical groupings in cross-sectional data; 
            it does not imply purchasing causality or prove customer intent.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 👥 Segment Population Breakdown")
    summary_table = compute_cluster_summary_table(df_segmented, cluster_col=cluster_col)
    st.dataframe(
        summary_table,
        use_container_width=True,
        hide_index=True,
    )

    # Demographic & Financial Distribution Plots
    st.markdown("#### 📈 Overall Attribute Distributions")
    col_a, col_b = st.columns(2)

    with col_a:
        inc_chart = (
            alt.Chart(df_segmented)
            .mark_bar(color="#0D9488", opacity=0.85)
            .encode(
                alt.X("Annual_Income:Q", bin=alt.Bin(maxbins=15), title="Annual Income ($k)"),
                alt.Y("count()", title="Customer Count"),
                tooltip=["count()"],
            )
            .properties(title="Annual Income Distribution ($k)", height=240)
        )
        st.altair_chart(inc_chart, use_container_width=True)

    with col_b:
        ss_chart = (
            alt.Chart(df_segmented)
            .mark_bar(color="#EA580C", opacity=0.85)
            .encode(
                alt.X("Spending_Score:Q", bin=alt.Bin(maxbins=15), title="Spending Score (1-100)"),
                alt.Y("count()", title="Customer Count"),
                tooltip=["count()"],
            )
            .properties(title="Spending Score Distribution (1-100)", height=240)
        )
        st.altair_chart(ss_chart, use_container_width=True)


def render_explorer_tab(filtered_df: pd.DataFrame, model_type: str) -> None:
    meta = get_model_metadata(model_type)
    cluster_col = meta["cluster_col"]
    model = get_model(model_type)

    st.markdown("### 🔍 Interactive Customer Cluster Explorer")

    callout_text = (
        "<strong>Classic Model (2D):</strong> Clusters purely on <code>Annual_Income</code> and "
        "<code>Spending_Score</code>. Age and Gender are NOT inputs to clustering."
        if model_type == "classic"
        else "<strong>Extended Model (4D):</strong> Clusters on <code>Age</code>, <code>Annual_Income</code>, "
        "<code>Spending_Score</code>, and binary <code>Gender</code>."
    )
    st.markdown(f'<div class="info-callout">{callout_text}</div>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("No customers match the current filter criteria. Adjust the sidebar filters to display data.")
        return

    # Prepare data for plotting
    plot_df = filtered_df.copy()
    plot_df["Cluster_Label"] = plot_df[cluster_col].apply(
        lambda c: f"Cluster {c}: {CLUSTER_DESCRIPTIONS.get(c, {}).get('persona', f'Segment {c}')}"
    )

    # Centroids in original domain units
    centers_df = model.get_cluster_centers(scaled=False).reset_index()
    centers_df["Cluster_Label"] = centers_df["index"].apply(
        lambda s: f"Centroid {s.replace('Cluster_', 'Cluster ')}"
    )

    # Interactive Altair Scatter Plot
    scatter = (
        alt.Chart(plot_df)
        .mark_circle(size=85, opacity=0.8)
        .encode(
            x=alt.X("Annual_Income:Q", title="Annual Income ($k)", scale=alt.Scale(zero=False, padding=10)),
            y=alt.Y("Spending_Score:Q", title="Spending Score (1-100)", scale=alt.Scale(domain=[0, 105])),
            color=alt.Color(
                "Cluster_Label:N",
                title="Customer Segment",
                scale=alt.Scale(range=CLUSTER_PALETTE),
            ),
            tooltip=[
                alt.Tooltip("CustomerID:Q", title="Customer ID"),
                alt.Tooltip("Cluster_Label:N", title="Segment"),
                alt.Tooltip("Annual_Income:Q", title="Income ($k)", format="$.1f"),
                alt.Tooltip("Spending_Score:Q", title="Spending Score", format=".0f"),
                alt.Tooltip("Age:Q", title="Age (yrs)"),
                alt.Tooltip("Gender:N", title="Gender"),
            ],
        )
    )

    # Centroid markers
    centroids_scatter = (
        alt.Chart(centers_df)
        .mark_point(shape="cross", size=220, strokeWidth=3, color="#000000")
        .encode(
            x="Annual_Income:Q",
            y="Spending_Score:Q",
            tooltip=[
                alt.Tooltip("Cluster_Label:N", title="Centroid"),
                alt.Tooltip("Annual_Income:Q", title="Mean Income ($k)", format="$.1f"),
                alt.Tooltip("Spending_Score:Q", title="Mean Spending Score", format=".1f"),
            ],
        )
    )

    final_chart = (
        (scatter + centroids_scatter)
        .properties(
            title=f"Annual Income vs. Spending Score ({meta['name']})",
            height=460,
        )
        .interactive()
    )

    st.altair_chart(final_chart, use_container_width=True)

    # Unscaled Cluster Centroids Table
    st.markdown("#### 🎯 Cluster Centroids (Original Unscaled Units)")
    st.caption("Centroids represent the multidimensional mean of each cluster in physical units.")
    centers_display = model.get_cluster_centers(scaled=False)
    st.dataframe(centers_display, use_container_width=True)


def render_profiles_tab(df_segmented: pd.DataFrame, model_type: str) -> None:
    meta = get_model_metadata(model_type)
    cluster_col = meta["cluster_col"]

    st.markdown("### 👥 Customer Segment Behavioral Profiles")
    st.markdown(
        f"""
        <div class="info-callout">
            Profiles describe empirical tendencies observed within each partition for <strong>{meta['name']}</strong>.
            All descriptions use objective, evidence-based retail language without ungrounded labels (e.g. "VIP").
        </div>
        """,
        unsafe_allow_html=True,
    )

    clusters = sorted(df_segmented[cluster_col].unique())
    total_customers = len(df_segmented)

    for c in clusters:
        group = df_segmented[df_segmented[cluster_col] == c]
        count = len(group)
        pct = count / total_customers * 100
        info = CLUSTER_DESCRIPTIONS.get(c, {
            "name": f"Cluster {c}",
            "persona": f"Segment {c}",
            "description": "Empirical cluster group.",
            "strategy": "Standard engagement.",
        })

        with st.expander(f"Cluster {c}: {info['persona']} ({count} customers — {pct:.1f}%)", expanded=(c in [0, 1])):
            col1, col2, col3 = st.columns([1.2, 1, 1.8])

            with col1:
                st.markdown(f"**Descriptive Label**: `{info['name']}`")
                st.markdown(f"**Cluster Size**: `{count} customers` ({pct:.1f}% of base)")
                st.markdown(f"**Mean Annual Income**: `${group['Annual_Income'].mean():.1f}k` (Median: `${group['Annual_Income'].median():.1f}k`)")
                st.markdown(f"**Mean Spending Score**: `{group['Spending_Score'].mean():.1f}/100` (Median: `{group['Spending_Score'].median():.1f}`)")

            with col2:
                female_pct = (group["Gender"] == "Female").mean() * 100
                male_pct = 100.0 - female_pct
                st.markdown(f"**Median Age**: `{group['Age'].median():.0f} years` (Mean: `{group['Age'].mean():.1f}`)")
                st.markdown(f"**Gender Balance**: `{female_pct:.1f}% Female` / `{male_pct:.1f}% Male`")
                if model_type == "classic":
                    st.caption("ℹ️ Age and gender are descriptive only; not fed to classic clustering.")

            with col3:
                st.markdown(f"**Observed Behavior**: {info['description']}")
                st.markdown(f"**Actionable Retail Strategy**: {info['strategy']}")
                st.caption(
                    "⚠️ *Methodological note: Correlation does not prove causation. "
                    "Observed spending reflects empirical tendencies in survey data.*"
                )


def render_prediction_tab(model_type: str) -> None:
    meta = get_model_metadata(model_type)
    model = get_model(model_type)

    st.markdown("### 🔮 Real-Time Customer Segment Inference")
    st.markdown(
        f"""
        <div class="info-callout">
            Input a customer's demographic and financial profile below to predict their segment under the 
            <strong>{meta['name']}</strong>.
            <br>
            <em>Notice:</em> Cluster labels (0, 1, 2, 3, 4) are arbitrary numerical identifiers assigned by 
            centroid initialization, not hierarchical scores or ranks.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("customer_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age_input = st.slider("Customer Age (years)", min_value=18, max_value=85, value=35, step=1)
            income_input = st.number_input(
                "Annual Income ($k USD)",
                min_value=10.0,
                max_value=200.0,
                value=65.0,
                step=1.0,
                help="Customer estimated annual household income in thousands of USD.",
            )
        with col2:
            spending_input = st.slider(
                "Spending Score (1-100)",
                min_value=1,
                max_value=100,
                value=50,
                step=1,
                help="Mall behavioral spending score assigned based on purchase frequency and basket size.",
            )
            gender_input = st.selectbox("Customer Gender", options=["Female", "Male"], index=0)

        predict_button = st.form_submit_button("Predict Customer Segment", use_container_width=True)

    if predict_button:
        try:
            # Build customer record
            cust_df = create_new_customer_record(
                age=age_input,
                annual_income=income_input,
                spending_score=spending_input,
                gender=gender_input,
            )

            # Predict without retraining
            predicted_cluster = int(model.predict(cust_df)[0])
            info = CLUSTER_DESCRIPTIONS.get(predicted_cluster, {
                "name": f"Cluster {predicted_cluster}",
                "persona": f"Segment {predicted_cluster}",
                "description": "Empirical cluster group.",
                "strategy": "Standard customer engagement.",
            })

            st.success(f"### Predicted Segment: Cluster {predicted_cluster} — {info['persona']}")

            p_col1, p_col2 = st.columns([1, 1.5])
            with p_col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Assigned Cluster</div>
                        <div class="metric-val" style="color: #2563EB;">Cluster {predicted_cluster}</div>
                        <div class="metric-sub">{info['name']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if model_type == "classic":
                    st.info(
                        "**Features Evaluated**: `Annual_Income` ($"
                        f"{income_input:.0f}k) & `Spending_Score` ({spending_input}).\n\n"
                        "Age and Gender were passed for context but ignored by the classic model."
                    )
                else:
                    st.info(
                        "**Features Evaluated**: `Age` ("
                        f"{age_input}), `Annual_Income` (${income_input:.0f}k), `Spending_Score` ({spending_input}), & `Gender` ({gender_input})."
                    )

            with p_col2:
                st.markdown(f"**Segment Persona**: `{info['persona']}`")
                st.markdown(f"**Behavioral Tendency**: {info['description']}")
                st.markdown(f"**Recommended Strategic Approach**: {info['strategy']}")

                # Comparison with cluster centroid
                centers = model.get_cluster_centers(scaled=False)
                c_row = centers.loc[f"Cluster_{predicted_cluster}"]
                st.markdown(
                    f"**Comparison to Segment Centroid**:\n"
                    f"- Customer Income: **${income_input:.1f}k** vs Centroid Mean: **${c_row['Annual_Income']:.1f}k**\n"
                    f"- Customer Spending Score: **{spending_input}** vs Centroid Mean: **{c_row['Spending_Score']:.1f}**"
                )

        except Exception as e:
            st.error(f"Error evaluating customer profile: {str(e)}")


def render_evaluation_tab() -> None:
    st.markdown("### 📈 Model Selection & Unsupervised Evaluation")
    st.markdown(
        """
        <div class="info-callout">
            Model selection was conducted across <strong>k = 2 to 10</strong> evaluating Inertia (Within-Cluster Sum of Squares),
            Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score.
            <br>
            <em>Methodological Notice:</em> Metrics from different feature spaces (2D vs 4D) cannot be compared directly, 
            as additional dimensions naturally decrease Euclidean density.
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics_df = get_evaluation_metrics()
    if not metrics_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Classic Model Evaluation Metrics")
            classic_metrics = metrics_df[metrics_df["feature_set"] == "classic"].drop(columns=["feature_set"])
            st.dataframe(classic_metrics, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("#### Extended Model Evaluation Metrics")
            ext_metrics = metrics_df[metrics_df["feature_set"] == "extended"].drop(columns=["feature_set"])
            st.dataframe(ext_metrics, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 📉 Publication-Ready Diagnostic Curves")
    st.caption("Curves generated during Phase 3 rigorous validation across candidate k values.")

    col_fig1, col_fig2 = st.columns(2)
    figures_dir = TASK2_DIR / "reports" / "figures"

    with col_fig1:
        st.markdown("**Classic Model (2D) Diagnostic Curves**")
        p_elbow_classic = figures_dir / "elbow_curve_classic.png"
        p_sil_classic = figures_dir / "silhouette_curve_classic.png"
        if p_elbow_classic.is_file():
            st.image(str(p_elbow_classic), caption="Classic Elbow Method (Inflection at k=5)", use_container_width=True)
        if p_sil_classic.is_file():
            st.image(str(p_sil_classic), caption="Classic Silhouette Score (Peak at k=5: 0.5547)", use_container_width=True)

    with col_fig2:
        st.markdown("**Extended Model (4D) Diagnostic Curves**")
        p_elbow_ext = figures_dir / "elbow_curve_extended.png"
        p_sil_ext = figures_dir / "silhouette_curve_extended.png"
        if p_elbow_ext.is_file():
            st.image(str(p_elbow_ext), caption="Extended Elbow Method", use_container_width=True)
        if p_sil_ext.is_file():
            st.image(str(p_sil_ext), caption="Extended Silhouette Score (k=6 peak: 0.3565 vs k=5: 0.3498)", use_container_width=True)

    st.markdown(
        """
        <div class="warning-callout">
            <strong>Key Metric Takeaways:</strong><br>
            • <strong>Classic k = 5</strong> is the unambiguous behavioral optimum with peak silhouette 
            (<strong>0.5547</strong>) and minimum Davies-Bouldin index (<strong>0.5722</strong>).<br>
            • <strong>Extended k = 6</strong> achieves a slightly higher silhouette score (<strong>0.3565</strong>) 
            than extended k = 5 (<strong>0.3498</strong>), because it further partitions gender. 
            Extended k = 5 is selected as an interpretable benchmark preserving core income/spending quadrants.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_methodology_tab(df_segmented: pd.DataFrame) -> None:
    st.markdown("### 📑 Data Provenance & ML Methodology")

    st.markdown(
        """
        #### 1. Dataset Provenance & Integrity
        - **Source**: Mall Customer Segmentation Dataset (`vjchoudhary7/customer-segmentation-tutorial-in-python`).
        - **Observations**: 200 customer records, 5 attributes.
        - **Data Quality**: Verified 0 missing/null entries and 0 duplicate rows.
        - **CustomerID Exclusion**: `CustomerID` is an arbitrary sequential key without behavioral significance; 
          it is strictly excluded from all model training and distance computations.

        #### 2. Preprocessing & Feature Engineering
        - **StandardScaler**: Numeric features are centered to $\mu = 0$ and scaled to $\sigma = 1$. 
          This prevents attributes with wide numerical ranges (e.g., Annual Income: $15k–$137k) from artificially 
          dominating Euclidean distance calculations over smaller-scale attributes.
        - **One-Hot Encoding**: Gender is encoded as binary `Gender_Male` (`Female = 0.0`, `Male = 1.0`) with `drop='first'`.
        - **Leakage Prevention**: All scaling parameters and categories are fitted strictly on training subsets.

        #### 3. Clustering Architecture & Assumptions
        - **Algorithm**: K-Means clustering with `k-means++` initialization and `n_init=10`.
        - **Assumptions**: K-Means assumes isotropic, convex clusters of similar variance.
        - **Limitations**: Non-probabilistic hard cluster assignments; sensitive to outliers and feature scaling.
        """
    )

    st.markdown("---")
    st.markdown("#### 📋 Segmented Customer Data Inspection")

    row_count = st.selectbox("Display Row Count", options=[10, 25, 50, 100, 200], index=0)
    st.dataframe(df_segmented.head(row_count), use_container_width=True)

    # Download button for processed dataset
    csv_data = df_segmented.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Segmented Customers CSV",
        data=csv_data,
        file_name="mall_customers_segmented.csv",
        mime="text/csv",
        use_container_width=True,
    )


# --- Main Application Controller ---
def main() -> None:
    st.set_page_config(
        page_title="Customer Segmentation Intelligence | SkillCraft ML",
        page_icon="🛍️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_styles()
    render_header()

    # Load data
    df_segmented = get_segmented_data()

    # Render Sidebar and retrieve active configuration
    model_type, filtered_df = render_sidebar(df_segmented)

    # Navigation Tabs
    tab_overview, tab_explorer, tab_profiles, tab_prediction, tab_eval, tab_methodology = st.tabs(
        [
            "📊 Executive Overview",
            "🔍 Cluster Explorer",
            "👥 Segment Profiles",
            "🔮 Customer Prediction",
            "📈 Model Evaluation",
            "📑 Data & Methodology",
        ]
    )

    with tab_overview:
        render_overview_tab(df_segmented, model_type)

    with tab_explorer:
        render_explorer_tab(filtered_df, model_type)

    with tab_profiles:
        render_profiles_tab(df_segmented, model_type)

    with tab_prediction:
        render_prediction_tab(model_type)

    with tab_eval:
        render_evaluation_tab()

    with tab_methodology:
        render_methodology_tab(df_segmented)


if __name__ == "__main__":
    main()
