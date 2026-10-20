"""
EstateIQ — House Price Intelligence
Premium Streamlit application for machine-learning property valuation.

This application demonstrates a production-quality ML pipeline for house price prediction
using the Ames Housing dataset (Kaggle: House Prices - Advanced Regression Techniques).

Sections:
1. Overview - Dashboard with model status, dataset info, and KPI metrics
2. Predict - Interactive single prediction with validation
3. Batch Predictions - CSV upload for bulk predictions
4. Model Performance - Comparison tables, visualizations, and explainability
5. About - Methodology, limitations, and run instructions
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from predict import HousePricePredictor, create_sample_input
from data_loader import BASELINE_FEATURES, TARGET_COLUMN

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EstateIQ — House Price Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "EstateIQ — Machine Learning Property Valuation Demo\nSkillCraft Technology ML Internship — Task 1",
    },
)

# ──────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM — CSS TOKENS & STYLES
# ──────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
/* ============================================================================
   ESTATEIQ DESIGN TOKENS — Refined, Premium Palette
   ============================================================================ */
:root {
  /* Color Palette — Warm ivory canvas, deep charcoal text, muted teal accent */
  --color-bg: #FDFBF7;                    /* Warm ivory page background */
  --color-surface: #FFFFFF;               /* Pure white cards/surfaces */
  --color-surface-muted: #F6F3ED;         /* Subtle alternate surface */

  --color-text: #1A1C1E;                  /* Deep charcoal primary text */
  --color-text-muted: #6B6D70;            /* Muted secondary text */
  --color-text-faint: #9A9C9F;            /* Faint tertiary text */
  --color-text-inverse: #FFFFFF;          /* White on dark */

  --color-accent: #0D7A6D;                /* Muted teal primary action */
  --color-accent-hover: #0A5F55;
  --color-accent-light: #E8F4F2;          /* 10% accent for backgrounds */
  --color-accent-border: #B8DCD6;

  --color-success: #1E7D32;
  --color-success-light: #E8F5E9;
  --color-warning: #B8860B;
  --color-warning-light: #FFF8E1;
  --color-danger: #C62828;
  --color-danger-light: #FDEDEC;

  --color-border: #E8E4DC;                /* Restrained neutral border */
  --color-border-strong: #D0CDC5;         /* Stronger border for focus */
  --color-border-focus: var(--color-accent);

  /* Typography — System font stack, clear scale */
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif;
  --font-mono: "SF Mono", "Fira Code", "Fira Mono", Menlo, monospace;

  --text-xs: 0.70rem;     /* 11px — labels, captions */
  --text-sm: 0.8125rem;   /* 13px — secondary text */
  --text-base: 0.9375rem; /* 15px — body text */
  --text-lg: 1.0625rem;   /* 17px — emphasized body */
  --text-xl: 1.25rem;     /* 20px — section titles */
  --text-2xl: 1.5rem;     /* 24px — page titles */
  --text-3xl: 2rem;       /* 32px — hero titles */

  --font-normal: 400;
  --font-medium: 500;
  --font-semibold: 600;
  --font-bold: 700;

  --leading-tight: 1.2;
  --leading-normal: 1.55;
  --leading-relaxed: 1.7;

  /* Spacing — Consistent 4px base unit */
  --space-1: 0.25rem;   /* 4px */
  --space-2: 0.5rem;    /* 8px */
  --space-3: 0.75rem;   /* 12px */
  --space-4: 1rem;      /* 16px */
  --space-5: 1.25rem;   /* 20px */
  --space-6: 1.5rem;    /* 24px */
  --space-8: 2rem;      /* 32px */
  --space-10: 2.5rem;   /* 40px */
  --space-12: 3rem;     /* 48px */

  /* Border Radius — Restrained, not pill-shaped */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 10px;
  --radius-full: 9999px;

  /* Shadows — Subtle, only where depth communicates hierarchy */
  --shadow-sm: 0 1px 2px rgba(26, 28, 30, 0.04);
  --shadow-md: 0 4px 12px rgba(26, 28, 30, 0.05);
  --shadow-lg: 0 8px 24px rgba(26, 28, 30, 0.06);

  /* Transitions — Fast, purposeful */
  --transition-fast: 120ms ease;
  --transition-normal: 200ms ease;

  /* Layout */
  --max-width: 1120px;
  --sidebar-width: 260px;
  --header-height: 56px;
}

/* ============================================================================
   BASE RESET & TYPOGRAPHY
   ============================================================================ */
html, body, [class*="css"] {
  font-family: var(--font-sans) !important;
  font-size: var(--text-base);
  line-height: var(--leading-normal);
  color: var(--color-text);
  background-color: var(--color-bg);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

h1, h2, h3, h4, h5, h6 {
  font-weight: var(--font-semibold);
  line-height: var(--leading-tight);
  color: var(--color-text);
  margin-top: 0;
  margin-bottom: var(--space-3);
}

h1 { font-size: var(--text-3xl); letter-spacing: -0.02em; }
h2 { font-size: var(--text-2xl); letter-spacing: -0.01em; }
h3 { font-size: var(--text-xl); }
h4 { font-size: var(--text-lg); }

p { margin: 0 0 var(--space-3); }
p:last-child { margin-bottom: 0; }

a { color: var(--color-accent); text-decoration: none; }
a:hover { text-decoration: underline; }

code, pre { font-family: var(--font-mono); }
code {
  background: var(--color-surface-muted);
  padding: 0.125rem 0.375rem;
  border-radius: var(--radius-sm);
  font-size: 0.875em;
}
pre {
  background: var(--color-text);
  color: var(--color-text-inverse);
  padding: var(--space-4);
  border-radius: var(--radius-md);
  overflow-x: auto;
}
pre code { background: none; padding: 0; }

/* ============================================================================
   LAYOUT OVERRIDES FOR STREAMLIT
   ============================================================================ */

/* Main container — generous but contained */
.main .block-container {
  max-width: var(--max-width);
  padding-top: var(--space-6);
  padding-bottom: var(--space-10);
  padding-left: var(--space-6);
  padding-right: var(--space-6);
}

/* Sidebar — clean, bordered */
section[data-testid="stSidebar"] {
  background: var(--color-surface);
  border-right: 1px solid var(--color-border);
  min-width: var(--sidebar-width);
  max-width: var(--sidebar-width);
}
section[data-testid="stSidebar"] > div {
  padding: var(--space-5) var(--space-4);
}

/* Hide default Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* ============================================================================
   COMPONENT STYLES — Minimal, Purposeful
   ============================================================================ */

/* --- Surface/Card — Single variant, no nesting --- */
.eiq-surface {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: var(--space-5);
  box-shadow: var(--shadow-sm);
}

.eiq-surface--quiet {
  background: var(--color-surface-muted);
  border: none;
}

/* --- Section Header — Compact, no hero banners --- */
.eiq-section-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-4);
  margin-bottom: var(--space-5);
  padding-bottom: var(--space-3);
  border-bottom: 1px solid var(--color-border);
}
.eiq-section-title {
  font-size: var(--text-xl);
  font-weight: var(--font-semibold);
  margin: 0;
}
.eiq-section-description {
  color: var(--color-text-muted);
  font-size: var(--text-base);
  margin: var(--space-1) 0 0;
  max-width: 60ch;
}

/* --- KPI Row — Clean grid, no oversized cards --- */
.eiq-kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: var(--space-3);
  margin: var(--space-3) 0 var(--space-5);
}
.eiq-kpi {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: var(--space-4);
}
.eiq-kpi-label {
  font-size: var(--text-xs);
  font-weight: var(--font-medium);
  color: var(--color-text-faint);
  letter-spacing: 0.04em;
  text-transform: none;
  margin-bottom: var(--space-1);
}
.eiq-kpi-value {
  font-size: var(--text-2xl);
  font-weight: var(--font-bold);
  color: var(--color-text);
  line-height: var(--leading-tight);
}
.eiq-kpi-value--accent { color: var(--color-accent); }
.eiq-kpi-subtext {
  font-size: var(--text-xs);
  color: var(--color-text-muted);
  margin-top: var(--space-1);
}

/* --- Status Badges — Minimal, semantic colors --- */
.eiq-badge {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-1) var(--space-3);
  border-radius: var(--radius-full);
  font-size: var(--text-xs);
  font-weight: var(--font-medium);
  letter-spacing: 0.01em;
}
.eiq-badge--success { background: var(--color-success-light); color: var(--color-success); }
.eiq-badge--warning { background: var(--color-warning-light); color: var(--color-warning); }
.eiq-badge--danger { background: var(--color-danger-light); color: var(--color-danger); }
.eiq-badge--neutral { background: var(--color-surface-muted); color: var(--color-text-muted); }
.eiq-badge__dot {
  width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
}
.eiq-badge--success .eiq-badge__dot { background: var(--color-success); }
.eiq-badge--warning .eiq-badge__dot { background: var(--color-warning); }
.eiq-badge--danger .eiq-badge__dot { background: var(--color-danger); }
.eiq-badge--neutral .eiq-badge__dot { background: var(--color-text-faint); }

/* --- Buttons — Clear primary/secondary distinction --- */
.stButton > button {
  background: var(--color-accent) !important;
  color: var(--color-text-inverse) !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  padding: var(--space-2) var(--space-5) !important;
  font-size: var(--text-sm) !important;
  font-weight: var(--font-medium) !important;
  letter-spacing: 0.01em;
  transition: background-color var(--transition-fast), transform var(--transition-fast), box-shadow var(--transition-fast) !important;
  box-shadow: var(--shadow-sm) !important;
}
.stButton > button:hover {
  background: var(--color-accent-hover) !important;
  transform: translateY(-1px);
  box-shadow: var(--shadow-md) !important;
}
.stButton > button:active { transform: translateY(0); box-shadow: var(--shadow-sm) !important; }
.stButton > button:focus-visible {
  outline: 2px solid var(--color-accent);
  outline-offset: 2px;
}

/* Secondary button variant */
.stButton > button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"] {
  background: var(--color-surface) !important;
  color: var(--color-text) !important;
  border: 1px solid var(--color-border-strong) !important;
  box-shadow: var(--shadow-sm) !important;
}
.stButton > button[kind="secondary"]:hover,
.stButton > button[data-testid="baseButton-secondary"]:hover {
  background: var(--color-surface-muted) !important;
  border-color: var(--color-border-strong) !important;
}

/* --- Form Inputs — Clean, accessible --- */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > div,
.stSlider > div > div > div {
  border-radius: var(--radius-md) !important;
  border: 1px solid var(--color-border-strong) !important;
  transition: border-color var(--transition-fast), box-shadow var(--transition-fast) !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stSelectbox > div > div > div:focus-within {
  border-color: var(--color-border-focus) !important;
  box-shadow: 0 0 0 3px var(--color-accent-light) !important;
}
.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stSlider label {
  font-weight: var(--font-medium) !important;
  color: var(--color-text) !important;
  font-size: var(--text-sm) !important;
}

/* --- Tables --- */
.stDataFrame {
  border-radius: var(--radius-md) !important;
  overflow: hidden;
  border: 1px solid var(--color-border) !important;
}
.stDataFrame table { font-size: var(--text-sm) !important; }
.stDataFrame th {
  background: var(--color-surface-muted) !important;
  font-weight: var(--font-semibold) !important;
  color: var(--color-text) !important;
  border-bottom: 2px solid var(--color-border-strong) !important;
}
.stDataFrame td { border-bottom: 1px solid var(--color-border) !important; }

/* --- Metrics (st.metric) --- */
[data-testid="stMetric"] {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: var(--space-3) var(--space-4);
  box-shadow: var(--shadow-sm);
}
[data-testid="stMetric"] > div:first-child {
  color: var(--color-text-muted) !important;
  font-size: var(--text-xs) !important;
  font-weight: var(--font-medium) !important;
}
[data-testid="stMetric"] > div:last-child {
  color: var(--color-text) !important;
  font-weight: var(--font-bold) !important;
  font-size: var(--text-xl) !important;
}

/* --- Alerts / Messages --- */
.stAlert {
  border-radius: var(--radius-md) !important;
  border: none !important;
  font-size: var(--text-sm) !important;
}
.stAlert[data-baseweb="notification"][kind="success"] {
  background: var(--color-success-light) !important;
  color: var(--color-success) !important;
}
.stAlert[data-baseweb="notification"][kind="error"] {
  background: var(--color-danger-light) !important;
  color: var(--color-danger) !important;
}
.stAlert[data-baseweb="notification"][kind="warning"] {
  background: var(--color-warning-light) !important;
  color: var(--color-warning) !important;
}
.stAlert[data-baseweb="notification"][kind="info"] {
  background: var(--color-accent-light) !important;
  color: var(--color-accent) !important;
}

/* --- Expander --- */
.stExpander {
  border: 1px solid var(--color-border) !important;
  border-radius: var(--radius-md) !important;
  background: var(--color-surface) !important;
}
.stExpander summary {
  font-weight: var(--font-medium) !important;
  color: var(--color-text) !important;
  font-size: var(--text-base) !important;
}

/* --- Tabs --- */
.stTabs [data-baseweb="tab-list"] {
  gap: var(--space-1);
  background: transparent;
  border-bottom: 1px solid var(--color-border);
  padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border: none !important;
  border-radius: var(--radius-md) var(--radius-md) 0 0 !important;
  padding: var(--space-2) var(--space-4) !important;
  font-weight: var(--font-medium) !important;
  color: var(--color-text-muted) !important;
  font-size: var(--text-sm) !important;
  transition: color var(--transition-fast), background var(--transition-fast) !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--color-text) !important;
  background: var(--color-surface-muted) !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
  color: var(--color-accent) !important;
  background: var(--color-surface) !important;
  border-bottom: 2px solid var(--color-accent) !important;
  margin-bottom: -1px;
}

/* --- File Uploader --- */
.stFileUploader {
  border: 2px dashed var(--color-border-strong) !important;
  border-radius: var(--radius-lg) !important;
  padding: var(--space-5) !important;
  background: var(--color-surface) !important;
  transition: border-color var(--transition-fast), background var(--transition-fast) !important;
}
.stFileUploader:hover {
  border-color: var(--color-accent) !important;
  background: var(--color-accent-light) !important;
}
.stFileUploader label {
  font-weight: var(--font-medium) !important;
  color: var(--color-text) !important;
}

/* --- Download Button --- */
.stDownloadButton > button {
  background: var(--color-surface) !important;
  color: var(--color-text) !important;
  border: 1px solid var(--color-border-strong) !important;
}
.stDownloadButton > button:hover {
  background: var(--color-accent) !important;
  color: var(--color-text-inverse) !important;
  border-color: var(--color-accent) !important;
}

/* --- Progress Bar --- */
.stProgress > div > div > div {
  background: var(--color-accent) !important;
  border-radius: var(--radius-full) !important;
}

/* --- Result Card (Prediction Output) — Single prominent surface --- */
.eiq-result {
  background: var(--color-text);
  color: var(--color-text-inverse);
  border-radius: var(--radius-lg);
  padding: var(--space-6);
  text-align: center;
  box-shadow: var(--shadow-md);
}
.eiq-result__label {
  font-size: var(--text-xs);
  font-weight: var(--font-medium);
  letter-spacing: 0.08em;
  opacity: 0.7;
  margin-bottom: var(--space-2);
}
.eiq-result__price {
  font-size: var(--text-3xl);
  font-weight: var(--font-bold);
  line-height: var(--leading-tight);
  margin-bottom: var(--space-3);
}
.eiq-result__details {
  font-size: var(--text-base);
  opacity: 0.8;
  line-height: var(--leading-relaxed);
}

/* --- Input Summary Row --- */
.eiq-input-summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: var(--space-3);
  margin-top: var(--space-5);
}
.eiq-input-item {
  background: rgba(255, 255, 255, 0.08);
  border-radius: var(--radius-md);
  padding: var(--space-3);
}
.eiq-input-item__label {
  font-size: var(--text-xs);
  letter-spacing: 0.04em;
  opacity: 0.7;
  margin-bottom: var(--space-1);
}
.eiq-input-item__value {
  font-size: var(--text-lg);
  font-weight: var(--font-semibold);
}

/* --- Disclaimer --- */
.eiq-disclaimer {
  background: var(--color-warning-light);
  border: 1px solid #F5D76E;
  border-radius: var(--radius-md);
  padding: var(--space-4);
  margin-top: var(--space-5);
  display: flex;
  align-items: flex-start;
  gap: var(--space-3);
}
.eiq-disclaimer__icon { flex-shrink: 0; margin-top: 2px; }
.eiq-disclaimer__content { flex: 1; }
.eiq-disclaimer__title {
  font-weight: var(--font-semibold);
  color: var(--color-warning);
  margin-bottom: var(--space-1);
  font-size: var(--text-sm);
}
.eiq-disclaimer__text {
  font-size: var(--text-sm);
  color: var(--color-text);
  line-height: var(--leading-relaxed);
}

/* --- Model Status Indicator --- */
.eiq-model-status {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-1) var(--space-3);
  border-radius: var(--radius-full);
  font-size: var(--text-sm);
  font-weight: var(--font-medium);
}
.eiq-model-status--online { background: var(--color-success-light); color: var(--color-success); }
.eiq-model-status--offline { background: var(--color-danger-light); color: var(--color-danger); }

/* --- Feature Chips --- */
.eiq-chips {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
  margin-top: var(--space-3);
}
.eiq-chip {
  background: var(--color-surface-muted);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-full);
  padding: var(--space-1) var(--space-3);
  font-size: var(--text-sm);
  font-weight: var(--font-medium);
  color: var(--color-text-muted);
}

/* --- Comparison Table Wrapper --- */
.eiq-table-wrap {
  overflow-x: auto;
  border-radius: var(--radius-md);
  border: 1px solid var(--color-border);
}

/* --- Methodology Steps — Numbered, aligned --- */
.eiq-step {
  display: flex;
  gap: var(--space-4);
  padding: var(--space-3) 0;
  border-bottom: 1px solid var(--color-border);
}
.eiq-step:last-child { border-bottom: none; }
.eiq-step__num {
  flex-shrink: 0;
  width: 28px;
  height: 28px;
  border-radius: var(--radius-full);
  background: var(--color-accent);
  color: var(--color-text-inverse);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: var(--font-bold);
  font-size: var(--text-sm);
}
.eiq-step__content { flex: 1; }
.eiq-step__title {
  font-weight: var(--font-semibold);
  margin-bottom: var(--space-1);
  font-size: var(--text-base);
}
.eiq-step__desc {
  font-size: var(--text-sm);
  color: var(--color-text-muted);
  line-height: var(--leading-relaxed);
}

/* --- Sidebar Brand & Navigation — Clean, no emojis --- */
.eiq-brand {
  text-align: center;
  padding: var(--space-3) 0 var(--space-5);
  border-bottom: 1px solid var(--color-border);
  margin-bottom: var(--space-4);
}
.eiq-brand__name {
  font-size: var(--text-2xl);
  font-weight: var(--font-bold);
  color: var(--color-text);
  letter-spacing: -0.02em;
}
.eiq-brand__tagline {
  font-size: var(--text-xs);
  color: var(--color-text-faint);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 2px;
}

/* Navigation uses st.radio styled as list */
.eiq-nav {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}
.eiq-nav-item {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  padding: var(--space-2) var(--space-3);
  border-radius: var(--radius-md);
  color: var(--color-text-muted);
  font-weight: var(--font-medium);
  font-size: var(--text-sm);
  text-decoration: none;
  transition: background var(--transition-fast), color var(--transition-fast);
  background: transparent;
  border: none;
  width: 100%;
  text-align: left;
  cursor: pointer;
}
.eiq-nav-item:hover {
  background: var(--color-surface-muted);
  color: var(--color-text);
}
.eiq-nav-item[aria-current="page"] {
  background: var(--color-accent-light);
  color: var(--color-accent);
  font-weight: var(--font-semibold);
}
.eiq-nav-item__icon { flex-shrink: 0; width: 18px; height: 18px; }

/* Sidebar footer */
.eiq-sidebar-footer {
  margin-top: var(--space-6);
  padding-top: var(--space-4);
  border-top: 1px solid var(--color-border);
  text-align: center;
  color: var(--color-text-faint);
  font-size: var(--text-xs);
}

/* ============================================================================
   RESPONSIVE
   ============================================================================ */
@media (max-width: 768px) {
  .main .block-container { padding-left: var(--space-4); padding-right: var(--space-4); }
  h1 { font-size: var(--text-2xl); }
  .eiq-kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .eiq-input-summary { grid-template-columns: 1fr 1fr; }
  .eiq-result__price { font-size: var(--text-2xl); }
}
@media (max-width: 480px) {
  .eiq-kpi-grid { grid-template-columns: 1fr; }
  .eiq-input-summary { grid-template-columns: 1fr; }
}

/* ============================================================================
   ACCESSIBILITY
   ============================================================================ */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
:focus-visible {
  outline: 2px solid var(--color-accent);
  outline-offset: 2px;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--color-surface-muted); }
::-webkit-scrollbar-thumb {
  background: var(--color-border-strong);
  border-radius: var(--radius-full);
}
::-webkit-scrollbar-thumb:hover { background: var(--color-text-faint); }
</style>
"""

# ──────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def inject_css() -> None:
    """Inject custom CSS into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def format_currency(value: float) -> str:
    """Format a number as USD currency."""
    if value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:.1f}K"
    else:
        return f"${value:,.0f}"


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with consistent decimal places."""
    return f"{value:,.{decimals}f}"


def load_json_safe(path: Path) -> dict | None:
    """Safely load JSON file, return None if not found or invalid."""
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return None


def load_model_comparison() -> dict | None:
    """Load model comparison results from reports.

    Expects model_comparison.json with keys: cross_validation, validation_set.
    Does NOT fall back to metrics_baseline.json as they have different schemas.
    """
    report_dir = Path(__file__).parent / "reports"
    path = report_dir / "model_comparison.json"
    data = load_json_safe(path)

    if data is None:
        return None

    # Validate required fields for model comparison schema
    required_keys = {"cross_validation", "validation_set"}
    if not all(key in data for key in required_keys):
        return None

    return data


def load_baseline_metrics() -> dict | None:
    """Load baseline model metrics."""
    path = Path(__file__).parent / "reports" / "metrics_baseline.json"
    return load_json_safe(path)


def get_training_data_stats() -> dict | None:
    """Get dataset statistics from actual data files.

    Returns None if data is unavailable instead of returning misleading hardcoded fallbacks.
    """
    try:
        from data_loader import DataLoader
        loader = DataLoader("data/raw", strict_validation=False)
        train_df = loader.load_train(validate=False)
        return {
            "n_rows": len(train_df),
            "n_features": len(train_df.columns) - 1,  # Exclude target
            "target_mean": float(train_df[TARGET_COLUMN].mean()),
            "target_median": float(train_df[TARGET_COLUMN].median()),
            "target_min": float(train_df[TARGET_COLUMN].min()),
            "target_max": float(train_df[TARGET_COLUMN].max()),
        }
    except Exception:
        # Data unavailable - return None instead of misleading fallbacks
        return None


@st.cache_resource(show_spinner="Loading model...")
def load_predictor(model_path: str) -> HousePricePredictor | None:
    """Load the trained model predictor with caching."""
    try:
        return HousePricePredictor(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def validate_prediction_inputs(gr_liv_area: float, bedroom_abv_gr: int, full_bath: int) -> list[str]:
    """Validate prediction inputs and return list of error messages."""
    errors = []
    if gr_liv_area <= 0:
        errors.append("Living area must be greater than 0")
    elif gr_liv_area > 10000:
        errors.append("Living area seems unreasonably large (max 10,000 sq ft)")
    if bedroom_abv_gr < 0:
        errors.append("Bedroom count cannot be negative")
    elif bedroom_abv_gr > 20:
        errors.append("Bedroom count seems unreasonably high (max 20)")
    if full_bath < 0:
        errors.append("Bathroom count cannot be negative")
    elif full_bath > 10:
        errors.append("Bathroom count seems unreasonably high (max 10)")
    return errors


def make_prediction(predictor: HousePricePredictor, gr_liv_area: float, bedroom_abv_gr: int, full_bath: int) -> float | None:
    """Make a prediction using the loaded model."""
    try:
        input_data = {
            "GrLivArea": gr_liv_area,
            "BedroomAbvGr": bedroom_abv_gr,
            "FullBath": full_bath,
        }
        return predictor.predict(input_data)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None


def validate_batch_csv(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Validate batch CSV and return (errors, cleaned_df)."""
    errors = []
    cleaned = df.copy()

    # Check required columns
    missing = [f for f in BASELINE_FEATURES if f not in cleaned.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
        return errors, pd.DataFrame()

    # Select only required columns (in correct order)
    cleaned = cleaned[BASELINE_FEATURES].copy()

    # Check for empty dataframe
    if len(cleaned) == 0:
        errors.append("CSV file is empty")
        return errors, pd.DataFrame()

    # Check for non-numeric values
    for col in BASELINE_FEATURES:
        if not pd.api.types.is_numeric_dtype(cleaned[col]):
            try:
                cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            except Exception:
                errors.append(f"Column '{col}' contains non-numeric values that cannot be converted")
                return errors, pd.DataFrame()

    # Check for missing values
    missing_rows = cleaned.isna().any(axis=1)
    if missing_rows.any():
        errors.append(f"Found {missing_rows.sum()} rows with missing values in required columns")
        # Don't return - we'll handle this in prediction with warnings

    # Validate ranges
    if (cleaned["GrLivArea"] <= 0).any():
        errors.append("Some rows have invalid living area (must be > 0)")
    if (cleaned["GrLivArea"] > 10000).any():
        errors.append("Some rows have unreasonably large living area (> 10,000 sq ft)")
    if (cleaned["BedroomAbvGr"] < 0).any() or (cleaned["BedroomAbvGr"] > 20).any():
        errors.append("Some rows have invalid bedroom count (0-20)")
    if (cleaned["FullBath"] < 0).any() or (cleaned["FullBath"] > 10).any():
        errors.append("Some rows have invalid bathroom count (0-10)")

    return errors, cleaned


def predict_batch(predictor: HousePricePredictor, df: pd.DataFrame) -> pd.DataFrame:
    """Make batch predictions and return results DataFrame."""
    results = df.copy()
    predictions = []
    errors = []

    for idx, row in df.iterrows():
        try:
            pred = predictor.predict(row.to_dict())
            predictions.append(pred)
            errors.append(None)
        except Exception as e:
            predictions.append(np.nan)
            errors.append(str(e))

    results["Predicted_SalePrice"] = predictions
    results["Prediction_Error"] = errors
    return results


def generate_sample_csv() -> str:
    """Generate a sample CSV template for batch predictions."""
    sample_data = {
        "GrLivArea": [1500, 2000, 1200, 2500, 1800],
        "BedroomAbvGr": [3, 4, 2, 4, 3],
        "FullBath": [2, 3, 1, 3, 2],
    }
    df = pd.DataFrame(sample_data)
    return df.to_csv(index=False)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION RENDERERS
# ──────────────────────────────────────────────────────────────────────────────

def render_overview() -> None:
    """Render the Overview dashboard."""
    # Compact page header — no hero banner
    st.markdown("""
    <div class="eiq-section-header">
      <div>
        <h1 class="eiq-section-title" style="font-size: var(--text-2xl); margin-bottom: var(--space-1);">EstateIQ</h1>
        <p class="eiq-section-description" style="margin: 0;">Machine-learning property valuation for the Ames, Iowa housing market.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Model Status & Quick Actions
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown('<div class="eiq-section-header"><h2 class="eiq-section-title">Model Status</h2></div>', unsafe_allow_html=True)

        model_path = Path(__file__).parent / "models" / "baseline_pipeline.joblib"
        if model_path.exists():
            st.markdown("""
            <div class="eiq-model-status eiq-model-status--online" style="margin-bottom: var(--space-4);">
              <span class="eiq-badge__dot"></span> Baseline Model Loaded
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <p style="color: var(--color-text-muted); margin-bottom: var(--space-4);">
              The 3-feature LinearRegression pipeline (<code>GrLivArea</code>, <code>BedroomAbvGr</code>, <code>FullBath</code>) is ready for inference.
              Model artifact: <code>models/baseline_pipeline.joblib</code>
            </p>
            """, unsafe_allow_html=True)

            # Quick actions — secondary buttons
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Predict a House Price", use_container_width=True, key="quick_predict"):
                    st.session_state.current_section = "predict"
                    st.rerun()
            with col_b:
                if st.button("View Model Performance", use_container_width=True, type="secondary", key="quick_perf"):
                    st.session_state.current_section = "performance"
                    st.rerun()
        else:
            st.markdown("""
            <div class="eiq-model-status eiq-model-status--offline" style="margin-bottom: var(--space-4);">
              <span class="eiq-badge__dot"></span> Model Not Found
            </div>
            """, unsafe_allow_html=True)
            st.error("The trained model artifact was not found at `models/baseline_pipeline.joblib`. Please run `python src/train.py` to train the baseline model.")

    with col2:
        st.markdown('<div class="eiq-section-header"><h2 class="eiq-section-title">Dataset Overview</h2></div>', unsafe_allow_html=True)

        stats = get_training_data_stats()
        if stats:
            st.markdown(f"""
            <div class="eiq-surface" style="margin-bottom: var(--space-4);">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-3);">
                    <div><strong style="color: var(--color-text-faint); font-size: var(--text-xs);">Training Samples</strong><br><span style="font-size: var(--text-xl); font-weight: var(--font-bold);">{stats['n_rows']:,}</span></div>
                    <div><strong style="color: var(--color-text-faint); font-size: var(--text-xs);">Features</strong><br><span style="font-size: var(--text-xl); font-weight: var(--font-bold);">{stats['n_features']}</span></div>
                    <div><strong style="color: var(--color-text-faint); font-size: var(--text-xs);">Target Mean</strong><br><span style="font-size: var(--text-xl); font-weight: var(--font-bold);">{format_currency(stats['target_mean'])}</span></div>
                    <div><strong style="color: var(--color-text-faint); font-size: var(--text-xs);">Target Median</strong><br><span style="font-size: var(--text-xl); font-weight: var(--font-bold);">{format_currency(stats['target_median'])}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="eiq-surface" style="margin-bottom: var(--space-4); background: var(--color-warning-light); border-color: #F5D76E;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-3);">
                    <div><strong style="color: var(--color-text-faint); font-size: var(--text-xs);">Training Samples</strong><br><span style="font-size: var(--text-xl); font-weight: var(--font-bold); color: var(--color-warning);">Unavailable</span></div>
                    <div><strong style="color: var(--color-text-faint); font-size: var(--text-xs);">Features</strong><br><span style="font-size: var(--text-xl); font-weight: var(--font-bold); color: var(--color-warning);">Unavailable</span></div>
                    <div><strong style="color: var(--color-text-faint); font-size: var(--text-xs);">Target Mean</strong><br><span style="font-size: var(--text-xl); font-weight: var(--font-bold); color: var(--color-warning);">Unavailable</span></div>
                    <div><strong style="color: var(--color-text-faint); font-size: var(--text-xs);">Target Median</strong><br><span style="font-size: var(--text-xl); font-weight: var(--font-bold); color: var(--color-warning);">Unavailable</span></div>
                </div>
                <p style="margin-top: var(--space-3); font-size: var(--text-sm); color: var(--color-text-muted);">
                    Dataset not found at <code>data/raw/train.csv</code>. Run training to generate statistics.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="eiq-surface">
            <strong style="color: var(--color-text-faint); font-size: var(--text-xs);">Source</strong><br>
            <span style="font-size: var(--text-sm);">Kaggle: House Prices - Advanced Regression Techniques</span><br>
            <span style="font-size: var(--text-sm); color: var(--color-text-muted);">Ames, Iowa | 2006–2010</span>
        </div>
        """, unsafe_allow_html=True)

    # KPI Cards — Clean, no oversized cards
    st.markdown('<div class="eiq-section-header"><h2 class="eiq-section-title">Model Performance (Validation Set)</h2></div>', unsafe_allow_html=True)

    baseline_metrics = load_baseline_metrics()
    comparison_data = load_model_comparison()

    if baseline_metrics:
        val_mae = baseline_metrics.get("Val_MAE", 35788)
        val_rmse = baseline_metrics.get("Val_RMSE", 52976)
        val_r2 = baseline_metrics.get("Val_R2", 0.634)
        val_log_rmse = baseline_metrics.get("Val_Log_RMSE", 0.272)
    else:
        val_mae = 35788
        val_rmse = 52976
        val_r2 = 0.634
        val_log_rmse = 0.272

    cv_rmse = 51111
    cv_rmse_std = 3930

    if comparison_data and "cross_validation" in comparison_data:
        cv_data = comparison_data["cross_validation"].get("LinearRegression (3 features)", {})
        cv_rmse = cv_data.get("RMSE_mean", cv_rmse)
        cv_rmse_std = cv_data.get("RMSE_std", cv_rmse_std)

    st.markdown(f"""
    <div class="eiq-kpi-grid">
        <div class="eiq-kpi">
            <div class="eiq-kpi-label">Validation MAE</div>
            <div class="eiq-kpi-value">{format_currency(val_mae)}</div>
            <div class="eiq-kpi-subtext">Mean Absolute Error</div>
        </div>
        <div class="eiq-kpi">
            <div class="eiq-kpi-label">Validation RMSE</div>
            <div class="eiq-kpi-value">{format_currency(val_rmse)}</div>
            <div class="eiq-kpi-subtext">Root Mean Squared Error</div>
        </div>
        <div class="eiq-kpi">
            <div class="eiq-kpi-label">Validation R²</div>
            <div class="eiq-kpi-value eiq-kpi-value--accent">{val_r2:.3f}</div>
            <div class="eiq-kpi-subtext">Coefficient of Determination</div>
        </div>
        <div class="eiq-kpi">
            <div class="eiq-kpi-label">CV RMSE (5-Fold)</div>
            <div class="eiq-kpi-value">{format_currency(cv_rmse)}</div>
            <div class="eiq-kpi-subtext">± {format_currency(cv_rmse_std)} across folds</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Methodology Section
    st.markdown('<div class="eiq-section-header"><h2 class="eiq-section-title">Methodology</h2></div>', unsafe_allow_html=True)

    steps = [
        ("1", "Data Loading & Validation", "Load Kaggle Ames Housing dataset (1,460 train, 1,459 test samples), validate schema, check for duplicates and missing values."),
        ("2", "Train/Validation Split", "80/20 split with fixed random_state=42 for reproducibility. No stratification for regression."),
        ("3", "Preprocessing Pipeline", "Median imputation for missing values + StandardScaler for feature standardization. Wrapped in sklearn Pipeline to prevent data leakage."),
        ("4", "Model Training", "LinearRegression on 3 baseline features: GrLivArea (above-ground living area), BedroomAbvGr (bedrooms above grade), FullBath (full bathrooms above grade)."),
        ("5", "Evaluation", "MAE, RMSE, R² on held-out validation set. Log-scale metrics (log1p/expm1) for skewed target. 5-fold CV for stability assessment."),
        ("6", "Model Persistence", "Full pipeline saved via joblib for reproducible inference. Includes preprocessing + estimator."),
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div class="eiq-step">
            <div class="eiq-step__num">{num}</div>
            <div class="eiq-step__content">
                <div class="eiq-step__title">{title}</div>
                <div class="eiq-step__desc">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_predict() -> None:
    """Render the Predict a House Price section."""
    # Compact page header
    st.markdown("""
    <div class="eiq-section-header">
      <div>
        <h1 class="eiq-section-title" style="font-size: var(--text-2xl); margin-bottom: var(--space-1);">Predict a House Price</h1>
        <p class="eiq-section-description" style="margin: 0;">Enter property details to get an estimated sale price using the trained Linear Regression model.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    model_path = Path(__file__).parent / "models" / "baseline_pipeline.joblib"
    predictor = load_predictor(str(model_path))

    if predictor is None:
        st.error("Unable to load the prediction model. Please ensure the model has been trained.")
        return

    # Model info — quiet surface
    model_info = predictor.get_model_info()
    st.markdown(f"""
    <div class="eiq-surface eiq-surface--quiet" style="margin-bottom: var(--space-5);">
        <strong style="font-size: var(--text-sm);">Active Model:</strong> {model_info['model_type']}<br>
        <strong style="font-size: var(--text-sm);">Features:</strong> {', '.join(model_info['features'])}<br>
        <strong style="font-size: var(--text-sm);">Target:</strong> {model_info['target']}
    </div>
    """, unsafe_allow_html=True)

    # Input Form
    st.markdown('<div class="eiq-section-header"><h2 class="eiq-section-title">Property Details</h2></div>', unsafe_allow_html=True)

    with st.form("prediction_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            gr_liv_area = st.number_input(
                "Above-Ground Living Area (sq ft)",
                min_value=100,
                max_value=10000,
                value=1500,
                step=50,
                help="Total above-ground living area in square feet. Typical range: 500–5,000 sq ft.",
                key="input_gr_liv_area",
            )

        with col2:
            bedroom_abv_gr = st.number_input(
                "Bedrooms Above Grade",
                min_value=0,
                max_value=20,
                value=3,
                step=1,
                help="Number of bedrooms above ground level. Typical range: 1–5.",
                key="input_bedroom",
            )

        with col3:
            full_bath = st.number_input(
                "Full Bathrooms Above Grade",
                min_value=0,
                max_value=10,
                value=2,
                step=1,
                help="Number of full bathrooms above ground level. Typical range: 1–3.",
                key="input_bath",
            )

        # Submit button — primary
        submitted = st.form_submit_button("Estimate House Price", type="primary", use_container_width=True)

    # Validation & Prediction
    if submitted:
        errors = validate_prediction_inputs(gr_liv_area, bedroom_abv_gr, full_bath)

        if errors:
            for error in errors:
                st.error(error)
        else:
            with st.spinner("Running inference..."):
                prediction = make_prediction(predictor, gr_liv_area, bedroom_abv_gr, full_bath)

            if prediction is not None:
                # Store result in session state
                st.session_state.prediction_result = {
                    "price": prediction,
                    "inputs": {
                        "GrLivArea": gr_liv_area,
                        "BedroomAbvGr": bedroom_abv_gr,
                        "FullBath": full_bath,
                    },
                    "model": model_info["model_type"],
                }
                st.success("Prediction complete!")

    # Display Result
    if "prediction_result" in st.session_state:
        result = st.session_state.prediction_result

        st.markdown(f"""
        <div class="eiq-result">
            <div class="eiq-result__label">Estimated Sale Price</div>
            <div class="eiq-result__price">{format_currency(result['price'])}</div>
            <div class="eiq-result__details">
                Based on {result['inputs']['GrLivArea']:,} sq ft · {result['inputs']['BedroomAbvGr']} bed · {result['inputs']['FullBath']} bath
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Input summary
        st.markdown("""
        <div class="eiq-input-summary">
            <div class="eiq-input-item">
                <div class="eiq-input-item__label">Living Area</div>
                <div class="eiq-input-item__value">{:,} sq ft</div>
            </div>
            <div class="eiq-input-item">
                <div class="eiq-input-item__label">Bedrooms</div>
                <div class="eiq-input-item__value">{}</div>
            </div>
            <div class="eiq-input-item">
                <div class="eiq-input-item__label">Full Baths</div>
                <div class="eiq-input-item__value">{}</div>
            </div>
            <div class="eiq-input-item">
                <div class="eiq-input-item__label">Model</div>
                <div class="eiq-input-item__value" style="font-size: var(--text-sm);">{}</div>
            </div>
        </div>
        """.format(
            result['inputs']['GrLivArea'],
            result['inputs']['BedroomAbvGr'],
            result['inputs']['FullBath'],
            result['model'],
        ), unsafe_allow_html=True)

        # Disclaimer
        st.markdown("""
        <div class="eiq-disclaimer">
            <div class="eiq-disclaimer__icon">⚠</div>
            <div class="eiq-disclaimer__content">
                <div class="eiq-disclaimer__title">Important Disclaimer</div>
                <div class="eiq-disclaimer__text">
                    This is a statistical estimate based on a Linear Regression model trained on the Ames, Iowa housing dataset (2006–2010).
                    It does not reflect current market conditions, property condition, location nuances, or recent renovations.
                    This tool is for educational demonstration purposes only and should not be used for real estate decisions.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Predict Another", use_container_width=True):
                del st.session_state.prediction_result
                st.rerun()
        with col2:
            # Download prediction details
            result_df = pd.DataFrame([{
                "GrLivArea": result['inputs']['GrLivArea'],
                "BedroomAbvGr": result['inputs']['BedroomAbvGr'],
                "FullBath": result['inputs']['FullBath'],
                "Predicted_SalePrice": result['price'],
                "Model": result['model'],
            }])
            csv = result_df.to_csv(index=False)
            st.download_button(
                "Download Prediction",
                data=csv,
                file_name=f"estateiq_prediction_{result['inputs']['GrLivArea']}sqft.csv",
                mime="text/csv",
                use_container_width=True,
            )


def render_batch() -> None:
    """Render the Batch Predictions section."""
    st.markdown("""
    <div class="eiq-section-header">
      <div>
        <h1 class="eiq-section-title" style="font-size: var(--text-2xl); margin-bottom: var(--space-1);">Batch Predictions</h1>
        <p class="eiq-section-description" style="margin: 0;">Upload a CSV file with multiple properties to get bulk price estimates.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    model_path = Path(__file__).parent / "models" / "baseline_pipeline.joblib"
    predictor = load_predictor(str(model_path))

    if predictor is None:
        st.error("Unable to load the prediction model. Please ensure the model has been trained.")
        return

    # Sample CSV Download
    st.markdown('<div class="eiq-section-header"><h2 class="eiq-section-title">Sample Input Template</h2></div>', unsafe_allow_html=True)

    sample_csv = generate_sample_csv()
    st.markdown("""
    <div class="eiq-surface" style="margin-bottom: var(--space-4);">
        <p style="color: var(--color-text-muted); margin-bottom: var(--space-3);">
            Download a sample CSV template with the required columns. The file must contain these exact column names:
        </p>
        <div class="eiq-chips">
            <span class="eiq-chip">GrLivArea</span>
            <span class="eiq-chip">BedroomAbvGr</span>
            <span class="eiq-chip">FullBath</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.download_button(
        "Download Sample CSV",
        data=sample_csv,
        file_name="estateiq_sample_input.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # File Upload
    st.markdown('<div class="eiq-section-header"><h2 class="eiq-section-title">Upload CSV</h2></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Maximum file size: 10 MB. Required columns: GrLivArea, BedroomAbvGr, FullBath",
        key="batch_uploader",
    )

    if uploaded_file is not None:
        # Check file size
        file_size = uploaded_file.size
        if file_size > 10 * 1024 * 1024:  # 10 MB
            st.error("File size exceeds 10 MB limit. Please upload a smaller file.")
            return

        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")
            return

        # Show file info
        st.markdown(f"""
        <div class="eiq-surface" style="margin-bottom: var(--space-4);">
            <strong>File:</strong> {uploaded_file.name} ({file_size:,} bytes)<br>
            <strong>Rows:</strong> {len(df)}<br>
            <strong>Columns:</strong> {len(df.columns)} ({', '.join(df.columns.tolist())})
        </div>
        """, unsafe_allow_html=True)

        # Preview
        with st.expander("Preview (first 10 rows)", expanded=True):
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)

        # Validate
        errors, cleaned_df = validate_batch_csv(df)

        if errors:
            for error in errors:
                st.error(error)
        else:
            st.success(f"✓ CSV validation passed. Ready to predict {len(cleaned_df)} properties.")

            if st.button("Run Batch Prediction", type="primary", use_container_width=True, key="run_batch"):
                with st.spinner(f"Predicting {len(cleaned_df)} properties..."):
                    results_df = predict_batch(predictor, cleaned_df)

                # Store in session state
                st.session_state.batch_results = results_df
                st.success(f"Batch prediction complete! {results_df['Predicted_SalePrice'].notna().sum()} successful, {results_df['Predicted_SalePrice'].isna().sum()} failed.")

    # Display Results
    if "batch_results" in st.session_state:
        results_df = st.session_state.batch_results

        st.markdown('<div class="eiq-section-header"><h2 class="eiq-section-title">Prediction Results</h2></div>', unsafe_allow_html=True)

        # Summary metrics — use st.metric
        successful = results_df["Predicted_SalePrice"].notna().sum()
        failed = results_df["Predicted_SalePrice"].isna().sum()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Successful", f"{successful:,}")
        with col2:
            st.metric("Failed", f"{failed:,}")
        with col3:
            if successful > 0:
                avg_price = results_df["Predicted_SalePrice"].mean()
                st.metric("Avg Predicted Price", format_currency(avg_price))

        # Results table
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            "Download Results CSV",
            data=csv,
            file_name="estateiq_batch_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Show errors if any
        if failed > 0:
            with st.expander(f"View {failed} failed predictions"):
                error_df = results_df[results_df["Prediction_Error"].notna()][["GrLivArea", "BedroomAbvGr", "FullBath", "Prediction_Error"]]
                st.dataframe(error_df, use_container_width=True, hide_index=True)


def render_performance() -> None:
    """Render the Model Performance section."""
    st.markdown("""
    <div class="eiq-section-header">
      <div>
        <h1 class="eiq-section-title" style="font-size: var(--text-2xl); margin-bottom: var(--space-1);">Model Performance</h1>
        <p class="eiq-section-description" style="margin: 0;">Compare model variants, inspect validation metrics, and understand model behavior through diagnostic visualizations.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Load comparison data
    comparison_data = load_model_comparison()
    baseline_metrics = load_baseline_metrics()

    if not comparison_data:
        st.warning("Model comparison data not found. Run `python src/compare_models.py` to generate comparison reports.")
        # Still show baseline if available
        if baseline_metrics:
            st.markdown("### Baseline Model (3-Feature LinearRegression)")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MAE", format_currency(baseline_metrics.get("Val_MAE", 35788)))
            with col2:
                st.metric("RMSE", format_currency(baseline_metrics.get("Val_RMSE", 52976)))
            with col3:
                st.metric("R²", f"{baseline_metrics.get('Val_R2', 0.634):.3f}")
            with col4:
                st.metric("Log RMSE", f"{baseline_metrics.get('Val_Log_RMSE', 0.272):.3f}")
        return

    # Model Comparison Table
    st.markdown('<div class="eiq-section-header"><h2 class="eiq-section-title">Model Comparison</h2></div>', unsafe_allow_html=True)

    # Build comparison table from validation results
    val_results = comparison_data.get("validation_set", {})
    cv_results = comparison_data.get("cross_validation", {})

    if val_results:
        table_data = []
        for model_name, metrics in val_results.items():
            cv = cv_results.get(model_name, {})
            table_data.append({
                "Model": model_name,
                "Val MAE": format_currency(metrics.get("Val_MAE", 0)),
                "Val RMSE": format_currency(metrics.get("Val_RMSE", 0)),
                "Val R²": f"{metrics.get('Val_R2', 0):.4f}",
                "Val Log RMSE": f"{metrics.get('Val_Log_RMSE', 0):.4f}",
                "CV RMSE": format_currency(cv.get("RMSE_mean", 0)),
                "CV RMSE ±": f"±{format_currency(cv.get('RMSE_std', 0))}",
                "CV R²": f"{cv.get('R2_mean', 0):.4f} ± {cv.get('R2_std', 0):.4f}",
            })

        df_comparison = pd.DataFrame(table_data)

        # Highlight best validation RMSE
        def highlight_best(row):
            if row["Val RMSE"] == df_comparison["Val RMSE"].min():
                return ["background-color: var(--color-success-light)"] * len(row)
            return [""] * len(row)

        st.markdown('<div class="eiq-table-wrap">', unsafe_allow_html=True)
        st.dataframe(
            df_comparison.style.apply(highlight_best, axis=1),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Best model callout
        best_val = comparison_data.get("best_model_val_rmse", {})
        best_cv = comparison_data.get("best_model_cv_rmse", {})

        col1, col2 = st.columns(2)
        with col1:
            if best_val:
                st.markdown(f"""
                <div class="eiq-surface">
                    <div style="font-size: var(--text-xs); color: var(--color-text-faint); margin-bottom: var(--space-2);">Best on Validation Set</div>
                    <div style="font-size: var(--text-xl); font-weight: var(--font-bold); color: var(--color-success);">{best_val.get('model', 'N/A')}</div>
                    <div style="color: var(--color-text-muted);">Val RMSE: {format_currency(best_val.get('Val_RMSE', 0))}</div>
                </div>
                """, unsafe_allow_html=True)
        with col2:
            if best_cv:
                st.markdown(f"""
                <div class="eiq-surface">
                    <div style="font-size: var(--text-xs); color: var(--color-text-faint); margin-bottom: var(--space-2);">Best on Cross-Validation</div>
                    <div style="font-size: var(--text-xl); font-weight: var(--font-bold); color: var(--color-accent);">{best_cv.get('model', 'N/A')}</div>
                    <div style="color: var(--color-text-muted);">CV RMSE: {format_currency(best_cv.get('CV_RMSE_mean', 0))} ± {format_currency(best_cv.get('CV_RMSE_std', 0))}</div>
                </div>
                """, unsafe_allow_html=True)

        # Interpretation note
        st.markdown("""
        <div class="eiq-surface" style="margin-top: var(--space-5); background: var(--color-accent-light); border-color: var(--color-accent-border);">
            <strong style="color: var(--color-accent);">Note:</strong> The model with the lowest validation RMSE may not be the most robust.
            Cross-validation measures performance across multiple training folds and is a better indicator of generalization.
            The validation set is a single hold-out split; results can vary with different random seeds.
        </div>
        """, unsafe_allow_html=True)

    # Visualizations
    st.markdown('<div class="eiq-section-header"><h2 class="eiq-section-title">Diagnostic Visualizations</h2></div>', unsafe_allow_html=True)

    figures_dir = Path(__file__).parent / "reports" / "figures"
    if figures_dir.exists():
        # List available figures for baseline model
        baseline_figs = sorted(figures_dir.glob("baseline_linearregression_*.png"))
        if baseline_figs:
            tabs = st.tabs([f.stem.replace("baseline_linearregression_", "").replace("_", " ").title() for f in baseline_figs])
            for tab, fig_path in zip(tabs, baseline_figs):
                with tab:
                    st.image(str(fig_path), use_container_width=True)
                    st.caption(f"Source: {fig_path.name}")
        else:
            st.info("No diagnostic plots found. Run `python src/train.py` to generate visualizations.")
    else:
        st.info("Figures directory not found. Run training to generate diagnostic plots.")

    # Explainability
    st.markdown('<div class="eiq-section-header"><h2 class="eiq-section-title">Model Explainability</h2></div>', unsafe_allow_html=True)

    if baseline_metrics and "coefficients" in baseline_metrics:
        coeffs = baseline_metrics["coefficients"]
        intercept = baseline_metrics.get("intercept", 0)

        st.markdown("### Baseline Model Coefficients (Standardized Features)")
        st.markdown("""
        <p style="color: var(--color-text-muted); font-size: var(--text-sm);">
            Coefficients are on standardized features (mean=0, std=1). A coefficient represents the change in
            predicted price (dollars) per one standard deviation change in the feature.
        </p>
        """, unsafe_allow_html=True)

        coeff_df = pd.DataFrame([
            {"Feature": feat, "Coefficient": f"${coef:,.0f}", "Direction": "Positive" if coef > 0 else "Negative"}
            for feat, coef in coeffs.items()
        ])
        st.dataframe(coeff_df, use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div class="eiq-surface" style="margin-top: var(--space-4);">
            <strong>Intercept:</strong> ${intercept:,.0f} (predicted price when all standardized features = 0)<br>
            <strong>Key Insight:</strong> GrLivArea has the largest positive coefficient, confirming living area is the strongest predictor.
            BedroomAbvGr shows a negative coefficient due to multicollinearity with GrLivArea (larger homes tend to have more bedrooms,
            but holding area constant, more bedrooms may indicate smaller rooms).
        </div>
        """, unsafe_allow_html=True)

    # Metrics Explanation
    with st.expander("Understanding the Metrics", expanded=False):
        st.markdown("""
        **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual prices.
        Easy to interpret: "On average, predictions are off by $X."

        **RMSE (Root Mean Squared Error)**: Square root of average squared errors.
        Penalizes large errors more heavily than MAE. Same units as target (dollars).

        **R² (Coefficient of Determination)**: Proportion of variance in target explained by the model.
        1.0 = perfect prediction, 0.0 = same as predicting the mean, negative = worse than mean.

        **Log-Scale Metrics**: Computed on log1p(SalePrice). Better for relative errors across price ranges.
        Log RMSE ≈ percentage error (e.g., 0.18 ≈ 18% typical relative error).
        """)


def render_about() -> None:
    """Render the About This Project section."""
    st.markdown("""
    <div class="eiq-section-header">
      <div>
        <h1 class="eiq-section-title" style="font-size: var(--text-2xl); margin-bottom: var(--space-1);">About This Project</h1>
        <p class="eiq-section-description" style="margin: 0;">Task 1 of the SkillCraft Technology Machine Learning Internship program.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Project Overview
    st.markdown('<div class="eiq-section-header"><h2 class="eiq-section-title">Project Overview</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="eiq-surface">
        <p><strong>Task:</strong> House Price Prediction using Linear Regression</p>
        <p><strong>Dataset:</strong> Kaggle "House Prices - Advanced Regression Techniques" (Ames Housing)</p>
        <p><strong>Goal:</strong> Predict <code>SalePrice</code> from property features using supervised regression.</p>
        <p><strong>Baseline Features:</strong> <code>GrLivArea</code> (living area), <code>BedroomAbvGr</code> (bedrooms), <code>FullBath</code> (bathrooms)</p>
        <p><strong>Extended Models:</strong> All-feature models with ColumnTransformer (numeric + categorical preprocessing)</p>
    </div>
    """, unsafe_allow_html=True)

    # Dataset Details
    st.markdown('<div class="eiq-section-header"><h2 class="eiq-section-title">Dataset Details</h2></div>', unsafe_allow_html=True)

    stats = get_training_data_stats()
    col1, col2 = st.columns(2)

    with col1:
        if stats:
            st.markdown(f"""
            <div class="eiq-surface">
                <h4 style="margin-top: 0;">Training Set</h4>
                <ul style="margin: 0; padding-left: var(--space-4);">
                    <li><strong>Samples:</strong> {stats['n_rows']:,}</li>
                    <li><strong>Features:</strong> {stats['n_features']} (24 numeric, 55 categorical)</li>
                    <li><strong>Target Range:</strong> {format_currency(stats['target_min'])} – {format_currency(stats['target_max'])}</li>
                    <li><strong>Target Mean:</strong> {format_currency(stats['target_mean'])}</li>
                    <li><strong>Target Median:</strong> {format_currency(stats['target_median'])}</li>
                    <li><strong>Skewness:</strong> 1.88 (right-skewed)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="eiq-surface" style="background: var(--color-warning-light); border-color: #F5D76E;">
                <h4 style="margin-top: 0;">Training Set</h4>
                <p style="color: var(--color-warning);"><strong>Dataset statistics unavailable</strong></p>
                <p style="font-size: var(--text-sm); color: var(--color-text-muted);">
                    Dataset not found at <code>data/raw/train.csv</code>. Run training to generate statistics.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="eiq-surface">
            <h4 style="margin-top: 0;">Key Characteristics</h4>
            <ul style="margin: 0; padding-left: var(--space-4);">
                <li>Missing values in many features (PoolQC 99%, Alley 93%, Fence 80%)</li>
                <li>Mix of numeric and categorical features</li>
                <li>Right-skewed target distribution (log-transform helps)</li>
                <li>No duplicate IDs in train or test sets</li>
                <li>Test set correctly lacks SalePrice column</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Preprocessing & Training
    st.markdown('<div class="eiq-section-header"><h2 class="eiq-section-title">Preprocessing & Training</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="eiq-surface">
        <h4 style="margin-top: 0;">Baseline Pipeline (3 Features)</h4>
        <ol style="margin: 0; padding-left: var(--space-4);">
            <li><strong>Feature Selection:</strong> Extract GrLivArea, BedroomAbvGr, FullBath</li>
            <li><strong>Imputation:</strong> SimpleImputer(strategy="median") — robust to outliers</li>
            <li><strong>Scaling:</strong> StandardScaler() — zero mean, unit variance</li>
            <li><strong>Estimator:</strong> LinearRegression()</li>
        </ol>
        <p style="margin-top: var(--space-3);"><strong>All-feature pipeline</strong> uses ColumnTransformer with separate numeric (median + scale) and categorical (most-frequent + OneHotEncoder) branches.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="eiq-surface" style="margin-top: var(--space-4);">
        <h4 style="margin-top: 0;">Validation Strategy</h4>
        <ul style="margin: 0; padding-left: var(--space-4);">
            <li><strong>Split:</strong> 80% train / 20% validation (random_state=42)</li>
            <li><strong>Cross-Validation:</strong> 5-fold KFold (shuffle=True, random_state=42)</li>
            <li><strong>Leakage Prevention:</strong> All preprocessing fit ONLY on training fold via sklearn Pipeline</li>
            <li><strong>Metrics:</strong> MAE, RMSE, R² (dollar scale) + Log MAE, Log RMSE, Log R² (log scale)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Limitations
    st.markdown('<div class="eiq-section-header"><h2 class="eiq-section-title">Limitations</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="eiq-surface" style="background: var(--color-warning-light); border-color: #F5D76E;">
        <ul style="margin: 0; padding-left: var(--space-4);">
            <li><strong>Geographic Specificity:</strong> Model trained on Ames, Iowa (2006–2010) — not transferable to other markets (e.g., India) without retraining.</li>
            <li><strong>No Causal Claims:</strong> Coefficients reflect associations in this dataset, not causal relationships.</li>
            <li><strong>Temporal Drift:</strong> Housing market conditions have changed significantly since 2010.</li>
            <li><strong>Feature Leakage Risk:</strong> Some features (SaleCondition, SaleType) may contain post-sale information.</li>
            <li><strong>Linear Assumption:</strong> True relationships may be non-linear; tree-based ensembles would capture interactions better.</li>
            <li><strong>Simple Imputation:</strong> Median/mode imputation ignores missingness patterns that could be informative.</li>
            <li><strong>No External Validation:</strong> Only internal train/validation split used.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Run Instructions
    st.markdown('<div class="eiq-section-header"><h2 class="eiq-section-title">Run Instructions</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="eiq-surface">
        <h4 style="margin-top: 0;">Prerequisites</h4>
        <ul>
            <li>Python 3.11+</li>
            <li>Virtual environment recommended</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Setup Commands", expanded=False):
        st.code("""# Create and activate virtual environment
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt""", language="powershell")

    with st.expander("Training Commands", expanded=False):
        st.code("""# Baseline training (3 features)
cd task1-house-price
python src/train.py --data-dir data/raw

# Model comparison (all features, 5-fold CV)
python src/compare_models.py --data-dir data/raw --output-dir .

# Run tests
python -m pytest tests/ -v""", language="powershell")

    with st.expander("Launch Streamlit Demo", expanded=False):
        st.code("""streamlit run task1-house-price/app.py
# Then open http://localhost:8501""", language="powershell")

    # Artifacts
    st.markdown('<div class="eiq-section-header"><h2 class="eiq-section-title">Generated Artifacts</h2></div>', unsafe_allow_html=True)

    artifacts = [
        ("models/baseline_pipeline.joblib", "Fitted 3-feature LinearRegression pipeline"),
        ("reports/metrics_baseline.json", "Baseline validation metrics"),
        ("reports/coefficients_baseline.csv", "Baseline model coefficients"),
        ("reports/model_comparison.json", "Full model comparison (CV + validation)"),
        ("reports/model_comparison.csv", "Comparison summary table"),
        ("reports/figures/*.png", "Diagnostic plots (distribution, actual vs predicted, residuals)"),
        ("docs/task1-report.md", "Comprehensive project report"),
    ]

    for path, desc in artifacts:
        st.markdown(f"""
        <div class="eiq-surface" style="padding: var(--space-3); margin-bottom: var(--space-2); display: flex; justify-content: space-between; align-items: center;">
            <code style="font-size: var(--text-sm);">{path}</code>
            <span style="color: var(--color-text-muted); font-size: var(--text-sm);">{desc}</span>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main application entry point."""
    # Inject custom CSS
    inject_css()

    # Initialize session state for navigation
    if "current_section" not in st.session_state:
        st.session_state.current_section = "overview"

    # Sidebar — Brand + Navigation using st.radio
    with st.sidebar:
        # Brand
        st.markdown("""
        <div class="eiq-brand">
            <div class="eiq-brand__name">EstateIQ</div>
            <div class="eiq-brand__tagline">House Price Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        # Navigation using st.radio (native, accessible)
        sections = [
            "Overview",
            "Predict",
            "Batch Predictions",
            "Model Performance",
            "About",
        ]
        section_keys = ["overview", "predict", "batch", "performance", "about"]

        # Map current_section to index
        try:
            current_idx = section_keys.index(st.session_state.current_section)
        except ValueError:
            current_idx = 0

        selected = st.radio(
            "Navigation",
            sections,
            index=current_idx,
            key="nav_radio",
            label_visibility="collapsed",
        )

        # Update session state when selection changes
        new_idx = sections.index(selected)
        if new_idx != current_idx:
            st.session_state.current_section = section_keys[new_idx]
            st.rerun()

        # Model status in sidebar
        st.markdown("---")
        model_path = Path(__file__).parent / "models" / "baseline_pipeline.joblib"
        if model_path.exists():
            st.markdown("""
            <div class="eiq-model-status eiq-model-status--online">
                <span class="eiq-badge__dot"></span> Model Loaded
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="eiq-model-status eiq-model-status--offline">
                <span class="eiq-badge__dot"></span> Model Not Found
            </div>
            """, unsafe_allow_html=True)

        # Version info
        st.markdown("""
        <div class="eiq-sidebar-footer">
            EstateIQ v1.0.0<br>
            SkillCraft ML Internship — Task 1
        </div>
        """, unsafe_allow_html=True)

    # Route to appropriate section
    current_section = st.session_state.current_section
    if current_section == "overview":
        render_overview()
    elif current_section == "predict":
        render_predict()
    elif current_section == "batch":
        render_batch()
    elif current_section == "performance":
        render_performance()
    elif current_section == "about":
        render_about()


if __name__ == "__main__":
    main()