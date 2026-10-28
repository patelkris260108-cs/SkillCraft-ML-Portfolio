"""
Unit tests for the ModelTrainer module (Task 3).
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.feature_extractor import HOGConfig
from src.model_trainer import (
    EvaluationMetrics,
    ModelTrainer,
    compute_metrics,
)


def test_compute_metrics_known_values():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 0])
    y_scores = np.array([0.1, 0.2, 0.8, 0.4])

    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_scores=y_scores,
        model_name="TestModel",
        split="val",
    )

    assert metrics.model_name == "TestModel"
    assert metrics.split == "val"
    assert metrics.accuracy == 0.75
    assert metrics.confusion_matrix == [[2, 0], [1, 1]]
    assert metrics.recall_cat == 1.0
    assert metrics.recall_dog == 0.5
    assert metrics.roc_auc is not None


def test_train_and_select_synthetic_data():
    np.random.seed(42)
    # 80 train samples, 20 val samples, 20 features
    X_train = np.random.randn(80, 20).astype(np.float32)
    y_train = (X_train[:, 0] > 0).astype(np.int64)

    X_val = np.random.randn(20, 20).astype(np.float32)
    y_val = (X_val[:, 0] > 0).astype(np.int64)

    trainer = ModelTrainer(random_state=42)
    best_name, best_pipe, val_metrics = trainer.train_and_select(X_train, y_train, X_val, y_val)

    assert best_name in trainer.get_candidate_models()
    assert isinstance(best_pipe, Pipeline)
    assert len(val_metrics) == len(trainer.get_candidate_models())

    # Check that baseline dummy is present
    assert "Baseline Dummy (Stratified)" in val_metrics

    # Check best model has highest F1
    best_f1 = val_metrics[best_name].f1_macro
    for m in val_metrics.values():
        assert m.f1_macro <= best_f1


def test_evaluate_test_set_once():
    np.random.seed(42)
    X_train = np.random.randn(60, 15).astype(np.float32)
    y_train = np.random.randint(0, 2, 60)

    X_test = np.random.randn(20, 15).astype(np.float32)
    y_test = np.random.randint(0, 2, 20)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(C=1.0, dual="auto", random_state=42)),
    ])
    pipe.fit(X_train, y_train)

    trainer = ModelTrainer(random_state=42)
    metrics, y_pred, y_scores = trainer.evaluate_test_set(pipe, "LinearSVC", X_test, y_test)

    assert metrics.split == "test"
    assert 0.0 <= metrics.accuracy <= 1.0
    assert len(y_pred) == 20
    assert len(y_scores) == 20
    assert len(metrics.confusion_matrix) == 2


def test_model_serialization(tmp_path: Path):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(C=0.1, dual="auto", random_state=42)),
    ])
    X = np.random.randn(30, 10).astype(np.float32)
    y = np.random.randint(0, 2, 30)
    pipe.fit(X, y)

    model_file = tmp_path / "model.joblib"
    joblib.dump(pipe, model_file)

    loaded_pipe = joblib.load(model_file)
    orig_pred = pipe.predict(X)
    loaded_pred = loaded_pipe.predict(X)

    np.testing.assert_array_equal(orig_pred, loaded_pred)


def test_figures_and_markdown_report_generation(tmp_path: Path):
    figures_dir = tmp_path / "figures"
    report_file = tmp_path / "report.md"

    m_val = {
        "Dummy": EvaluationMetrics(
            model_name="Dummy", split="val", accuracy=0.5, precision_macro=0.5,
            recall_macro=0.5, f1_macro=0.5, precision_cat=0.5, recall_cat=0.5,
            f1_cat=0.5, precision_dog=0.5, recall_dog=0.5, f1_dog=0.5,
            roc_auc=0.5, confusion_matrix=[[5, 5], [5, 5]], fit_time_seconds=0.01,
        ),
        "LinearSVM": EvaluationMetrics(
            model_name="LinearSVM", split="val", accuracy=0.8, precision_macro=0.8,
            recall_macro=0.8, f1_macro=0.8, precision_cat=0.8, recall_cat=0.8,
            f1_cat=0.8, precision_dog=0.8, recall_dog=0.8, f1_dog=0.8,
            roc_auc=0.85, confusion_matrix=[[8, 2], [2, 8]], fit_time_seconds=0.05,
        ),
    }

    m_test = EvaluationMetrics(
        model_name="LinearSVM", split="test", accuracy=0.78, precision_macro=0.78,
        recall_macro=0.78, f1_macro=0.78, precision_cat=0.78, recall_cat=0.78,
        f1_cat=0.78, precision_dog=0.78, recall_dog=0.78, f1_dog=0.78,
        roc_auc=0.84, confusion_matrix=[[78, 22], [22, 78]], fit_time_seconds=0.05,
    )

    y_test = np.array([0] * 10 + [1] * 10)
    y_pred = np.array([0] * 8 + [1] * 2 + [0] * 2 + [1] * 8)
    y_scores = np.linspace(-1, 1, 20)

    figs = ModelTrainer.generate_figures(m_val, m_test, y_test, y_pred, y_scores, figures_dir)
    assert len(figs) == 3
    for f in figs:
        assert f.exists()
        assert f.stat().st_size > 0

    report = ModelTrainer.generate_markdown_report(m_val, m_test, HOGConfig(), report_file)
    assert report_file.exists()
    assert "LinearSVM" in report
    assert "Confusion Matrix" in report
    assert "Executive Summary" in report
