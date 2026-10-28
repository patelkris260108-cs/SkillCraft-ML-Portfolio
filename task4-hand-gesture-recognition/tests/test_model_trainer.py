"""
Unit tests for GestureModelTrainer (Task 4: Hand Gesture Recognition Phase 4B).

Focuses on:
- Candidate model initialization
- Accurate metric calculation (multiclass accuracy, macro F1, confusion matrix, ROC-AUC)
- Validation-based model selection (zero test leakage)
- Model artifact serialization and reload verification
- Report and metrics JSON generation
"""

from pathlib import Path
import pytest
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

from src.model_trainer import (
    GestureModelTrainer,
    get_candidate_models,
)
from src.feature_extractor import GESTURE_CLASSES


# -------------------------------------------------------------------------
# 1. Candidate Model Pipeline Tests
# -------------------------------------------------------------------------

def test_get_candidate_models():
    """Verify all expected candidate pipelines are returned with valid steps."""
    candidates = get_candidate_models(random_state=42)
    expected_names = {
        "Dummy_Baseline",
        "Linear_SVM_C1.0",
        "Linear_SVM_C0.1",
        "SGD_LogisticRegression",
        "SGD_LinearSVM",
    }
    assert set(candidates.keys()) == expected_names
    for name, pipe in candidates.items():
        assert isinstance(pipe, Pipeline)
        assert hasattr(pipe, "fit")
        assert hasattr(pipe, "predict")


# -------------------------------------------------------------------------
# 2. Metric Computation Tests
# -------------------------------------------------------------------------

def test_compute_metrics():
    """Verify precision, recall, F1, accuracy, and confusion matrix calculation."""
    trainer = GestureModelTrainer("data/processed", "models", "reports")

    # 10 samples, 10 classes
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # 8 correct, 2 incorrect (class 1 predicted as 0, class 9 predicted as 8)
    y_pred = np.array([0, 0, 2, 3, 4, 5, 6, 7, 8, 8])

    metrics = trainer.compute_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 0.8
    assert metrics["macro_f1"] < 1.0
    assert len(metrics["confusion_matrix"]) == 10
    assert "01_palm" in metrics["per_class"]
    assert metrics["per_class"]["03_fist"]["f1-score"] == 1.0


# -------------------------------------------------------------------------
# 3. Model Training & Selection Tests (Synthetic Fixture)
# -------------------------------------------------------------------------

def test_train_and_select_mock(tmp_path: Path):
    """
    Verify training and validation selection logic runs cleanly on small synthetic fixture.
    Ensures zero test leakage during model selection.
    """
    np.random.seed(42)
    n_features = 2700
    n_train = 60
    n_val = 20

    # Synthetic separable data
    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    y_train = np.repeat(np.arange(10), n_train // 10)

    X_val = np.random.randn(n_val, n_features).astype(np.float32)
    y_val = np.repeat(np.arange(10), n_val // 10)

    trainer = GestureModelTrainer(tmp_path / "processed", tmp_path / "models", tmp_path / "reports")
    best_name, best_pipe, val_evals = trainer.train_and_select(X_train, y_train, X_val, y_val)

    assert best_name in val_evals
    assert isinstance(best_pipe, Pipeline)
    assert hasattr(best_pipe, "predict")

    # Verify best model was selected by highest validation macro F1
    best_val_f1 = val_evals[best_name]["macro_f1"]
    for name, met in val_evals.items():
        assert met["macro_f1"] <= best_val_f1


def test_model_serialization_and_reload(tmp_path: Path):
    """Verify saved model artifact can be reloaded and yields matching predictions."""
    np.random.seed(42)
    X_sample = np.random.randn(5, 2700).astype(np.float32)
    y_sample = np.array([0, 1, 2, 3, 4])

    models = get_candidate_models()
    pipe = models["Linear_SVM_C1.0"]
    pipe.fit(X_sample, y_sample)

    expected_preds = pipe.predict(X_sample)

    trainer = GestureModelTrainer(tmp_path / "processed", tmp_path / "models", tmp_path / "reports")
    model_path = trainer.save_model_artifacts("Linear_SVM_C1.0", pipe, {"feature_dim": 2700})

    assert model_path.exists()
    assert (tmp_path / "models" / "model_config.json").exists()

    # Reload model artifact
    loaded_pipe = joblib.load(model_path)
    loaded_preds = loaded_pipe.predict(X_sample)

    np.testing.assert_array_equal(expected_preds, loaded_preds)


def test_save_metrics_and_report(tmp_path: Path):
    """Verify modeling_metrics.json and modeling_report.md are generated with proper content."""
    trainer = GestureModelTrainer(tmp_path / "processed", tmp_path / "models", tmp_path / "reports")

    val_evals = {
        "Dummy_Baseline": {"accuracy": 0.1, "macro_precision": 0.01, "macro_recall": 0.1, "macro_f1": 0.018, "train_time_sec": 0.01},
        "Linear_SVM_C1.0": {"accuracy": 0.85, "macro_precision": 0.86, "macro_recall": 0.85, "macro_f1": 0.854, "train_time_sec": 1.2},
    }

    test_metrics = {
        "accuracy": 0.82,
        "macro_precision": 0.83,
        "macro_recall": 0.82,
        "macro_f1": 0.824,
        "roc_auc_ovr_macro": 0.945,
        "confusion_matrix": [[400 if i == j else 0 for j in range(10)] for i in range(10)],
        "per_class": {
            g: {"precision": 0.82, "recall": 0.82, "f1-score": 0.82, "support": 400} for g in GESTURE_CLASSES
        },
    }

    metrics_file, report_file = trainer.save_metrics_and_report("Linear_SVM_C1.0", val_evals, test_metrics)

    assert metrics_file.exists()
    assert report_file.exists()

    report_text = report_file.read_text(encoding="utf-8")
    assert "Linear_SVM_C1.0" in report_text
    assert "Held-Out Test Set Performance" in report_text
    assert "Participant Sample Limitations" in report_text


def test_actual_trained_model_artifacts():
    """Verify the real trained model, config, and metrics files if present on disk."""
    base_dir = Path(__file__).resolve().parent.parent
    model_file = base_dir / "models" / "best_gesture_model.joblib"
    config_file = base_dir / "models" / "model_config.json"
    metrics_file = base_dir / "reports" / "modeling_metrics.json"
    report_file = base_dir / "reports" / "modeling_report.md"

    if not model_file.exists():
        pytest.skip("Model artifact not yet generated.")

    assert config_file.exists()
    assert metrics_file.exists()
    assert report_file.exists()

    # Load and test model
    pipe = joblib.load(model_file)
    assert hasattr(pipe, "predict")
    sample_feat = np.zeros((1, 2700), dtype=np.float32)
    pred = pipe.predict(sample_feat)
    assert len(pred) == 1
    assert 0 <= pred[0] < 10

    # Check metrics consistency
    import json
    with open(metrics_file, mode="r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["best_model_name"] == "SGD_LinearSVM"
    assert "validation_results" in metrics
    assert "test_results" in metrics
    assert metrics["test_results"]["accuracy"] == 0.622
    assert metrics["test_results"]["macro_f1"] == 0.6125

