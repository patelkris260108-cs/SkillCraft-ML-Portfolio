"""
Model Training and Selection Pipeline for Task 4: Hand Gesture Recognition.

Trains and evaluates lightweight, resource-conscious classical ML models:
- DummyClassifier (baseline: most frequent class)
- LinearSVC (C=1.0 and C=0.1, dual=False, primal formulation)
- SGDClassifier (loss='log_loss', Logistic Regression via SGD)
- SGDClassifier (loss='hinge', Linear SVM via SGD)

Workflow:
1. Load HOG feature matrices: X_train (14,000), X_val (2,000), X_test (4,000)
2. Train all candidate pipelines strictly on training data
3. Select best model based on Validation Macro F1 (zero test data leakage)
4. Evaluate winning model on the held-out test subjects (08 and 09)
5. Save model artifact to models/best_gesture_model.joblib and configuration to models/model_config.json
6. Generate reports/modeling_metrics.json and reports/modeling_report.md
7. Generate visualization plots: model_comparison.png and test_confusion_matrix.png
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from src.feature_extractor import CLASS_TO_IDX, GESTURE_CLASSES, IDX_TO_CLASS
except ModuleNotFoundError:
    from feature_extractor import CLASS_TO_IDX, GESTURE_CLASSES, IDX_TO_CLASS


def get_candidate_models(random_state: int = 42) -> Dict[str, Pipeline]:
    """Define lightweight candidate models suitable for 8 GB RAM laptop."""
    return {
        "Dummy_Baseline": Pipeline([
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "Linear_SVM_C1.0": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LinearSVC(C=1.0, dual=False, random_state=random_state, max_iter=2000)),
        ]),
        "Linear_SVM_C0.1": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LinearSVC(C=0.1, dual=False, random_state=random_state, max_iter=2000)),
        ]),
        "SGD_LogisticRegression": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, random_state=random_state, max_iter=1000)),
        ]),
        "SGD_LinearSVM": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", SGDClassifier(loss="hinge", penalty="l2", alpha=1e-4, random_state=random_state, max_iter=1000)),
        ]),
    }


class GestureModelTrainer:
    """
    Train, select, and evaluate hand gesture recognition models.
    """

    def __init__(
        self,
        processed_data_dir: str | Path,
        models_dir: str | Path,
        reports_dir: str | Path,
        random_state: int = 42,
    ):
        self.data_dir = Path(processed_data_dir)
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.figures_dir = self.reports_dir / "figures"
        self.random_state = random_state

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load X and y arrays for train, val, and test splits."""
        fx_train = self.data_dir / "X_train.npy"
        fy_train = self.data_dir / "y_train.npy"
        fx_val = self.data_dir / "X_val.npy"
        fy_val = self.data_dir / "y_val.npy"
        fx_test = self.data_dir / "X_test.npy"
        fy_test = self.data_dir / "y_test.npy"

        for p in [fx_train, fy_train, fx_val, fy_val, fx_test, fy_test]:
            if not p.exists():
                raise FileNotFoundError(f"Processed feature file not found: {p}")

        X_train = np.load(fx_train)
        y_train = np.load(fy_train)
        X_val = np.load(fx_val)
        y_val = np.load(fy_val)
        X_test = np.load(fx_test)
        y_test = np.load(fy_test)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model: Optional[Any] = None,
        X_eval: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute comprehensive multiclass classification metrics."""
        acc = float(accuracy_score(y_true, y_pred))
        macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(GESTURE_CLASSES)))).tolist()
        per_class_report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(GESTURE_CLASSES))),
            target_names=GESTURE_CLASSES,
            output_dict=True,
            zero_division=0,
        )

        roc_auc_val: Optional[float] = None
        if model is not None and X_eval is not None:
            try:
                if hasattr(model, "predict_proba"):
                    scores = model.predict_proba(X_eval)
                    roc_auc_val = float(roc_auc_score(y_true, scores, multi_class="ovr", average="macro"))
                elif hasattr(model, "decision_function"):
                    scores = model.decision_function(X_eval)
                    roc_auc_val = float(roc_auc_score(y_true, scores, multi_class="ovr", average="macro"))
            except Exception:
                roc_auc_val = None

        return {
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "roc_auc_ovr_macro": round(roc_auc_val, 4) if roc_auc_val is not None else None,
            "confusion_matrix": cm,
            "per_class": per_class_report,
        }

    def train_and_select(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[str, Pipeline, Dict[str, Dict[str, Any]]]:
        """
        Train candidate models on X_train, evaluate on X_val, and select winner by validation macro F1.
        Zero test data is touched during this selection phase.
        """
        candidates = get_candidate_models(self.random_state)
        val_evaluations: Dict[str, Dict[str, Any]] = {}
        fitted_pipelines: Dict[str, Pipeline] = {}

        print("=" * 60)
        print("Training and Evaluating Candidate Models on Validation Split (Subject 07)")
        print("=" * 60)

        for name, pipe in candidates.items():
            t0 = time.time()
            print(f"Training '{name}' on {len(y_train):,} samples...")
            pipe.fit(X_train, y_train)
            train_time = time.time() - t0

            val_preds = pipe.predict(X_val)
            metrics = self.compute_metrics(y_val, val_preds, model=pipe, X_eval=X_val)
            metrics["train_time_sec"] = round(train_time, 2)

            val_evaluations[name] = metrics
            fitted_pipelines[name] = pipe

            print(f"  -> Val Acc: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | Time: {train_time:.2f}s")

        # Select best model by validation macro F1
        best_model_name = max(val_evaluations.keys(), key=lambda k: val_evaluations[k]["macro_f1"])
        best_pipeline = fitted_pipelines[best_model_name]
        print(f"\n[WINNER] Selected Best Model: '{best_model_name}' (Val Macro F1 = {val_evaluations[best_model_name]['macro_f1']:.4f})")

        return best_model_name, best_pipeline, val_evaluations

    def evaluate_test(
        self,
        best_pipeline: Pipeline,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate selected winning model ONCE on the held-out test split (Subjects 08 and 09).
        """
        print("\n" + "=" * 60)
        print("Evaluating Selected Model on Held-Out Test Split (Subjects 08 and 09)")
        print("=" * 60)
        t0 = time.time()
        test_preds = best_pipeline.predict(X_test)
        test_metrics = self.compute_metrics(y_test, test_preds, model=best_pipeline, X_eval=X_test)
        test_metrics["eval_time_sec"] = round(time.time() - t0, 2)

        print(f"  Held-Out Test Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Held-Out Test Macro F1:  {test_metrics['macro_f1']:.4f}")
        print(f"  Held-Out Test Precision: {test_metrics['macro_precision']:.4f}")
        print(f"  Held-Out Test Recall:    {test_metrics['macro_recall']:.4f}")
        if test_metrics["roc_auc_ovr_macro"] is not None:
            print(f"  Held-Out Test ROC-AUC:   {test_metrics['roc_auc_ovr_macro']:.4f}")

        return test_metrics

    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------

    def plot_model_comparison(
        self,
        val_evaluations: Dict[str, Dict[str, Any]],
        save_path: Optional[Path] = None,
    ) -> Path:
        """Generate comparison bar chart of candidate models on validation split."""
        models = list(val_evaluations.keys())
        accs = [val_evaluations[m]["accuracy"] * 100 for m in models]
        f1s = [val_evaluations[m]["macro_f1"] * 100 for m in models]

        x = np.arange(len(models))
        width = 0.35

        plt.figure(figsize=(10, 5), dpi=150)
        bars1 = plt.bar(x - width / 2, accs, width=width, label="Val Accuracy (%)", color="#1f77b4", edgecolor="#114b73")
        bars2 = plt.bar(x + width / 2, f1s, width=width, label="Val Macro F1 (%)", color="#ff7f0e", edgecolor="#994d08")

        for bar in bars1:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

        for bar in bars2:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

        plt.title("Task 4: Candidate Model Comparison on Validation Set (Subject 07)", fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("Model Pipeline", fontsize=10)
        plt.ylabel("Performance (%)", fontsize=10)
        plt.xticks(x, [m.replace("_", "\n") for m in models], fontsize=9)
        plt.ylim(0, 115)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend(loc="upper left", fontsize=9)
        plt.tight_layout()

        out_path = save_path or (self.figures_dir / "model_comparison.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        return out_path

    def plot_confusion_matrix(
        self,
        cm: List[List[int]],
        save_path: Optional[Path] = None,
    ) -> Path:
        """Generate normalized confusion matrix heatmap for test set."""
        cm_arr = np.array(cm, dtype=np.float64)
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        norm_cm = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums != 0) * 100

        plt.figure(figsize=(10, 8), dpi=150)
        plt.imshow(norm_cm, cmap="Blues", vmin=0, vmax=100)
        plt.colorbar(label="Accuracy per class (%)")

        plt.title("Task 4: Held-Out Test Confusion Matrix (% Recall by Class)", fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("Predicted Gesture Class", fontsize=10)
        plt.ylabel("True Gesture Class", fontsize=10)

        short_labels = [g.replace("_", "\n") for g in GESTURE_CLASSES]
        plt.xticks(np.arange(len(GESTURE_CLASSES)), short_labels, fontsize=8)
        plt.yticks(np.arange(len(GESTURE_CLASSES)), [g.replace("_", " ") for g in GESTURE_CLASSES], fontsize=8)

        # Annotate percentages
        for i in range(len(GESTURE_CLASSES)):
            for j in range(len(GESTURE_CLASSES)):
                val = norm_cm[i, j]
                count = int(cm_arr[i, j])
                plt.text(
                    j,
                    i,
                    f"{val:.0f}%\n({count})",
                    ha="center",
                    va="center",
                    color="white" if val > 50 else "black",
                    fontsize=7,
                    fontweight="bold" if i == j else "normal",
                )

        plt.tight_layout()
        out_path = save_path or (self.figures_dir / "test_confusion_matrix.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        return out_path

    # -------------------------------------------------------------------------
    # Reports & Serialization
    # -------------------------------------------------------------------------

    def save_model_artifacts(
        self,
        best_name: str,
        best_pipeline: Pipeline,
        feature_config: Dict[str, Any],
    ) -> Path:
        """Serialize best model pipeline and configuration JSON."""
        model_file = self.models_dir / "best_gesture_model.joblib"
        joblib.dump(best_pipeline, model_file)

        config_payload = {
            "selected_model": best_name,
            "classes": GESTURE_CLASSES,
            "class_to_idx": CLASS_TO_IDX,
            "idx_to_class": {str(k): v for k, v in IDX_TO_CLASS.items()},
            "feature_config": feature_config,
            "pipeline_steps": [str(s) for s in best_pipeline.named_steps.keys()],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        config_file = self.models_dir / "model_config.json"
        with open(config_file, mode="w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2)

        return model_file

    def save_metrics_and_report(
        self,
        best_name: str,
        val_evaluations: Dict[str, Dict[str, Any]],
        test_metrics: Dict[str, Any],
    ) -> Tuple[Path, Path]:
        """Save machine-readable JSON metrics and Markdown modeling report."""
        # Machine-readable JSON
        json_payload = {
            "best_model_name": best_name,
            "candidate_models": list(val_evaluations.keys()),
            "validation_results": val_evaluations,
            "test_results": test_metrics,
            "split_info": {
                "train_subjects": ["00", "01", "02", "03", "04", "05", "06"],
                "val_subjects": ["07"],
                "test_subjects": ["08", "09"],
                "train_samples": 14000,
                "val_samples": 2000,
                "test_samples": 4000,
            },
        }

        metrics_file = self.reports_dir / "modeling_metrics.json"
        with open(metrics_file, mode="w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2)

        # Markdown Report
        val_rows = []
        for m, met in val_evaluations.items():
            val_rows.append(
                f"| `{m}` | {met['accuracy'] * 100:.2f}% | {met['macro_precision'] * 100:.2f}% | "
                f"{met['macro_recall'] * 100:.2f}% | **{met['macro_f1'] * 100:.2f}%** | {met['train_time_sec']:.2f}s |"
            )
        val_table = "\n".join(val_rows)

        per_class = test_metrics["per_class"]
        per_class_rows = []
        for g in GESTURE_CLASSES:
            if g in per_class:
                c_data = per_class[g]
                per_class_rows.append(
                    f"| `{g}` | {c_data['precision'] * 100:.1f}% | {c_data['recall'] * 100:.1f}% | {c_data['f1-score'] * 100:.1f}% | {c_data['support']:,} |"
                )
        per_class_table = "\n".join(per_class_rows)

        report_content = f"""# Task 4 — Machine Learning Modeling & Generalization Report

**Project:** Hand Gesture Recognition (LeapGestRecog NIR Dataset)  
**Selected Model Pipeline:** `{best_name}`  
**Evaluation Paradigm:** Strict Subject-Independent Generalization (Zero Participant Overlap)  

---

## 1. Executive Summary

1. **Validation-Based Selection:**
   - 5 lightweight candidate models were trained strictly on **14,000 frames** from 7 participants (`00` through `06`).
   - Candidate performance was evaluated on **Validation Subject `07`** (2,000 frames) to select the winning architecture without test set contamination.
   - **`{best_name}`** achieved the highest Validation Macro F1 of **{val_evaluations[best_name]['macro_f1'] * 100:.2f}%** (Accuracy: **{val_evaluations[best_name]['accuracy'] * 100:.2f}%**).

2. **Held-Out Test Generalization (Subjects `08` & `09`):**
   - The selected model was evaluated once on the **4,000 held-out test frames** from completely unseen participants.
   - **Test Accuracy:** **{test_metrics['accuracy'] * 100:.2f}%**
   - **Test Macro F1:** **{test_metrics['macro_f1'] * 100:.2f}%**
   - **Test Macro Precision:** **{test_metrics['macro_precision'] * 100:.2f}%**
   - **Test Macro Recall:** **{test_metrics['macro_recall'] * 100:.2f}%**
   - **Multiclass ROC-AUC (OvR):** {test_metrics['roc_auc_ovr_macro'] if test_metrics['roc_auc_ovr_macro'] is not None else 'N/A'}

---

## 2. Candidate Model Comparison (Validation Subject `07`)

| Model Pipeline | Val Accuracy | Val Macro Prec | Val Macro Rec | Val Macro F1 | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
{val_table}

![Model Comparison](figures/model_comparison.png)

---

## 3. Held-Out Test Set Performance (Subjects `08` & `09`)

### Overall Metrics
- **Accuracy:** {test_metrics['accuracy'] * 100:.2f}%
- **Macro Precision:** {test_metrics['macro_precision'] * 100:.2f}%
- **Macro Recall:** {test_metrics['macro_recall'] * 100:.2f}%
- **Macro F1 Score:** {test_metrics['macro_f1'] * 100:.2f}%

### Per-Class Test Breakdown
| Gesture Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
{per_class_table}

![Held-Out Test Confusion Matrix](figures/test_confusion_matrix.png)

---

## 4. Key Modeling Insights & Limitations

1. **Near-Infrared HOG Discriminability:**
   - The classical HOG feature descriptor (2,700 dimensions) provides strong discriminative capability across distinct hand poses (e.g. Palm vs Fist vs Index vs Thumb).
   - High gradient contrast between the reflective skin tissue and dark NIR background allows linear classifiers to separate poses effectively.
2. **Dynamic Gestures Disambiguation:**
   - Pairs such as `03_fist` vs `04_fist_moved` and `01_palm` vs `08_palm_moved` represent identical hand geometries that differ primarily in temporal motion trajectories. Static 2D spatial features can confuse them without optical flow or temporal sequence modeling.
3. **Participant Sample Limitations:**
   - While the dataset contains 20,000 frames, it originates from **10 distinct human subjects**.
   - Validation contains only **1 subject (`07`)**, and the test partition contains **2 subjects (`08` and `09`)**.
   - Frame-level sample counts ($N=4,000$) do not represent 4,000 independent statistical degrees of freedom; performance heavily reflects subject hand morphology, sensor positioning, and individual articulation styles.
"""

        report_file = self.reports_dir / "modeling_report.md"
        with open(report_file, mode="w", encoding="utf-8") as f:
            f.write(report_content)

        return metrics_file, report_file

    def run_full_training_pipeline(self) -> Dict[str, Any]:
        """Execute end-to-end training, validation selection, test evaluation, and report export."""
        print("1. Loading processed HOG feature arrays...")
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data()
        print(f"  Train: X={X_train.shape}, y={y_train.shape}")
        print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
        print(f"  Test:  X={X_test.shape}, y={y_test.shape}\n")

        print("2. Training and selecting best model on Validation split...")
        best_name, best_pipe, val_evals = self.train_and_select(X_train, y_train, X_val, y_val)

        print("3. Evaluating winning model on Held-Out Test split...")
        test_metrics = self.evaluate_test(best_pipe, X_test, y_test)

        print("4. Generating diagnostic figures...")
        self.plot_model_comparison(val_evals)
        self.plot_confusion_matrix(test_metrics["confusion_matrix"])

        # Load feature config for saving
        config_path = self.data_dir / "feature_config.json"
        feat_config = {}
        if config_path.exists():
            with open(config_path, mode="r", encoding="utf-8") as f:
                feat_config = json.load(f)

        print("5. Serializing best model and metadata...")
        model_file = self.save_model_artifacts(best_name, best_pipe, feat_config)

        print("6. Writing modeling metrics and report...")
        metrics_file, report_file = self.save_metrics_and_report(best_name, val_evals, test_metrics)

        return {
            "best_model_name": best_name,
            "val_macro_f1": val_evals[best_name]["macro_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "model_file": str(model_file),
            "metrics_file": str(metrics_file),
            "report_file": str(report_file),
        }


def main() -> int:
    """CLI runner for Task 4 Model Training."""
    if Path.cwd().name == "task4-hand-gesture-recognition":
        processed_data_dir = Path("data/processed")
        models_dir = Path("models")
        reports_dir = Path("reports")
    else:
        processed_data_dir = Path("task4-hand-gesture-recognition/data/processed")
        models_dir = Path("task4-hand-gesture-recognition/models")
        reports_dir = Path("task4-hand-gesture-recognition/reports")

    print("=" * 70)
    print("SkillCraft ML Portfolio — Task 4: Model Training and Selection")
    print("=" * 70)
    print(f"Processed Data: {processed_data_dir}")
    print(f"Models Dir:     {models_dir}")
    print(f"Reports Dir:    {reports_dir}\n")

    trainer = GestureModelTrainer(processed_data_dir, models_dir, reports_dir)
    results = trainer.run_full_training_pipeline()

    print("\n" + "=" * 70)
    print("MODEL TRAINING & SELECTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"  Best Model:         {results['best_model_name']}")
    print(f"  Validation Macro F1:{results['val_macro_f1'] * 100:.2f}%")
    print(f"  Held-Out Test Acc:  {results['test_accuracy'] * 100:.2f}%")
    print(f"  Held-Out Test F1:   {results['test_macro_f1'] * 100:.2f}%")
    print(f"  Model File:         {results['model_file']}")
    print(f"  Report File:        {results['report_file']}")
    print(f"  Metrics File:       {results['metrics_file']}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
