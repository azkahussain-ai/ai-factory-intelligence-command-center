"""Stage VIII — MLOps / MLflow.

Logs the actual, already-trained Stage II models and their real evaluation
results (from `reports/stage2_baseline_metrics.json` and
`reports/stage2_gru_metrics.json`) into MLflow. No model is retrained -
every parameter and metric logged here was either read from those existing
JSON reports or copied verbatim from the training scripts' known
hyperparameters (`src/ml/train_baseline.py`, `src/deep_learning/train_gru.py`).

A local SQLite-backed tracking store is used (rather than the plain file
store) because it is the store MLflow's Model Registry actually requires -
the brief asks to "integrate [registry] appropriately... if supported by
the installed MLflow setup", and this setup only supports it with a
database-backed URI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.pyfunc

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.config import PROJECT_ROOT, REPORTS_DIR

MODELS_DIR = PROJECT_ROOT / "models"
TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "ai_factory_stage2_failure_prediction"
REGISTERED_MODEL_NAME = "ai_factory_failure_predictor"


class GRUPyfuncWrapper(mlflow.pyfunc.PythonModel):
    """Wraps the existing hand-rolled NumPy GRU (`src/deep_learning/gru_numpy.py`)
    as an MLflow pyfunc model, purely so it can be logged as a proper
    "logged model" (with an MLmodel manifest) and registered in the Model
    Registry - `mlflow.log_artifact` alone produces a bare file with no
    model manifest, which `mlflow.register_model` cannot point at. No
    retraining, no change to the model's weights or predictions.
    """

    def load_context(self, context):
        sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
        from src.deep_learning.gru_numpy import GRUBinaryClassifier
        self.model = GRUBinaryClassifier.load(context.artifacts["gru_model"])

    def predict(self, context, model_input, params=None):
        import numpy as np
        X = np.asarray(model_input)
        return self.model.predict_proba(X)

# Hyperparameters exactly as used in the existing training scripts - copied
# here for logging purposes only, never used to retrain. See
# src/ml/train_baseline.py and src/deep_learning/train_gru.py.
BASELINE_PARAMS = {
    "logistic_regression": {"max_iter": 2000, "class_weight": "balanced", "random_state": 42},
    "random_forest": {"n_estimators": 300, "max_depth": 6, "class_weight": "balanced",
                       "random_state": 42, "n_jobs": -1},
    "gradient_boosting": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05,
                           "random_state": 42, "note": "class-imbalance handled via sample_weight, no class_weight param"},
}
BASELINE_MODEL_FILES = {
    "logistic_regression": "baseline_logistic_regression.pkl",
    "random_forest": "baseline_random_forest.pkl",
    "gradient_boosting": "baseline_gradient_boosting.pkl",
}
METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def _init_tracking():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def _log_metrics_block(metrics: dict, prefix: str):
    for key in METRIC_KEYS:
        value = metrics.get(key)
        if value is not None:
            mlflow.log_metric(f"{prefix}_{key}", float(value))
    cm = metrics.get("confusion_matrix")
    if cm:
        for k, v in cm.items():
            mlflow.log_metric(f"{prefix}_cm_{k}", float(v))
    if "threshold" in metrics:
        mlflow.log_metric(f"{prefix}_threshold", float(metrics["threshold"]))
    if "n_samples" in metrics:
        mlflow.log_metric(f"{prefix}_n_samples", float(metrics["n_samples"]))
    if "n_positive" in metrics:
        mlflow.log_metric(f"{prefix}_n_positive", float(metrics["n_positive"]))


def log_baseline_model_run(model_name: str) -> str:
    """Log one baseline model's real params/metrics/artifact as one MLflow
    run. Returns the run_id. Raises FileNotFoundError if Stage II's outputs
    are missing (never fabricates a run for a model that wasn't trained).
    """
    if model_name not in BASELINE_MODEL_FILES:
        raise ValueError(f"Unknown baseline model '{model_name}'")

    metrics_path = REPORTS_DIR / "stage2_baseline_metrics.json"
    model_path = MODELS_DIR / BASELINE_MODEL_FILES[model_name]
    if not metrics_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Stage II outputs missing for '{model_name}' - run Stage II first")

    all_metrics = json.loads(metrics_path.read_text())
    model_metrics = all_metrics[model_name]
    model = joblib.load(model_path)

    _init_tracking()
    with mlflow.start_run(run_name=f"stage2_{model_name}") as run:
        mlflow.set_tag("stage", "II")
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("source", "existing Stage II training run (not retrained)")
        mlflow.log_params(BASELINE_PARAMS[model_name])
        mlflow.log_param("selected_threshold", model_metrics.get("selected_threshold"))
        for split in ("train", "validation", "test"):
            if split in model_metrics:
                _log_metrics_block(model_metrics[split], split)
        mlflow.log_artifact(str(metrics_path), artifact_path="stage2_reports")
        mlflow.sklearn.log_model(model, name="model")
        return run.info.run_id


def log_gru_run() -> str:
    """Log the GRU's real params/metrics/artifact as one MLflow run."""
    metrics_path = REPORTS_DIR / "stage2_gru_metrics.json"
    model_path = MODELS_DIR / "gru_model.npz"
    contract_path = MODELS_DIR / "gru_feature_contract.json"
    if not metrics_path.exists() or not model_path.exists():
        raise FileNotFoundError("Stage II GRU outputs missing - run Stage II first")

    gru_metrics = json.loads(metrics_path.read_text())

    _init_tracking()
    with mlflow.start_run(run_name="stage2_gru") as run:
        mlflow.set_tag("stage", "II")
        mlflow.set_tag("model_type", "gru")
        mlflow.set_tag("source", "existing Stage II training run (not retrained)")
        mlflow.set_tag("framework", "hand-rolled NumPy (no TF/PyTorch available in the original sandbox)")
        mlflow.log_param("window", gru_metrics.get("window"))
        mlflow.log_param("hidden_size", gru_metrics.get("hidden_size"))
        mlflow.log_param("epochs", gru_metrics.get("epochs"))
        mlflow.log_param("selected_threshold", gru_metrics.get("selected_threshold"))
        for split in ("train", "validation", "test"):
            if split in gru_metrics:
                _log_metrics_block(gru_metrics[split], split)
        if gru_metrics.get("train_loss_curve"):
            for step, loss in enumerate(gru_metrics["train_loss_curve"]):
                mlflow.log_metric("train_loss", float(loss), step=step)
        mlflow.log_artifact(str(metrics_path), artifact_path="stage2_reports")
        if contract_path.exists():
            mlflow.log_artifact(str(contract_path), artifact_path="model_contract")
        # Logged as a proper pyfunc model (not a bare file artifact) so the
        # Model Registry can point at it - see GRUPyfuncWrapper above.
        gru_numpy_path = PROJECT_ROOT / "src" / "deep_learning" / "gru_numpy.py"
        mlflow.pyfunc.log_model(
            name="model",
            python_model=GRUPyfuncWrapper(),
            artifacts={"gru_model": str(model_path)},
            code_paths=[str(gru_numpy_path)],
        )
        return run.info.run_id


def log_all_stage2_runs() -> dict:
    """Log all 4 real Stage II models (3 baseline + GRU) as 4 separate
    MLflow runs. Returns {model_name: run_id}.
    """
    run_ids = {}
    for model_name in BASELINE_MODEL_FILES:
        run_ids[model_name] = log_baseline_model_run(model_name)
    run_ids["gru"] = log_gru_run()
    return run_ids


def register_best_model(run_ids: dict) -> dict:
    """Register Stage II's own already-selected best model (per
    `reports/stage2_model_comparison.json::best_model_by_validation_f1` -
    not re-decided here) in the MLflow Model Registry, pointing at the run
    just logged for it.
    """
    comparison_path = REPORTS_DIR / "stage2_model_comparison.json"
    if not comparison_path.exists():
        raise FileNotFoundError("reports/stage2_model_comparison.json missing - run Stage II first")
    comparison = json.loads(comparison_path.read_text())
    best_model_name = comparison["best_model_by_validation_f1"]

    if best_model_name not in run_ids:
        raise ValueError(f"No logged run for Stage II's selected best model '{best_model_name}'")

    _init_tracking()
    run_id = run_ids[best_model_name]
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    return {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "selected_model": best_model_name,
        "selection_basis": "reports/stage2_model_comparison.json::best_model_by_validation_f1 "
                            "(Stage II's own selection, not re-decided here)",
        "version": result.version,
        "source_run_id": run_id,
    }
