"""Stage VI — SHAP explainer for Stage II's baseline tabular models.

Random Forest and Gradient Boosting (`models/baseline_random_forest.pkl`,
`models/baseline_gradient_boosting.pkl`) are real scikit-learn estimators,
directly compatible with SHAP's exact `TreeExplainer`. Logistic Regression
uses SHAP's `LinearExplainer`. This satisfies the Stage VI brief's
preferred method for tabular/predictive-maintenance XAI.

These baseline models are not the ones Stage V's live agent scores with
(that's the GRU - see `src/xai/gru_explainer.py`), but they are real,
already-trained Stage II artifacts, and SHAP is fully appropriate for them,
so they are explained here as the project's direct SHAP demonstration.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.config import PROCESSED_DIR, PROJECT_ROOT
from src.ml.train_baseline import build_feature_frame, CATEGORICAL_COLS, ID_COLS, TARGET
from src.agents.schemas import unavailable_result, error_result

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILES = {
    "random_forest": "baseline_random_forest.pkl",
    "gradient_boosting": "baseline_gradient_boosting.pkl",
    "logistic_regression": "baseline_logistic_regression.pkl",
}
TREE_MODELS = {"random_forest", "gradient_boosting"}


def _load_contract() -> dict | None:
    path = MODELS_DIR / "baseline_feature_contract.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _build_row(machine_id: str, contract: dict) -> pd.DataFrame | None:
    """Most recent row for a machine, encoded exactly as Stage II encoded
    training rows (same one-hot + missing-flag + train-only-median
    imputation contract).
    """
    row = None
    for split_name in ["test.csv", "validation.csv", "train.csv"]:
        path = PROCESSED_DIR / split_name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        g = df[df["machine_id"] == machine_id]
        if len(g):
            row = g.sort_values(["sim_day"]).tail(1)
            break
    if row is None:
        return None

    feature_cols = [c for c in row.columns if c not in ID_COLS + [TARGET]]
    feature_names = contract["feature_names"]
    dummy_cols = [c for c in feature_names if not c.endswith("_missing")]
    X = build_feature_frame(row, feature_cols, dummy_cols)

    medians = contract["impute_medians"]
    missing_flags = {}
    for col, med in medians.items():
        if col in X.columns:
            flag_name = f"{col}_missing"
            if flag_name in feature_names:
                missing_flags[flag_name] = X[col].isna().astype(int)
            X[col] = X[col].fillna(med)
    if missing_flags:
        X = pd.concat([X, pd.DataFrame(missing_flags, index=X.index)], axis=1)

    X = X.reindex(columns=feature_names, fill_value=0)
    return X


def _load_background_sample(contract: dict, n: int = 50) -> pd.DataFrame | None:
    """A representative background sample from the training split, encoded
    identically to the explained row - required for SHAP's LinearExplainer
    (and used as TreeExplainer's optional background) so contributions are
    measured against realistic feature distributions, not a degenerate
    single-point reference. Capped at `n` rows so this stays fast on a
    laptop, per the Stage VI performance guidance.
    """
    path = PROCESSED_DIR / "train.csv"
    if not path.exists():
        return None
    train = pd.read_csv(path)
    sample = train.sample(n=min(n, len(train)), random_state=42)

    feature_cols = [c for c in sample.columns if c not in ID_COLS + [TARGET]]
    feature_names = contract["feature_names"]
    dummy_cols = [c for c in feature_names if not c.endswith("_missing")]
    X = build_feature_frame(sample, feature_cols, dummy_cols)

    medians = contract["impute_medians"]
    missing_flags = {}
    for col, med in medians.items():
        if col in X.columns:
            flag_name = f"{col}_missing"
            if flag_name in feature_names:
                missing_flags[flag_name] = X[col].isna().astype(int)
            X[col] = X[col].fillna(med)
    if missing_flags:
        X = pd.concat([X, pd.DataFrame(missing_flags, index=X.index)], axis=1)
    return X.reindex(columns=feature_names, fill_value=0)


def explain(machine_id: str, model_name: str = "random_forest", top_k: int = 5) -> dict:
    if model_name not in MODEL_FILES:
        return error_result(f"Unknown baseline model '{model_name}'; choose from {list(MODEL_FILES)}")

    contract = _load_contract()
    model_path = MODELS_DIR / MODEL_FILES[model_name]
    if contract is None or not model_path.exists():
        return unavailable_result(f"Stage II baseline model/contract not found for '{model_name}'")

    try:
        X = _build_row(machine_id, contract)
        if X is None:
            return unavailable_result(f"No recorded shifts available for machine {machine_id}")

        model = joblib.load(model_path)
        prob = float(model.predict_proba(X)[:, 1][0])

        if model_name in TREE_MODELS:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            # Newer SHAP returns (n_samples, n_features, n_classes) for
            # binary classifiers; older returns a 2-item list [class0, class1].
            if isinstance(shap_values, list):
                row_values = np.asarray(shap_values[1])[0]
            elif shap_values.ndim == 3:
                row_values = shap_values[0, :, 1]
            else:
                row_values = shap_values[0]
        else:
            background = _load_background_sample(contract)
            if background is None:
                return unavailable_result("No training data available to build a SHAP background sample")
            explainer = shap.LinearExplainer(model, background)
            row_values = np.asarray(explainer.shap_values(X))[0]
    except Exception as exc:
        return error_result(f"SHAP explanation failed for {model_name} on {machine_id}: {exc}")

    order = np.argsort(-np.abs(row_values))
    feature_names = contract["feature_names"]
    feature_explanations = []
    for idx in order[:top_k]:
        contribution = float(row_values[idx])
        feature_explanations.append({
            "feature": feature_names[idx],
            "value": float(X.iloc[0, idx]),
            "contribution": round(contribution, 6),
            "direction": "increases_risk" if contribution > 0 else "decreases_risk",
        })

    return {
        "status": "ok",
        "machine_id": machine_id,
        "model_name": model_name,
        "method": "SHAP TreeExplainer" if model_name in TREE_MODELS else "SHAP LinearExplainer",
        "prediction": {"probability": round(prob, 4)},
        "feature_explanations": feature_explanations,
        "top_features": [f["feature"] for f in feature_explanations],
    }
