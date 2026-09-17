"""Stage VI — Predictive-maintenance XAI for the GRU model.

Stage V's Predictive Maintenance Agent scores machines with Stage II's
trained GRU (`models/gru_model.npz`, a hand-rolled NumPy RNN - it is the
model actually driving Stage V's decisions, so it is the one explained
here). SHAP's DeepExplainer requires a TensorFlow/PyTorch graph to hook
gradients into, which this model is not; SHAP's KernelExplainer is
model-agnostic but needs many forward passes per feature and is
unnecessarily slow for a workflow meant to run per agent call. Permutation
importance is explicitly listed as an acceptable alternative in the Stage
VI brief and needs only forward passes through the model exactly as it
already runs - no changes to the trained model.

(SHAP is used directly, unmodified, for the Stage II baseline tree models -
see `src/xai/shap_tabular_explainer.py` - since those are fully
SHAP-compatible.)

Method: for each (timestep, feature) position in the machine's standardized
input window, replace that single value with the *training* mean (a neutral
reference, computed once - see `models/gru_feature_contract.json:
means`/`stds`, which are themselves train-only Stage II statistics) and
measure how much the predicted probability moves. Aggregated across
timesteps into one importance score per feature.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.config import PROCESSED_DIR
from src.agents.predictive_maintenance_agent import (
    SHIFT_ORDER,
    _load_gru_artifacts,
    _load_gru_threshold,
    _probability_to_risk_level,
    build_gru_input,
)
from src.agents.schemas import unavailable_result, error_result


def _build_window_as_of(machine_id: str, contract: dict, sim_day: int, shift: str):
    """Build the standardized GRU input for the window ending at a specific
    historical (sim_day, shift) - used for explaining a specific past
    prediction (e.g. a known failure event) rather than only "right now".
    Reuses the exact same feature-processing steps as
    `predictive_maintenance_agent.build_gru_input`, just starting from a
    caller-chosen anchor row instead of the latest available one.
    """
    window = contract["window"]
    status_cols = contract["status_cols"]

    frames = []
    for split_name in ["train.csv", "validation.csv", "test.csv"]:
        path = PROCESSED_DIR / split_name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return None, None
    df = pd.concat(frames, ignore_index=True)

    g = df[df["machine_id"] == machine_id].copy()
    g["_shift_order"] = g["shift"].map(SHIFT_ORDER)
    g = g.sort_values(["sim_day", "_shift_order"]).reset_index(drop=True)

    matches = g[(g["sim_day"] == sim_day) & (g["shift"] == shift)]
    if matches.empty:
        return None, None
    anchor_idx = matches.index[0]
    if anchor_idx + 1 < window:
        return None, None
    recent = g.iloc[anchor_idx - window + 1: anchor_idx + 1].copy()

    for c, med in contract["impute_medians"].items():
        if c in recent.columns:
            recent[c] = recent[c].fillna(med)
    status_dummies = pd.get_dummies(recent["operating_status"], prefix="operating_status")
    status_dummies = status_dummies.reindex(columns=status_cols, fill_value=0)
    for c in status_cols:
        recent[c] = status_dummies[c].values

    feature_cols = contract["feature_cols"]
    means = pd.Series(contract["means"])
    stds = pd.Series(contract["stds"])
    X = recent[feature_cols].astype(float)
    X = (X - means[feature_cols]) / stds[feature_cols]
    X_seq = X.to_numpy().reshape(1, window, len(feature_cols))
    return X_seq, recent


def explain(machine_id: str, top_k: int = 5, sim_day: int | None = None, shift: str | None = None) -> dict:
    """Explain the GRU's prediction for a machine.

    By default explains the *current* (most recent) window - the same one
    the live Predictive Maintenance Agent scores. Pass `sim_day`/`shift` to
    instead explain a specific historical window (e.g. a known past failure
    event), for validation/demonstration purposes.
    """
    model, contract = _load_gru_artifacts()
    if model is None:
        return unavailable_result("Stage II GRU model/feature contract not found for XAI")

    try:
        if sim_day is not None and shift is not None:
            X_seq, recent = _build_window_as_of(machine_id, contract, sim_day, shift)
        else:
            X_seq, recent = build_gru_input(machine_id, contract)
        if X_seq is None:
            return unavailable_result(
                f"Not enough recorded history to build a {contract['window']}-shift window "
                f"for machine {machine_id}" + (f" ending {sim_day}/{shift}" if sim_day else ""))

        base_prob = float(model.predict_proba(X_seq)[0])
        feature_cols = contract["feature_cols"]
        window = contract["window"]

        # Neutral reference value per feature = its standardized training
        # mean, i.e. 0.0 after standardization (means/stds are train-only).
        importances = np.zeros(len(feature_cols))
        for f_idx in range(len(feature_cols)):
            perturbed = X_seq.copy()
            perturbed[0, :, f_idx] = 0.0  # neutralize this feature across the whole window
            perturbed_prob = float(model.predict_proba(perturbed)[0])
            importances[f_idx] = base_prob - perturbed_prob  # positive = feature was pushing risk up

        order = np.argsort(-np.abs(importances))
        last_row_actual = recent[feature_cols].iloc[-1]
    except Exception as exc:
        return error_result(f"GRU permutation-importance explanation failed for {machine_id}: {exc}")

    feature_explanations = []
    for idx in order[:top_k]:
        contribution = float(importances[idx])
        feature_explanations.append({
            "feature": feature_cols[idx],
            "value": float(last_row_actual.iloc[idx]),
            "contribution": round(contribution, 6),
            "direction": "increases_risk" if contribution > 0 else "decreases_risk",
        })

    threshold = _load_gru_threshold()
    return {
        "status": "ok",
        "machine_id": machine_id,
        "model_name": "gru",
        "method": "permutation_importance",
        "method_reason": "SHAP DeepExplainer requires a TF/PyTorch graph this NumPy GRU does not have; "
                          "KernelExplainer is model-agnostic but too slow for per-call agent use.",
        "prediction": {
            "probability": round(base_prob, 4),
            "predicted_failure": int(base_prob >= threshold),
            "risk_level": _probability_to_risk_level(base_prob, threshold),
        },
        "feature_explanations": feature_explanations,
        "top_features": [f["feature"] for f in feature_explanations],
        "window_shifts_used": window,
    }
