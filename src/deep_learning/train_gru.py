"""Stage II - Deep Learning: GRU sequence model for `failure_next_shift`.

Builds sliding-window shift sequences per machine (window=5 consecutive
shifts, strictly within one split's date range - no windows span the
train/validation/test day boundaries set in Stage I), then trains a
from-scratch NumPy GRU (see gru_numpy.py) as a binary classifier.

Unlike the baseline models (which consume Stage I's already-engineered
lag/rollmean features), this model is fed the raw per-shift sensor means
directly and learns temporal dynamics itself via the recurrence - this is
what actually makes it a sequence/deep-learning model rather than a
re-run of the tabular baseline on the same engineered columns.

Run:
    python -m src.deep_learning.train_gru
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.metrics_utils import compute_metrics, best_threshold_by_f1  # noqa: E402
from src.deep_learning.gru_numpy import GRUBinaryClassifier, bce_loss  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET = "failure_next_shift"
WINDOW = 5
SHIFT_ORDER = {"Morning": 0, "Afternoon": 1, "Night": 2}
RANDOM_STATE = 42

# Raw (non-lagged) per-shift signals - the GRU learns temporal patterns
# itself instead of consuming Stage I's pre-computed lag/rollmean features.
NUMERIC_SEQ_COLS = [
    "production_rate", "quality_score", "downtime_minutes",
    "temperature_k_mean", "process_temperature_k_mean",
    "vibration_mm_s_mean", "pressure_bar_mean", "torque_nm_mean",
    "rotational_speed_rpm_mean", "load_pct_mean",
    "downtime_ratio", "production_efficiency",
    "days_since_last_maintenance", "maintenance_count_past_7d",
    "has_recent_maintenance", "failure_occurred_this_shift",
]
CATEGORICAL_SEQ_COLS = ["operating_status"]  # RUNNING / DOWN / MAINTENANCE


def _sort_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_shift_order"] = df["shift"].map(SHIFT_ORDER)
    return df.sort_values(["machine_id", "sim_day", "_shift_order"]).reset_index(drop=True)


def build_sequences(df: pd.DataFrame, feature_cols: list[str], window: int = WINDOW):
    """Sliding windows of `window` consecutive shifts per machine, target =
    failure_next_shift of the LAST shift in the window. Windows never span
    machines (grouped) and only use rows already present in this split
    (so never span the Stage I day-range split boundaries).
    """
    df = _sort_key(df)
    X_list, y_list, meta = [], [], []
    for machine_id, g in df.groupby("machine_id", sort=False):
        g = g.reset_index(drop=True)
        vals = g[feature_cols].to_numpy(dtype=float)
        targets = g[TARGET].to_numpy(dtype=float)
        for start in range(0, len(g) - window + 1):
            end = start + window
            X_list.append(vals[start:end])
            y_list.append(targets[end - 1])
            meta.append({
                "machine_id": machine_id,
                "date": g.loc[end - 1, "date"],
                "shift": g.loc[end - 1, "shift"],
            })
    X = np.stack(X_list) if X_list else np.zeros((0, window, len(feature_cols)))
    y = np.array(y_list)
    return X, y, meta


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "validation.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    # One-hot the categorical per-shift status, fit column set on train only.
    train_status = pd.get_dummies(train[CATEGORICAL_SEQ_COLS], columns=CATEGORICAL_SEQ_COLS)
    status_cols = list(train_status.columns)
    for split_df in (train, val, test):
        dummies = pd.get_dummies(split_df[CATEGORICAL_SEQ_COLS], columns=CATEGORICAL_SEQ_COLS)
        dummies = dummies.reindex(columns=status_cols, fill_value=0)
        for c in status_cols:
            split_df[c] = dummies[c].values

    feature_cols = NUMERIC_SEQ_COLS + status_cols

    # Train-only median imputation for the raw numeric sequence columns.
    medians = {c: float(train[c].median()) for c in NUMERIC_SEQ_COLS if train[c].isna().any()}
    for split_df in (train, val, test):
        for c, m in medians.items():
            split_df[c] = split_df[c].fillna(m)

    # Train-only standardization.
    means = train[feature_cols].mean()
    stds = train[feature_cols].std().replace(0, 1.0)
    for split_df in (train, val, test):
        split_df[feature_cols] = (split_df[feature_cols] - means) / stds

    X_train, y_train, _ = build_sequences(train, feature_cols)
    X_val, y_val, _ = build_sequences(val, feature_cols)
    X_test, y_test, meta_test = build_sequences(test, feature_cols)

    print(f"Sequences: train={X_train.shape} val={X_val.shape} test={X_test.shape}")
    print(f"Positives: train={int(y_train.sum())} val={int(y_val.sum())} test={int(y_test.sum())}")

    model = GRUBinaryClassifier(n_features=len(feature_cols), hidden_size=16, seed=RANDOM_STATE)

    n_epochs = 60
    batch_size = 32
    lr = 0.01
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(X_train)
    pos_weight = (n - y_train.sum()) / max(y_train.sum(), 1)  # upweight rare failures in loss grad

    history = []
    for epoch in range(1, n_epochs + 1):
        idx = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            Xb, yb = X_train[batch_idx], y_train[batch_idx]
            probs, cache = model.forward(Xb)
            epoch_loss += bce_loss(probs, yb) * len(batch_idx)

            # Weight the gradient contribution of positive examples so the
            # rare failure class isn't drowned out (~1% positive rate).
            sample_w = np.where(yb == 1, pos_weight, 1.0)
            grads = model.backward(cache, yb, sample_weight=sample_w)
            model.adam_step(grads, lr=lr)
        history.append(epoch_loss / n)
        if epoch % 10 == 0 or epoch == 1:
            val_probs = model.predict_proba(X_val) if len(X_val) else np.array([])
            val_loss = bce_loss(val_probs, y_val) if len(X_val) else float("nan")
            print(f"Epoch {epoch:3d}  train_loss={history[-1]:.4f}  val_loss={val_loss:.4f}")

    train_probs = model.predict_proba(X_train)
    val_probs = model.predict_proba(X_val)
    test_probs = model.predict_proba(X_test)

    thresh = best_threshold_by_f1(y_val, val_probs)
    results = {
        "train": compute_metrics(y_train, train_probs, threshold=thresh),
        "validation": compute_metrics(y_val, val_probs, threshold=thresh),
        "test": compute_metrics(y_test, test_probs, threshold=thresh),
        "selected_threshold": thresh,
        "window": WINDOW,
        "hidden_size": 16,
        "epochs": n_epochs,
        "train_loss_curve": history,
    }
    print(f"\nGRU val: P={results['validation']['precision']:.3f} "
          f"R={results['validation']['recall']:.3f} F1={results['validation']['f1']:.3f} "
          f"ROC-AUC={results['validation']['roc_auc']}")
    print(f"GRU test: P={results['test']['precision']:.3f} "
          f"R={results['test']['recall']:.3f} F1={results['test']['f1']:.3f} "
          f"ROC-AUC={results['test']['roc_auc']}")

    # Error analysis: false negatives / false positives on test set.
    test_pred = (test_probs >= thresh).astype(int)
    errors = []
    for i, (m, p, yt) in enumerate(zip(meta_test, test_pred, y_test)):
        if p != yt:
            errors.append({**m, "true": int(yt), "predicted": int(p), "prob": float(test_probs[i])})
    results["test_error_cases"] = errors

    model.save(MODELS_DIR / "gru_model.npz")
    with open(MODELS_DIR / "gru_feature_contract.json", "w") as f:
        json.dump({
            "feature_cols": feature_cols, "window": WINDOW,
            "means": means.to_dict(), "stds": stds.to_dict(),
            "impute_medians": medians, "status_cols": status_cols,
        }, f, indent=2)
    with open(REPORTS_DIR / "stage2_gru_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved GRU model to {MODELS_DIR / 'gru_model.npz'}")
    print(f"Saved metrics to {REPORTS_DIR / 'stage2_gru_metrics.json'}")


if __name__ == "__main__":
    main()
