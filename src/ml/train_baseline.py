"""Stage II - Baseline ML models for `failure_next_shift` prediction.

Trains Logistic Regression, Random Forest, and Gradient Boosting on the
Stage I train/validation/test splits (data/processed/{train,validation,test}.csv).
No leakage: imputation stats, categorical column alignment, and the
decision threshold are all fit on train/validation only, never on test.

Run:
    python -m src.ml.train_baseline
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.metrics_utils import compute_metrics, best_threshold_by_f1  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET = "failure_next_shift"
# Identifier / non-predictive columns (kept for reporting, dropped from X).
ID_COLS = ["machine_id", "date", "shift", "sim_day"]
CATEGORICAL_COLS = ["machine_type", "operating_status"]
RANDOM_STATE = 42


def load_splits():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "validation.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, val, test


def build_feature_frame(df: pd.DataFrame, feature_cols: list[str], dummy_cols: list[str]):
    """One-hot encode categoricals, then reindex to the exact train-fit column set."""
    X = pd.get_dummies(df[feature_cols], columns=CATEGORICAL_COLS, dummy_na=False)
    X = X.reindex(columns=dummy_cols, fill_value=0)
    return X


def add_missing_indicators_and_impute(train_X, val_X, test_X):
    """Add `<col>_missing` flags (missingness is informative - e.g. DOWN state
    sensors, or a machine with no maintenance history yet), then median-impute
    using TRAIN-only medians.
    """
    cols_with_na = [c for c in train_X.columns if train_X[c].isna().any()
                     or val_X[c].isna().any() or test_X[c].isna().any()]
    medians = {}
    train_flags, val_flags, test_flags = {}, {}, {}
    for c in cols_with_na:
        flag = f"{c}_missing"
        train_flags[flag] = train_X[c].isna().astype(int)
        val_flags[flag] = val_X[c].isna().astype(int)
        test_flags[flag] = test_X[c].isna().astype(int)
        med = train_X[c].median()
        medians[c] = float(med) if not pd.isna(med) else 0.0
        train_X[c] = train_X[c].fillna(medians[c])
        val_X[c] = val_X[c].fillna(medians[c])
        test_X[c] = test_X[c].fillna(medians[c])

    train_X = pd.concat([train_X, pd.DataFrame(train_flags, index=train_X.index)], axis=1)
    val_X = pd.concat([val_X, pd.DataFrame(val_flags, index=val_X.index)], axis=1)
    test_X = pd.concat([test_X, pd.DataFrame(test_flags, index=test_X.index)], axis=1)
    return train_X, val_X, test_X, medians


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train, val, test = load_splits()

    feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
    y_train, y_val, y_test = train[TARGET], val[TARGET], test[TARGET]

    train_X_raw = pd.get_dummies(train[feature_cols], columns=CATEGORICAL_COLS, dummy_na=False)
    dummy_cols = list(train_X_raw.columns)  # frozen column contract from TRAIN only

    train_X = train_X_raw.reindex(columns=dummy_cols, fill_value=0)
    val_X = build_feature_frame(val, feature_cols, dummy_cols)
    test_X = build_feature_frame(test, feature_cols, dummy_cols)

    train_X, val_X, test_X, medians = add_missing_indicators_and_impute(train_X, val_X, test_X)
    feature_names = list(train_X.columns)

    print(f"Feature matrix: train={train_X.shape}, val={val_X.shape}, test={test_X.shape}")
    print(f"Train positives: {int(y_train.sum())}/{len(y_train)}  "
          f"Val positives: {int(y_val.sum())}/{len(y_val)}  "
          f"Test positives: {int(y_test.sum())}/{len(y_test)}")

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }

    # GradientBoostingClassifier has no class_weight param -> use sample_weight instead.
    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    sample_weight = np.where(y_train == 1, pos_weight, 1.0)

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        if name == "gradient_boosting":
            model.fit(train_X, y_train, sample_weight=sample_weight)
        else:
            model.fit(train_X, y_train)

        val_prob = model.predict_proba(val_X)[:, 1]
        # Threshold fit on validation only (never test) - class is too rare for 0.5.
        thresh = best_threshold_by_f1(y_val, val_prob)

        train_prob = model.predict_proba(train_X)[:, 1]
        test_prob = model.predict_proba(test_X)[:, 1]

        results[name] = {
            "train": compute_metrics(y_train, train_prob, threshold=thresh),
            "validation": compute_metrics(y_val, val_prob, threshold=thresh),
            "test": compute_metrics(y_test, test_prob, threshold=thresh),
            "selected_threshold": thresh,
        }
        print(f"  val: P={results[name]['validation']['precision']:.3f} "
              f"R={results[name]['validation']['recall']:.3f} "
              f"F1={results[name]['validation']['f1']:.3f} "
              f"ROC-AUC={results[name]['validation']['roc_auc']}")

        joblib.dump(model, MODELS_DIR / f"baseline_{name}.pkl")

    # Select best model by validation F1 (imbalanced target -> not accuracy).
    best_name = max(results, key=lambda k: results[k]["validation"]["f1"])
    results["best_model"] = best_name
    print(f"\nBest baseline model (by validation F1): {best_name}")

    with open(MODELS_DIR / "baseline_feature_contract.json", "w") as f:
        json.dump(
            {"feature_names": feature_names, "categorical_cols": CATEGORICAL_COLS,
             "id_cols": ID_COLS, "target": TARGET, "impute_medians": medians},
            f, indent=2,
        )
    with open(REPORTS_DIR / "stage2_baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved models to {MODELS_DIR}")
    print(f"Saved metrics to {REPORTS_DIR / 'stage2_baseline_metrics.json'}")


if __name__ == "__main__":
    main()
