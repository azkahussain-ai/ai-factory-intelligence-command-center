"""Stage III - NLP: incident-type classification on maintenance notes.

Uses the existing `data/processed/cleaned_maintenance.csv` (Stage I output,
not regenerated here). Splits by the same simulation-day boundaries Stage I
already defined (`config.TRAIN_DAY_RANGE` etc., reused via
`src.features.split_data.add_sim_day`) applied to the maintenance
timestamp's date, so the split methodology is consistent with Stage I/II
and no new split logic is invented.

Target: `incident_type` (Scheduled / Mechanical / Electrical / Quality).
Pipeline: lowercase + punctuation-stripped text -> TF-IDF (fit on train
only) -> Logistic Regression (class_weight='balanced', since Scheduled
dominates the 49-row dataset).

Run:
    python -m src.nlp.train_nlp
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.split_data import add_sim_day  # noqa: E402
from config import config  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_maintenance.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_STATE = 42
TARGET = "incident_type"


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    df = add_sim_day(df)

    train_lo, train_hi = config.TRAIN_DAY_RANGE
    val_lo, val_hi = config.VAL_DAY_RANGE
    test_lo, test_hi = config.TEST_DAY_RANGE
    train = df[(df["sim_day"] >= train_lo) & (df["sim_day"] <= train_hi)].copy()
    val = df[(df["sim_day"] >= val_lo) & (df["sim_day"] <= val_hi)].copy()
    test = df[(df["sim_day"] >= test_lo) & (df["sim_day"] <= test_hi)].copy()

    for split_df in (train, val, test):
        split_df["clean_note"] = split_df["maintenance_note"].apply(clean_text)

    print(f"Maintenance records: train={len(train)} val={len(val)} test={len(test)}")
    print(f"Train class counts: {train[TARGET].value_counts().to_dict()}")

    vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 2), min_df=1)
    X_train = vectorizer.fit_transform(train["clean_note"])
    X_val = vectorizer.transform(val["clean_note"])
    X_test = vectorizer.transform(test["clean_note"])

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    clf.fit(X_train, train[TARGET])

    def eval_split(X, y_true, name):
        y_pred = clf.predict(X)
        labels = sorted(df[TARGET].unique())
        return {
            "n_samples": int(len(y_true)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "confusion_matrix_labels": labels,
        }

    results = {
        "train": eval_split(X_train, train[TARGET], "train"),
        "validation": eval_split(X_val, val[TARGET], "validation") if len(val) else None,
        "test": eval_split(X_test, test[TARGET], "test") if len(test) else None,
        "note": (
            "Only 49 maintenance records exist in total (train=24/val=9/test=16); "
            "some classes (e.g. Quality, Electrical) have very few examples per "
            "split, so per-class precision/recall for those classes is high-variance. "
            "Macro-averaged metrics are reported since classes are imbalanced."
        ),
    }
    print(f"\nNLP val: acc={results['validation']['accuracy']:.3f} "
          f"F1_macro={results['validation']['f1_macro']:.3f}")
    print(f"NLP test: acc={results['test']['accuracy']:.3f} "
          f"F1_macro={results['test']['f1_macro']:.3f}")

    joblib.dump(vectorizer, MODELS_DIR / "nlp_tfidf_vectorizer.pkl")
    joblib.dump(clf, MODELS_DIR / "nlp_incident_classifier.pkl")
    with open(REPORTS_DIR / "stage3_nlp_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Structured, machine-readable predictions for later stages (all splits).
    all_preds = []
    for split_name, split_df, X_split in (
        ("train", train, X_train), ("validation", val, X_val), ("test", test, X_test)
    ):
        if len(split_df) == 0:
            continue
        probs = clf.predict_proba(X_split)
        preds = clf.predict(X_split)
        classes = list(clf.classes_)
        for i, (_, row) in enumerate(split_df.iterrows()):
            confidence = float(probs[i][classes.index(preds[i])])
            all_preds.append({
                "machine_id": row["machine_id"],
                "timestamp": row["timestamp"],
                "split": split_name,
                "true_incident_type": row[TARGET],
                "nlp_prediction": preds[i],
                "nlp_confidence": confidence,
            })
    with open(REPORTS_DIR / "stage3_nlp_predictions.json", "w") as f:
        json.dump(all_preds, f, indent=2)

    print(f"\nSaved model/vectorizer to {MODELS_DIR}")
    print(f"Saved metrics to {REPORTS_DIR / 'stage3_nlp_metrics.json'}")
    print(f"Saved predictions to {REPORTS_DIR / 'stage3_nlp_predictions.json'}")


if __name__ == "__main__":
    main()
