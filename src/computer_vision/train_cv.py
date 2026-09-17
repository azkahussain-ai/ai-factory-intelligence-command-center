"""Stage III - Computer Vision: defect classification on synthetic equipment images.

Loads `data/synthetic/images_metadata.csv` (see
`generate_synthetic_images.py`), trains the from-scratch NumPy CNN
(`cnn_numpy.py`) on the train split, tunes nothing on test, and evaluates
with accuracy/precision/recall/F1/confusion matrix on validation and test.

Run:
    python -m src.computer_vision.train_cv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.computer_vision.cnn_numpy import SimpleCNNBinaryClassifier, bce_loss  # noqa: E402

METADATA_PATH = PROJECT_ROOT / "data" / "synthetic" / "images_metadata.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_STATE = 42
LABEL_MAP = {"normal": 0, "defect": 1}


def load_split(df: pd.DataFrame, split: str):
    rows = df[df["split"] == split]
    X = np.stack([
        np.array(Image.open(PROJECT_ROOT / p), dtype=np.float64) / 255.0
        for p in rows["image_path"]
    ])
    y = rows["label"].map(LABEL_MAP).to_numpy(dtype=float)
    return X, y, rows


def classification_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(METADATA_PATH)
    X_train, y_train, _ = load_split(df, "train")
    X_val, y_val, _ = load_split(df, "validation")
    X_test, y_test, rows_test = load_split(df, "test")

    print(f"Images: train={X_train.shape} val={X_val.shape} test={X_test.shape}")

    model = SimpleCNNBinaryClassifier(img_size=32, n_filters=4, kernel=3,
                                       hidden_size=16, seed=RANDOM_STATE)

    n_epochs = 40
    batch_size = 16
    lr = 0.01
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(X_train)

    history = []
    for epoch in range(1, n_epochs + 1):
        idx = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            Xb, yb = X_train[batch_idx], y_train[batch_idx]
            probs, cache = model.forward(Xb)
            epoch_loss += bce_loss(probs, yb) * len(batch_idx)
            grads = model.backward(cache, yb)
            model.adam_step(grads, lr=lr)
        history.append(epoch_loss / n)
        if epoch % 10 == 0 or epoch == 1:
            val_probs = model.predict_proba(X_val)
            val_loss = bce_loss(val_probs, y_val)
            print(f"Epoch {epoch:3d}  train_loss={history[-1]:.4f}  val_loss={val_loss:.4f}")

    train_probs = model.predict_proba(X_train)
    val_probs = model.predict_proba(X_val)
    test_probs = model.predict_proba(X_test)

    results = {
        "train": classification_metrics(y_train, train_probs),
        "validation": classification_metrics(y_val, val_probs),
        "test": classification_metrics(y_test, test_probs),
        "threshold": 0.5,
        "epochs": n_epochs,
        "train_loss_curve": history,
        "note": (
            "Trained on a 100-image synthetic dataset (documented in "
            "data/synthetic/SOURCE.md) - no real equipment images exist in "
            "this project. Metrics reflect separability of the synthetic "
            "generation rule, not real-world defect detection performance."
        ),
    }
    print(f"\nCV test: acc={results['test']['accuracy']:.3f} "
          f"P={results['test']['precision']:.3f} R={results['test']['recall']:.3f} "
          f"F1={results['test']['f1']:.3f}")

    model.save(MODELS_DIR / "cv_cnn_model.npz")
    with open(REPORTS_DIR / "stage3_cv_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Structured, machine-readable predictions for later stages (all splits).
    all_preds = []
    for split in ("train", "validation", "test"):
        X_s, y_s, rows_s = load_split(df, split)
        probs_s = model.predict_proba(X_s)
        for (_, row), prob in zip(rows_s.iterrows(), probs_s):
            all_preds.append({
                "image_path": row["image_path"],
                "machine_id": row["machine_id"],
                "split": split,
                "true_label": row["label"],
                "cv_prediction": "defect" if prob >= 0.5 else "normal",
                "cv_confidence": float(prob if prob >= 0.5 else 1 - prob),
            })
    with open(REPORTS_DIR / "stage3_cv_predictions.json", "w") as f:
        json.dump(all_preds, f, indent=2)

    print(f"Saved model to {MODELS_DIR / 'cv_cnn_model.npz'}")
    print(f"Saved metrics to {REPORTS_DIR / 'stage3_cv_metrics.json'}")
    print(f"Saved predictions to {REPORTS_DIR / 'stage3_cv_predictions.json'}")


if __name__ == "__main__":
    main()
