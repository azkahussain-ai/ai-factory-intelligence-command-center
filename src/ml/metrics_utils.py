"""Shared evaluation utilities for Stage II (ML + DL).

Kept separate from sklearn's own scorers so both the sklearn baselines and
the from-scratch GRU report metrics the exact same way, and so accuracy is
never used as the headline metric given the severe class imbalance
(failure_next_shift positives are ~1% of rows).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Compute the metric set required by the hackathon spec (never accuracy-only).

    y_prob: predicted probability of the positive (failure) class.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    n_pos = int(y_true.sum())
    metrics = {
        "n_samples": int(len(y_true)),
        "n_positive": n_pos,
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # ROC-AUC / PR-AUC are undefined with a single class present in y_true
    # (can happen with only 3-7 positives in a split) - guard explicitly
    # instead of letting sklearn silently warn or crash.
    if n_pos > 0 and n_pos < len(y_true):
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["confusion_matrix"] = {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }
    return metrics


def best_threshold_by_f1(y_true, y_prob) -> float:
    """Scan thresholds on a validation set and return the one maximizing F1.

    Used instead of a hard-coded 0.5 cutoff because the positive class is
    so rare that 0.5 rarely fires; this is fit on validation only, never
    on test.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    candidates = np.unique(np.round(y_prob, 3))
    if len(candidates) == 0:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t
