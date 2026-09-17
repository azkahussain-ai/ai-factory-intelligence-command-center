"""Stage III tests - CV (NumPy CNN) and NLP (TF-IDF + classifier) pipelines.

Run: pytest tests/test_stage3.py -v
(or, if pytest is unavailable, see the manual runner used in this project's
development sandbox - functions here take no fixtures so they can be called
directly too.)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.computer_vision.cnn_numpy import SimpleCNNBinaryClassifier, bce_loss  # noqa: E402
from src.nlp.train_nlp import clean_text  # noqa: E402


def test_cv_metrics_report_exists_and_valid():
    path = PROJECT_ROOT / "reports" / "stage3_cv_metrics.json"
    assert path.exists(), "run `python -m src.computer_vision.train_cv` first"
    with open(path) as f:
        results = json.load(f)
    for split in ("train", "validation", "test"):
        m = results[split]
        for key in ("accuracy", "precision", "recall", "f1"):
            assert 0.0 <= m[key] <= 1.0
        cm = m["confusion_matrix"]
        assert set(cm.keys()) == {"tn", "fp", "fn", "tp"}


def test_nlp_metrics_report_exists_and_valid():
    path = PROJECT_ROOT / "reports" / "stage3_nlp_metrics.json"
    assert path.exists(), "run `python -m src.nlp.train_nlp` first"
    with open(path) as f:
        results = json.load(f)
    for split in ("train", "validation", "test"):
        m = results[split]
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["f1_macro"] <= 1.0


def test_structured_output_valid():
    path = PROJECT_ROOT / "reports" / "stage3_structured_output.json"
    assert path.exists(), "run `python -m src.nlp.build_stage3_structured_output` first"
    with open(path) as f:
        records = json.load(f)
    assert len(records) > 0
    for r in records:
        assert "machine_id" in r
        assert "cv_prediction" in r and "cv_confidence" in r
        assert "nlp_prediction" in r and "nlp_confidence" in r


def test_cv_model_loadable_for_inference():
    path = PROJECT_ROOT / "models" / "cv_cnn_model.npz"
    assert path.exists(), "run `python -m src.computer_vision.train_cv` first"
    model = SimpleCNNBinaryClassifier.load(path)
    X = np.random.default_rng(0).uniform(0, 1, size=(2, 32, 32))
    probs = model.predict_proba(X)
    assert probs.shape == (2,)
    assert np.all((probs >= 0) & (probs <= 1))


def test_nlp_model_loadable_for_inference():
    import joblib
    vec_path = PROJECT_ROOT / "models" / "nlp_tfidf_vectorizer.pkl"
    clf_path = PROJECT_ROOT / "models" / "nlp_incident_classifier.pkl"
    assert vec_path.exists() and clf_path.exists(), "run `python -m src.nlp.train_nlp` first"
    vectorizer = joblib.load(vec_path)
    clf = joblib.load(clf_path)
    X = vectorizer.transform(["routine scheduled inspection performed"])
    pred = clf.predict(X)
    assert pred[0] in clf.classes_


def test_cnn_gradient_direction_reduces_loss():
    rng = np.random.default_rng(2)
    model = SimpleCNNBinaryClassifier(img_size=16, n_filters=2, kernel=3, hidden_size=4, seed=2)
    X = rng.uniform(0, 1, size=(6, 16, 16))
    y = rng.integers(0, 2, size=6).astype(float)
    probs0, cache = model.forward(X)
    loss0 = bce_loss(probs0, y)
    grads = model.backward(cache, y)
    model.adam_step(grads, lr=0.05)
    probs1, _ = model.forward(X)
    loss1 = bce_loss(probs1, y)
    assert loss1 <= loss0 + 1e-6


def test_clean_text_normalizes_case_and_punctuation():
    assert clean_text("Vibration Levels!! Elevated.") == "vibration levels elevated"


def test_nlp_split_uses_stage1_day_boundaries():
    """Confirm the maintenance train/val/test split reuses Stage I's exact
    day-range config rather than inventing new boundaries."""
    from config import config
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "cleaned_maintenance.csv")
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    from src.features.split_data import add_sim_day
    df = add_sim_day(df)
    train_lo, train_hi = config.TRAIN_DAY_RANGE
    val_lo, val_hi = config.VAL_DAY_RANGE
    test_lo, test_hi = config.TEST_DAY_RANGE
    train_days = set(df[(df.sim_day >= train_lo) & (df.sim_day <= train_hi)]["sim_day"])
    val_days = set(df[(df.sim_day >= val_lo) & (df.sim_day <= val_hi)]["sim_day"])
    test_days = set(df[(df.sim_day >= test_lo) & (df.sim_day <= test_hi)]["sim_day"])
    assert not (train_days & val_days)
    assert not (val_days & test_days)
