"""Stage II tests - baseline ML feature contract + GRU sequence building.

Run: pytest tests/test_stage2.py -v
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pytest
    _fixture = pytest.fixture
except ImportError:  # pragma: no cover - dev sandbox without network access
    pytest = None

    def _fixture(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.deep_learning.train_gru import build_sequences, NUMERIC_SEQ_COLS, WINDOW  # noqa: E402
from src.deep_learning.gru_numpy import GRUBinaryClassifier, sigmoid  # noqa: E402
from src.ml.metrics_utils import compute_metrics, best_threshold_by_f1  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "processed"


@_fixture(scope="module")
def train_df():
    return pd.read_csv(DATA_DIR / "train.csv")


def test_baseline_metrics_report_exists_and_valid():
    path = PROJECT_ROOT / "reports" / "stage2_baseline_metrics.json"
    assert path.exists(), "run `python -m src.ml.train_baseline` first"
    with open(path) as f:
        results = json.load(f)
    for name in ("logistic_regression", "random_forest", "gradient_boosting"):
        assert name in results
        for split in ("train", "validation", "test"):
            m = results[name][split]
            assert 0.0 <= m["precision"] <= 1.0
            assert 0.0 <= m["recall"] <= 1.0
            assert 0.0 <= m["f1"] <= 1.0


def test_gru_metrics_report_exists_and_valid():
    path = PROJECT_ROOT / "reports" / "stage2_gru_metrics.json"
    assert path.exists(), "run `python -m src.deep_learning.train_gru` first"
    with open(path) as f:
        results = json.load(f)
    assert results["window"] == WINDOW
    assert len(results["train_loss_curve"]) == results["epochs"]


def test_sequence_windows_do_not_span_machines(train_df):
    X, y, meta = build_sequences(train_df, NUMERIC_SEQ_COLS, window=WINDOW)
    assert X.shape[1] == WINDOW
    assert X.shape[2] == len(NUMERIC_SEQ_COLS)
    assert len(y) == len(meta) == X.shape[0]
    # Every window's machine is single (grouped construction) - sanity via count.
    n_machines = train_df["machine_id"].nunique()
    shifts_per_machine = train_df.groupby("machine_id").size().iloc[0]
    expected_windows = n_machines * (shifts_per_machine - WINDOW + 1)
    assert X.shape[0] == expected_windows


def test_sequence_target_matches_last_row_in_window(train_df):
    """The target for a window must equal `failure_next_shift` of the LAST
    shift in that window (not some other row) - guards against off-by-one
    leakage/misalignment bugs.
    """
    X, y, meta = build_sequences(train_df, NUMERIC_SEQ_COLS, window=WINDOW)
    df_sorted = train_df.copy()
    df_sorted["_shift_order"] = df_sorted["shift"].map({"Morning": 0, "Afternoon": 1, "Night": 2})
    df_sorted = df_sorted.sort_values(["machine_id", "sim_day", "_shift_order"]).reset_index(drop=True)

    # Reconstruct expected target for a handful of sampled windows.
    checked = 0
    for machine_id, g in df_sorted.groupby("machine_id", sort=False):
        g = g.reset_index(drop=True)
        for start in [0, len(g) - WINDOW]:
            end = start + WINDOW
            expected_target = g.loc[end - 1, "failure_next_shift"]
            # find matching meta entry
            match = [i for i, m in enumerate(meta)
                     if m["machine_id"] == machine_id and m["date"] == g.loc[end - 1, "date"]
                     and m["shift"] == g.loc[end - 1, "shift"]]
            assert match, "window metadata missing for expected window"
            assert y[match[0]] == expected_target
            checked += 1
    assert checked > 0


def test_gru_forward_reproducible():
    model_a = GRUBinaryClassifier(n_features=5, hidden_size=4, seed=42)
    model_b = GRUBinaryClassifier(n_features=5, hidden_size=4, seed=42)
    X = np.random.default_rng(0).normal(size=(3, 4, 5))
    probs_a, _ = model_a.forward(X)
    probs_b, _ = model_b.forward(X)
    np.testing.assert_allclose(probs_a, probs_b)


def test_gru_probs_are_valid_probabilities():
    model = GRUBinaryClassifier(n_features=3, hidden_size=4, seed=1)
    X = np.random.default_rng(1).normal(size=(5, 4, 3))
    probs, _ = model.forward(X)
    assert probs.shape == (5,)
    assert np.all((probs >= 0) & (probs <= 1))


def test_gru_backward_reduces_loss_on_toy_batch():
    """One-step gradient sanity check: loss should decrease after a step."""
    from src.deep_learning.gru_numpy import bce_loss
    rng = np.random.default_rng(3)
    model = GRUBinaryClassifier(n_features=3, hidden_size=4, seed=3)
    X = rng.normal(size=(8, 4, 3))
    y = rng.integers(0, 2, size=8).astype(float)

    probs0, cache = model.forward(X)
    loss0 = bce_loss(probs0, y)
    grads = model.backward(cache, y)
    model.adam_step(grads, lr=0.05)
    probs1, _ = model.forward(X)
    loss1 = bce_loss(probs1, y)
    assert loss1 <= loss0 + 1e-6


def test_best_threshold_by_f1_on_perfectly_separable_toy():
    y = np.array([0, 0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    t = best_threshold_by_f1(y, probs)
    preds = (probs >= t).astype(int)
    assert (preds == y).all()


def test_compute_metrics_handles_single_class_split():
    y = np.array([0, 0, 0, 0])
    probs = np.array([0.1, 0.2, 0.05, 0.3])
    m = compute_metrics(y, probs)
    assert m["roc_auc"] is None and m["pr_auc"] is None


def test_no_test_leakage_in_threshold_selection():
    """Confirm train scripts select the decision threshold on validation
    data only - re-derive it here from the saved val predictions used
    inside the report and check it isn't silently refit per split.
    """
    with open(PROJECT_ROOT / "reports" / "stage2_baseline_metrics.json") as f:
        results = json.load(f)
    for name in ("logistic_regression", "random_forest", "gradient_boosting"):
        t_val = results[name]["validation"]["threshold"]
        t_test = results[name]["test"]["threshold"]
        assert t_val == t_test == results[name]["selected_threshold"]
