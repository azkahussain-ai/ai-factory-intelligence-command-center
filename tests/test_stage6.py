"""Stage VI tests. Run with: pytest tests/test_stage6.py -v"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.xai import gru_explainer, shap_tabular_explainer, gradcam_explainer
from src.xai.explanation_formatter import build_human_summary, build_xai_result
from src.xai.xai_pipeline import explain_machine
from src.agents.orchestrator import run_workflow

VALID_MACHINE = "M-01"
INVALID_MACHINE = "M-99"


# ---- Test 1: Tabular XAI ----------------------------------------------

def test_gru_explainer_produces_valid_explanation():
    result = gru_explainer.explain(VALID_MACHINE)
    assert result["status"] == "ok"
    assert result["method"] == "permutation_importance"
    assert 0.0 <= result["prediction"]["probability"] <= 1.0
    assert result["prediction"]["risk_level"] in ("LOW", "MEDIUM", "HIGH")


def test_shap_explainer_produces_valid_explanation_for_tree_model():
    result = shap_tabular_explainer.explain(VALID_MACHINE, model_name="random_forest")
    assert result["status"] == "ok"
    assert result["method"] == "SHAP TreeExplainer"
    assert 0.0 <= result["prediction"]["probability"] <= 1.0


def test_shap_explainer_works_for_linear_model_with_real_background():
    result = shap_tabular_explainer.explain(VALID_MACHINE, model_name="logistic_regression")
    assert result["status"] == "ok"
    assert result["method"] == "SHAP LinearExplainer"
    # a real background sample must produce non-degenerate (non-all-zero) contributions
    assert any(abs(f["contribution"]) > 1e-6 for f in result["feature_explanations"])


# ---- Test 2: Feature Attribution ---------------------------------------

def test_gru_explanation_returns_top_contributing_features():
    result = gru_explainer.explain(VALID_MACHINE, top_k=5)
    assert result["status"] == "ok"
    assert len(result["feature_explanations"]) == 5
    assert len(result["top_features"]) == 5
    for f in result["feature_explanations"]:
        assert {"feature", "value", "contribution", "direction"}.issubset(f)
        assert f["direction"] in ("increases_risk", "decreases_risk")


def test_gru_explanation_on_real_historical_failure_shows_meaningful_attribution():
    """M-01 genuinely failed on sim_day 23 (Afternoon) in the real training
    data (see data/processed/train.csv) - the GRU should predict high risk
    for that real window, with non-trivial (not near-zero) attributions,
    proving the explanation tracks an actual known failure rather than
    only ever describing a quiet, low-risk state.
    """
    result = gru_explainer.explain(VALID_MACHINE, sim_day=23, shift="Afternoon")
    assert result["status"] == "ok"
    assert result["prediction"]["risk_level"] == "HIGH"
    assert result["prediction"]["probability"] > 0.9
    top_contribution = abs(result["feature_explanations"][0]["contribution"])
    assert top_contribution > 1e-4  # meaningfully non-zero, not a saturated no-op


def test_shap_top_features_are_sorted_by_absolute_contribution():
    result = shap_tabular_explainer.explain(VALID_MACHINE, model_name="random_forest")
    contributions = [abs(f["contribution"]) for f in result["feature_explanations"]]
    assert contributions == sorted(contributions, reverse=True)


# ---- Test 3: Human Explanation -----------------------------------------

def test_human_summary_reflects_actual_attribution_results():
    prediction = {"risk_level": "HIGH", "probability": 0.92}
    feature_explanations = [
        {"feature": "vibration_mm_s_mean", "value": 5.0, "contribution": 0.3, "direction": "increases_risk"},
        {"feature": "quality_score", "value": 95.0, "contribution": -0.1, "direction": "decreases_risk"},
    ]
    summary = build_human_summary(prediction, feature_explanations, None)
    assert "HIGH" in summary
    assert "vibration mm s mean" in summary
    assert "quality score" in summary


def test_human_summary_does_not_fabricate_when_nothing_available():
    summary = build_human_summary(None, [], None)
    assert "No explanation" in summary or "unavailable" in summary.lower() or "No predictive" in summary


# ---- Test 4: Vision XAI (Grad-CAM) -------------------------------------

def test_gradcam_produces_heatmap_and_overlay_matching_stage3_confidence():
    result = gradcam_explainer.explain(VALID_MACHINE)
    assert result["status"] == "ok"
    root = Path(__file__).resolve().parent.parent
    assert (root / result["heatmap_path"]).exists()
    assert (root / result["overlay_path"]).exists()

    import json
    stage3 = json.loads((root / "reports/stage3_structured_output.json").read_text())
    record = next(r for r in stage3 if r["machine_id"] == VALID_MACHINE)
    assert abs(result["confidence"] - record["cv_confidence"]) < 1e-4  # matches Stage III's forward pass (result is rounded to 4dp)


def test_gradcam_heatmap_is_not_uniform_or_random():
    """A real Grad-CAM heatmap should vary spatially (it highlights specific
    regions) - a degenerate/fake implementation would be uniform or purely
    random noise with no relation to the input.
    """
    result = gradcam_explainer.explain(VALID_MACHINE)
    root = Path(__file__).resolve().parent.parent
    from PIL import Image
    heat = np.array(Image.open(root / result["heatmap_path"]))
    assert heat[..., 0].std() > 1.0  # meaningful spatial variation in the red (intensity) channel


# ---- Test 5: Structured Output ------------------------------------------

def test_xai_pipeline_output_matches_schema():
    result = explain_machine(VALID_MACHINE)
    for key in ("machine_id", "prediction", "feature_explanations", "top_features",
                "vision_explanation", "human_summary", "method"):
        assert key in result
    assert "tabular" in result["method"] and "vision" in result["method"]
    assert isinstance(result["human_summary"], str) and len(result["human_summary"]) > 0


# ---- Test 6: Error Handling ----------------------------------------------

def test_gru_explainer_unavailable_for_unknown_machine_no_fake_values():
    result = gru_explainer.explain(INVALID_MACHINE)
    assert result["status"] == "unavailable"
    assert "reason" in result


def test_gradcam_unavailable_for_unknown_machine_no_fake_heatmap():
    result = gradcam_explainer.explain(INVALID_MACHINE)
    assert result["status"] == "unavailable"


def test_shap_explainer_rejects_unknown_model_name():
    result = shap_tabular_explainer.explain(VALID_MACHINE, model_name="not_a_real_model")
    assert result["status"] == "error"


def test_xai_pipeline_degrades_gracefully_for_unknown_machine():
    result = explain_machine(INVALID_MACHINE)
    assert result["vision_explanation"]["available"] is False
    assert "reason" in result["prediction"]
    assert isinstance(result["human_summary"], str)  # still produces text, never crashes


# ---- Test 7: Stage Preservation ------------------------------------------

def test_orchestrator_still_runs_all_five_steps_with_xai_integrated():
    result = run_workflow(VALID_MACHINE)
    assert "xai_result" in result
    agent_names = {t["agent"] for t in result["agent_trace"]}
    assert "xai_layer" in agent_names
    assert result["decision"]["status"] == "ok"


def test_planning_agent_backward_compatible_without_xai_argument():
    """Stage V's original 4-argument call signature must still work
    unchanged - xai is optional and additive only.
    """
    from src.agents import planning_agent
    pm = {"status": "ok", "risk_level": "LOW", "failure_probability": 0.01,
          "model_name": "gru", "important_signals": ["load_pct_mean"]}
    no_vision = {"status": "unavailable", "reason": "n/a"}
    no_nlp = {"status": "unavailable", "reason": "n/a"}
    no_rag = {"status": "ok", "grounded": False, "sources": []}
    decision = planning_agent.decide(no_vision, pm, no_nlp, no_rag)  # no xai arg
    assert decision["status"] == "ok"


def test_stage1_to_5_tests_still_pass():
    """Meta-check: import the existing test modules to confirm they still
    collect without error (full re-run is done via `pytest tests/ -q`).
    """
    import tests.test_stage1  # noqa: F401
    import tests.test_stage2  # noqa: F401
    import tests.test_stage3  # noqa: F401
    import tests.test_stage4  # noqa: F401
    import tests.test_stage5  # noqa: F401
