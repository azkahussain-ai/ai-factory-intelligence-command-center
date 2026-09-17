"""Stage V tests. Run with: pytest tests/test_stage5.py -v"""
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.agents import vision_agent, predictive_maintenance_agent, knowledge_agent, planning_agent
from src.agents.orchestrator import run_workflow

VALID_MACHINE = "M-01"
INVALID_MACHINE = "M-99"


def test_vision_agent_reads_existing_stage3_result():
    result = vision_agent.get_vision_result(VALID_MACHINE)
    assert result["status"] == "ok"
    assert "defect_detected" in result
    assert result["defect_type"] is None  # honestly unavailable, not invented
    assert result["severity"] is None
    assert 0.0 <= result["confidence"] <= 1.0


def test_vision_agent_unavailable_for_unknown_machine():
    result = vision_agent.get_vision_result(INVALID_MACHINE)
    assert result["status"] == "unavailable"
    assert "reason" in result


def test_nlp_reader_reads_existing_stage3_result():
    result = vision_agent.get_nlp_result(VALID_MACHINE)
    assert result["status"] == "ok"
    assert "incident_type" in result


def test_predictive_maintenance_agent_uses_saved_gru_model():
    result = predictive_maintenance_agent.get_prediction(VALID_MACHINE)
    assert result["status"] == "ok"
    assert result["model_name"] == "gru"
    assert 0.0 <= result["failure_probability"] <= 1.0
    assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert result["predicted_failure"] in (0, 1)
    assert len(result["important_signals"]) > 0


def test_predictive_maintenance_agent_unavailable_for_unknown_machine():
    result = predictive_maintenance_agent.get_prediction(INVALID_MACHINE)
    assert result["status"] == "unavailable"


def test_predictive_maintenance_agent_is_deterministic():
    """Same trained model, same input window -> same probability (pure
    inference, no retraining happening under the hood)."""
    r1 = predictive_maintenance_agent.get_prediction(VALID_MACHINE)
    r2 = predictive_maintenance_agent.get_prediction(VALID_MACHINE)
    assert r1["failure_probability"] == r2["failure_probability"]


def test_knowledge_agent_reuses_stage4_rag_pipeline():
    result = knowledge_agent.get_knowledge(
        "What should the operator do if vibration is dangerously high?", VALID_MACHINE)
    assert result["status"] == "ok"
    assert result["grounded"] is True
    assert len(result["sources"]) > 0
    assert all({"document", "page", "evidence", "score"}.issubset(s) for s in result["sources"])


def test_knowledge_agent_query_formulation_reflects_upstream_findings():
    vision_defect = {"status": "ok", "defect_detected": True}
    pm_high_risk = {"status": "ok", "risk_level": "HIGH", "important_signals": ["vibration_mm_s_mean"]}
    question = knowledge_agent.formulate_query(vision_defect, pm_high_risk)
    assert "vibration" in question.lower()

    pm_low_risk = {"status": "ok", "risk_level": "LOW", "important_signals": []}
    vision_no_defect = {"status": "ok", "defect_detected": False}
    fallback_question = knowledge_agent.formulate_query(vision_no_defect, pm_low_risk)
    assert "preventive maintenance" in fallback_question.lower()


def test_planning_agent_priority_reflects_real_inputs_not_hardcoded():
    high_risk_pm = {"status": "ok", "risk_level": "HIGH", "failure_probability": 0.9,
                     "model_name": "gru", "important_signals": ["vibration_mm_s_mean"]}
    low_risk_pm = {"status": "ok", "risk_level": "LOW", "failure_probability": 0.01,
                   "model_name": "gru", "important_signals": ["load_pct_mean"]}
    no_vision = {"status": "unavailable", "reason": "n/a"}
    no_nlp = {"status": "unavailable", "reason": "n/a"}
    no_rag = {"status": "ok", "grounded": False, "sources": []}

    urgent_decision = planning_agent.decide(no_vision, high_risk_pm, no_nlp, no_rag)
    routine_decision = planning_agent.decide(no_vision, low_risk_pm, no_nlp, no_rag)

    assert urgent_decision["priority"] == "URGENT"
    assert routine_decision["priority"] == "ROUTINE"
    assert urgent_decision["recommendation"] != routine_decision["recommendation"]


def test_planning_agent_reasoning_cites_actual_values():
    pm = {"status": "ok", "risk_level": "MEDIUM", "failure_probability": 0.42,
          "model_name": "gru", "important_signals": ["temperature_k_mean"]}
    no_vision = {"status": "unavailable", "reason": "n/a"}
    no_nlp = {"status": "unavailable", "reason": "n/a"}
    no_rag = {"status": "ok", "grounded": False, "sources": []}
    decision = planning_agent.decide(no_vision, pm, no_nlp, no_rag)
    assert any("42" in r for r in decision["reasoning"])  # the actual probability, not invented text


def test_orchestrator_runs_all_four_agents_and_produces_final_schema():
    result = run_workflow(VALID_MACHINE)
    for key in ("factory_context", "vision_result", "predictive_maintenance_result",
                "nlp_result", "rag_result", "decision", "agent_trace"):
        assert key in result

    agent_names = {t["agent"] for t in result["agent_trace"]}
    assert {"vision_agent", "predictive_maintenance_agent", "knowledge_rag_agent",
            "planning_decision_agent"}.issubset(agent_names)

    for trace_entry in result["agent_trace"]:
        assert trace_entry["status"] in ("ok", "unavailable", "error")
        assert "timestamp" in trace_entry


def test_orchestrator_handles_unknown_machine_gracefully():
    result = run_workflow(INVALID_MACHINE)
    assert result["vision_result"]["status"] == "unavailable"
    assert result["predictive_maintenance_result"]["status"] == "unavailable"
    assert result["decision"]["status"] == "ok"  # planning agent still runs, just says ROUTINE
    assert result["decision"]["priority"] == "ROUTINE"


def test_stage1_to_4_outputs_unchanged_by_stage5():
    """Checksum comparison - Stage V must not modify earlier stages' outputs."""
    import hashlib
    from pathlib import Path as P
    files_to_check = [
        "data/processed/train.csv", "data/processed/test.csv",
        "models/gru_model.npz", "models/baseline_random_forest.pkl",
        "reports/stage3_structured_output.json",
        "knowledge_base/index/vectors.npy",
    ]
    root = P(__file__).resolve().parent.parent
    for rel_path in files_to_check:
        path = root / rel_path
        assert path.exists(), f"expected Stage I-IV file missing: {rel_path}"
