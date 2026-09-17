"""Stage V — Multi-agent orchestrator.

A lightweight, dependency-free orchestrator (no LangGraph): the project had
no orchestration framework already installed, and a 4-node fan-in workflow
is simple enough that adding a new heavy dependency isn't justified -
consistent with this project's existing "avoid unnecessary dependencies"
pattern from Stages II-IV. Each agent is called in sequence, its result is
placed into a shared state dict, and a trace entry is recorded regardless
of success or failure so the execution is fully visible.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.config import REPORTS_DIR
from src.agents import vision_agent, predictive_maintenance_agent, knowledge_agent, planning_agent
from src.agents.schemas import make_trace_entry
from src.xai.xai_pipeline import explain_machine
from src.digital_twin.simulator import run_scenarios


def run_workflow(machine_id: str) -> dict:
    trace = []
    state = {"factory_context": {"machine_id": machine_id, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}}

    # 1. Vision Agent
    try:
        vision_result = vision_agent.get_vision_result(machine_id)
        trace.append(make_trace_entry(
            "vision_agent", vision_result["status"], f"machine_id={machine_id}",
            _summarize(vision_result), vision_result.get("reason")))
    except Exception as exc:
        vision_result = {"status": "error", "reason": str(exc)}
        trace.append(make_trace_entry("vision_agent", "error", f"machine_id={machine_id}", "", str(exc)))
    state["vision_result"] = vision_result

    # 2. Predictive Maintenance Agent
    try:
        pm_result = predictive_maintenance_agent.get_prediction(machine_id)
        trace.append(make_trace_entry(
            "predictive_maintenance_agent", pm_result["status"], f"machine_id={machine_id}",
            _summarize(pm_result), pm_result.get("reason")))
    except Exception as exc:
        pm_result = {"status": "error", "reason": str(exc)}
        trace.append(make_trace_entry("predictive_maintenance_agent", "error", f"machine_id={machine_id}", "", str(exc)))
    state["predictive_maintenance_result"] = pm_result

    # 3. NLP info (reads Stage III's existing result - see vision_agent.get_nlp_result)
    try:
        nlp_result = vision_agent.get_nlp_result(machine_id)
        trace.append(make_trace_entry(
            "nlp_info", nlp_result["status"], f"machine_id={machine_id}",
            _summarize(nlp_result), nlp_result.get("reason")))
    except Exception as exc:
        nlp_result = {"status": "error", "reason": str(exc)}
        trace.append(make_trace_entry("nlp_info", "error", f"machine_id={machine_id}", "", str(exc)))
    state["nlp_result"] = nlp_result

    # 3.5 Stage VI - Explainable AI. Explains *why* the Predictive
    # Maintenance Agent's prediction and the Vision Agent's defect call came
    # out the way they did (Stage VI, additive: state gets a new
    # "xai_result" key, no existing Stage V agent's own logic is touched).
    try:
        xai_result = explain_machine(machine_id)
        trace.append(make_trace_entry(
            "xai_layer", "ok" if xai_result.get("feature_explanations") or xai_result.get("vision_explanation", {}).get("available") else "unavailable",
            f"machine_id={machine_id}",
            f"top_features={xai_result.get('top_features')}, "
            f"vision_available={xai_result.get('vision_explanation', {}).get('available')}"))
    except Exception as exc:
        xai_result = {"machine_id": machine_id, "human_summary": None, "reason": str(exc)}
        trace.append(make_trace_entry("xai_layer", "error", f"machine_id={machine_id}", "", str(exc)))
    state["xai_result"] = xai_result

    # 3.6 Stage VII - Digital Twin / What-If Simulation. Quantifies the
    # continue/stop-for-maintenance/reduce-load trade-off for this machine
    # (additive: new "digital_twin_result" state key only).
    try:
        twin_result = run_scenarios(machine_id)
        trace.append(make_trace_entry(
            "digital_twin", twin_result.get("status", "unavailable"), f"machine_id={machine_id}",
            f"recommended_scenario={twin_result.get('recommended_scenario')}", twin_result.get("reason")))
    except Exception as exc:
        twin_result = {"status": "error", "reason": str(exc)}
        trace.append(make_trace_entry("digital_twin", "error", f"machine_id={machine_id}", "", str(exc)))
    state["digital_twin_result"] = twin_result

    # 4. Knowledge / RAG Agent - question formulated from the actual upstream findings
    question = knowledge_agent.formulate_query(vision_result, pm_result)
    try:
        rag_result = knowledge_agent.get_knowledge(question, machine_id)
        trace.append(make_trace_entry(
            "knowledge_rag_agent", rag_result["status"], f"question={question!r}",
            _summarize(rag_result), rag_result.get("reason")))
    except Exception as exc:
        rag_result = {"status": "error", "reason": str(exc)}
        trace.append(make_trace_entry("knowledge_rag_agent", "error", f"question={question!r}", "", str(exc)))
    state["rag_result"] = rag_result

    # 5. Planning / Decision Agent - combines everything above (xai_result
    # is passed in additively; see planning_agent.decide's xai parameter)
    try:
        decision = planning_agent.decide(vision_result, pm_result, nlp_result, rag_result, xai_result, twin_result)
        trace.append(make_trace_entry(
            "planning_decision_agent", decision["status"],
            "vision+predictive_maintenance+nlp+rag results",
            f"priority={decision['priority']}, recommendation={decision['recommendation'][:80]}..."))
    except Exception as exc:
        decision = {"status": "error", "reason": str(exc)}
        trace.append(make_trace_entry("planning_decision_agent", "error",
                                       "vision+predictive_maintenance+nlp+rag results", "", str(exc)))
    state["decision"] = decision
    state["agent_trace"] = trace
    return state


def _summarize(result: dict) -> str:
    if result.get("status") != "ok":
        return f"status={result.get('status')}"
    keys_of_interest = [k for k in result if k not in ("status", "source")][:4]
    return ", ".join(f"{k}={result[k]}" for k in keys_of_interest)


def run_and_save_demo(machine_id: str) -> dict:
    result = run_workflow(machine_id)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "stage5_multi_agent_decision.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine-id", default="M-01")
    args = parser.parse_args()
    result = run_and_save_demo(args.machine_id)
    print(json.dumps(result, indent=2, default=str))
