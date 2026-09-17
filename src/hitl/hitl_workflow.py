"""Stage VIII — Human-in-the-Loop.

Wraps an existing AI recommendation (Stage V's `orchestrator.run_workflow`
output, which already carries Stage VI's XAI and Stage VII's digital-twin
evidence via the additive integration built in those stages) in a human
decision step. The human supervisor's choice - APPROVE, REJECT, or MODIFY -
is recorded alongside the untouched original AI recommendation; nothing
about the AI output is ever rewritten by a human decision.
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.hitl.audit_store import append_record

VALID_DECISIONS = {"APPROVE", "REJECT", "MODIFY"}


class HumanDecisionError(ValueError):
    pass


def _extract_ai_recommendation(ai_output: dict) -> dict:
    """Pull out exactly the fields that make up "the AI recommendation" from
    a real orchestrator output - never invented, always read from the
    actual workflow result passed in.
    """
    decision = ai_output.get("decision", {})
    context = ai_output.get("factory_context", {})
    return {
        "machine_id": context.get("machine_id"),
        "generated_at": context.get("generated_at"),
        "recommendation": decision.get("recommendation"),
        "priority": decision.get("priority"),
        "reasoning": decision.get("reasoning", []),
        "narrative": decision.get("narrative"),
    }


def _extract_evidence(ai_output: dict) -> dict:
    """The supporting evidence behind the recommendation - vision, predictive
    maintenance, XAI, digital twin, and RAG results - preserved alongside
    the recommendation so the audit record is self-contained and doesn't
    require re-running the whole pipeline to understand why the AI said
    what it said.
    """
    return {
        "vision_result": ai_output.get("vision_result"),
        "predictive_maintenance_result": ai_output.get("predictive_maintenance_result"),
        "nlp_result": ai_output.get("nlp_result"),
        "xai_result": ai_output.get("xai_result"),
        "digital_twin_result": ai_output.get("digital_twin_result"),
        "rag_result": ai_output.get("rag_result"),
    }


def submit_decision(
    ai_output: dict,
    human_decision: str,
    comment: str | None = None,
    modified_action: str | None = None,
    supervisor: str | None = None,
) -> dict:
    """Record a human supervisor's decision on a real AI recommendation.

    `ai_output` must be an actual result of `src.agents.orchestrator.run_workflow`
    (or an equivalent structure with `factory_context`/`decision` keys) -
    this function never generates or guesses a recommendation itself.

    Raises HumanDecisionError for an invalid decision value, a REJECT with
    no comment, or a MODIFY with no modified_action/comment - the workflow
    requires a reason for anything other than a plain approval.
    """
    decision_upper = (human_decision or "").upper()
    if decision_upper not in VALID_DECISIONS:
        raise HumanDecisionError(
            f"human_decision must be one of {sorted(VALID_DECISIONS)}, got {human_decision!r}")

    if decision_upper == "REJECT" and not comment:
        raise HumanDecisionError("REJECT requires a reason/comment")
    if decision_upper == "MODIFY":
        if not modified_action:
            raise HumanDecisionError("MODIFY requires a modified_action")
        if not comment:
            raise HumanDecisionError("MODIFY requires a reason/comment")

    original_recommendation = _extract_ai_recommendation(ai_output)
    if original_recommendation["recommendation"] is None:
        raise HumanDecisionError(
            "ai_output has no decision.recommendation - is this a real orchestrator result?")

    record = {
        "decision_id": uuid.uuid4().hex,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "original_ai_recommendation": original_recommendation,
        "ai_evidence": _extract_evidence(ai_output),
        "human_decision": decision_upper,
        "modified_action": modified_action if decision_upper == "MODIFY" else None,
        "human_comment": comment,
        "supervisor": supervisor,
        "final_action": (
            modified_action if decision_upper == "MODIFY"
            else original_recommendation["recommendation"] if decision_upper == "APPROVE"
            else None  # REJECT -> no action taken
        ),
    }

    append_record(record)
    return record
