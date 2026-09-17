"""Stage V — Planning / Decision Agent.

Combines the Vision, Predictive Maintenance, NLP, and Knowledge/RAG results
into one structured recommendation. The recommendation and priority are
determined by explicit, documented rules over the actual upstream values
(never a fixed string) - this keeps the LLM out of the predictive/decision
path, matching the Stage V brief ("The LLM is NOT the predictive engine").

An LLM is used only to phrase the reasoning as readable prose when an API
key is configured (same optional-live-LLM / local-fallback pattern as
Stage IV's `src/rag/generator.py`); the recommendation and priority values
themselves are never touched by the LLM call.
"""
from __future__ import annotations

import os


def _build_reasoning(vision: dict, pm: dict, nlp: dict, rag: dict, xai: dict | None = None,
                      twin: dict | None = None) -> list[str]:
    reasoning = []

    if pm.get("status") == "ok":
        reasoning.append(
            f"Predictive maintenance model ({pm['model_name']}) estimates a "
            f"{pm['failure_probability']:.1%} failure probability for the next shift "
            f"(risk level: {pm['risk_level']}), driven most by: {', '.join(pm['important_signals'])}."
        )
    else:
        reasoning.append(f"Predictive maintenance result unavailable: {pm.get('reason', pm.get('status'))}.")

    if vision.get("status") == "ok":
        if vision["defect_detected"]:
            reasoning.append(
                f"Vision inspection flagged a defect (confidence {vision['confidence']:.1%}); "
                "defect type/severity were not classified by the Stage III model."
            )
        else:
            reasoning.append(f"Vision inspection found no defect (confidence {vision['confidence']:.1%}).")
    else:
        reasoning.append(f"Vision result unavailable: {vision.get('reason', vision.get('status'))}.")

    if nlp.get("status") == "ok":
        reasoning.append(
            f"Most recent maintenance note classified as '{nlp['incident_type']}' "
            f"(confidence {nlp['confidence']:.1%})."
        )
    else:
        reasoning.append(f"NLP result unavailable: {nlp.get('reason', nlp.get('status'))}.")

    if rag.get("status") == "ok" and rag.get("grounded"):
        doc_names = sorted({s["document"] for s in rag["sources"]})
        reasoning.append(f"Relevant SOP evidence retrieved from: {', '.join(doc_names)}.")
    elif rag.get("status") == "ok":
        reasoning.append("Knowledge base returned no sufficiently relevant evidence for the formulated question.")
    else:
        reasoning.append(f"Knowledge/RAG result unavailable: {rag.get('reason', rag.get('status'))}.")

    # Stage VI explainability evidence, when available - cites the actual
    # top attributed features, never an invented explanation.
    if xai and xai.get("top_features"):
        method = xai.get("method", {}).get("tabular", "the XAI layer")
        reasoning.append(
            f"Explainability analysis ({method}) attributes this prediction most to: "
            f"{', '.join(f.replace('_', ' ') for f in xai['top_features'][:3])}."
        )

    # Stage VII digital-twin evidence, when available - cites the actual
    # simulated recommendation, never an invented cost/scenario.
    if twin and twin.get("status") == "ok":
        reasoning.append(
            f"Digital twin what-if simulation over the next {twin['horizon_shifts']} shifts "
            f"favors '{twin['recommended_scenario'].replace('_', ' ')}' "
            f"({twin['recommendation_basis']})."
        )

    return reasoning


def _determine_priority_and_recommendation(vision: dict, pm: dict, nlp: dict, rag: dict) -> tuple[str, str]:
    """Explicit rules over actual upstream values - not a hard-coded string.
    Every branch below reads a real field from a real agent result.
    """
    risk = pm.get("risk_level") if pm.get("status") == "ok" else None
    defect = vision.get("defect_detected") if vision.get("status") == "ok" else None
    incident_type = nlp.get("incident_type") if nlp.get("status") == "ok" else None
    unsafe_incident = incident_type == "Safety"

    if unsafe_incident or risk == "HIGH":
        priority = "URGENT"
        if defect:
            recommendation = (
                "Stop the machine for immediate inspection. Both the predictive model's "
                f"{'HIGH' if risk == 'HIGH' else risk} risk estimate and a vision-confirmed "
                "defect point to the same machine; correlate the sensor anomaly with the "
                "visual finding before restarting."
            )
        else:
            recommendation = (
                "Schedule preventive maintenance before the next shift begins; do not wait "
                "for a further sensor escalation."
            )
    elif risk == "MEDIUM" or defect:
        priority = "ELEVATED"
        if defect and risk == "MEDIUM":
            recommendation = (
                "Increase inspection frequency this shift and flag the current production "
                "batch for quality review; the predictive model shows elevated (not yet high) risk."
            )
        elif defect:
            recommendation = "Flag the current production batch for quality review and inspect tooling."
        else:
            recommendation = "Increase monitoring frequency for this machine over the next shift."
    else:
        priority = "ROUTINE"
        recommendation = "Continue normal operation; monitor on the standard schedule."

    if rag.get("status") == "ok" and rag.get("grounded"):
        recommendation += " Follow the retrieved SOP guidance for the specific steps and thresholds."

    return priority, recommendation


def _try_live_llm_narrative(reasoning: list[str], recommendation: str) -> str | None:
    """Optional: ask a live LLM to phrase the reasoning as prose. Returns
    None on any failure (no key, no network, API error) so the caller falls
    back to the deterministic reasoning list - same pattern as Stage IV's
    `src/rag/generator.py::_try_live_llm`. The recommendation/priority
    values themselves are decided above and are never generated by the LLM.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import json
        import urllib.request

        prompt = (
            "Rewrite the following factory decision-support evidence as one concise "
            "paragraph for a human supervisor. Do not add any new facts, numbers, or "
            "recommendations beyond what is listed.\n\n"
            f"Evidence:\n- " + "\n- ".join(reasoning) + f"\n\nRecommendation already decided: {recommendation}"
        )
        body = json.dumps({
            "model": "claude-sonnet-4-6",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "content-type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        return "".join(block.get("text", "") for block in data.get("content", []))
    except Exception:
        return None


def decide(vision: dict, pm: dict, nlp: dict, rag: dict, xai: dict | None = None,
           twin: dict | None = None) -> dict:
    """`xai`/`twin` are optional and additive (Stage VI/VII integration) -
    omitting them reproduces Stage V's original behavior exactly; passing
    them only adds reasoning lines citing the actual explainability/
    simulation evidence. Priority and recommendation are unaffected by
    either - they remain decided purely from the four Stage V agent
    results, per the Stage V brief.
    """
    reasoning = _build_reasoning(vision, pm, nlp, rag, xai, twin)
    priority, recommendation = _determine_priority_and_recommendation(vision, pm, nlp, rag)

    narrative = _try_live_llm_narrative(reasoning, recommendation)

    return {
        "status": "ok",
        "recommendation": recommendation,
        "priority": priority,
        "reasoning": reasoning,
        "narrative": narrative,  # None unless a live LLM key was configured and reachable
        "used_live_llm": narrative is not None,
    }
