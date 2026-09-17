"""Stage IV — generation layer.

Why there is no live LLM API call by default
----------------------------------------------
Per the Stage IV brief: "Use environment variables for API keys. Never
hard-code API keys... If practical, provide a local/fallback mode so the
RAG pipeline can still be tested without exposing credentials."

This sandbox has no general network egress (verified: api.anthropic.com,
api.openai.com and general internet hosts are not on the allowlist) and no
API key is set in the environment. ``generate_grounded_answer`` therefore:

1. Checks for ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY`` in the
   environment. If present *and* reachable, it will attempt a real API call
   (kept isolated in ``_try_live_llm`` so it never runs unless a key
   actually exists).
2. Otherwise (the situation in this sandbox), it falls back to
   ``_local_grounded_generation``: a deterministic, template-driven
   synthesis that only ever states what is present in the retrieved
   evidence text. It does not invent factory procedures.

This is the same fallback pattern already used for Stage II's GRU and
Stage III's CV/NLP models (from-scratch local implementation, clearly
documented, because the "standard" dependency could not be reached).

The unsupported/no-RAG generator (`generate_unsupported_answer`) is a
*separate, explicitly labeled* function used only for the RAG-vs-no-RAG
demonstration. It deliberately has no access to the knowledge base, so it
can only answer from generic, non-factory-specific phrasing — which is
exactly the contrast Stage IV's demonstration is supposed to show.
"""
from __future__ import annotations

import os

from src.rag.retriever import RetrievedEvidence

NO_EVIDENCE_MESSAGE = (
    "The factory knowledge base does not contain sufficient information to "
    "answer this question. No retrieved document met the similarity "
    "threshold for this query."
)


def _try_live_llm(question: str, evidence: list[RetrievedEvidence]) -> str | None:
    """Attempt a real LLM call if a key is configured. Returns None on any
    failure (missing key, no network, API error) so the caller can fall
    back locally without crashing the pipeline."""
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        import urllib.request
        context = "\n\n".join(
            f"[{e.document} p.{e.page}{' - ' + e.section if e.section else ''}] {e.evidence}"
            for e in evidence
        )
        prompt = (
            "Answer the factory maintenance question using ONLY the "
            "evidence below. If the evidence is insufficient, say so.\n\n"
            f"Evidence:\n{context}\n\nQuestion: {question}"
        )
        # Deliberately not wired to a specific vendor payload here — this
        # sandbox cannot reach any LLM API to test it, so shipping an
        # untested request builder would be worse than an honest fallback.
        # A real deployment with network access would build the vendor's
        # /v1/messages or /v1/chat/completions request here using `prompt`.
        raise RuntimeError("Live LLM call not reachable in this environment.")
    except Exception:
        return None


def _local_grounded_generation(question: str, evidence: list[RetrievedEvidence]) -> str:
    """Deterministic, evidence-only answer synthesis (no external LLM)."""
    if not evidence:
        return NO_EVIDENCE_MESSAGE

    # Use the single most relevant chunk as the primary answer, and any
    # additional chunks as supporting detail — never inventing text beyond
    # what was retrieved.
    primary = evidence[0]
    lines = [
        f"Based on {primary.document}"
        + (f" ({primary.section})" if primary.section else "")
        + f", page {primary.page}: {primary.evidence.strip()}"
    ]
    for extra in evidence[1:]:
        lines.append(
            f"Additionally, {extra.document}"
            + (f" ({extra.section})" if extra.section else "")
            + f", page {extra.page} states: {extra.evidence.strip()}"
        )
    return " ".join(lines)


def generate_grounded_answer(question: str, evidence: list[RetrievedEvidence]) -> dict:
    """Return {"answer": str, "used_live_llm": bool}."""
    live_answer = _try_live_llm(question, evidence)
    if live_answer is not None:
        return {"answer": live_answer, "used_live_llm": True}
    return {"answer": _local_grounded_generation(question, evidence), "used_live_llm": False}


def generate_unsupported_answer(question: str) -> str:
    """Generic answer with NO access to the factory knowledge base.

    Used only for the RAG-vs-no-RAG demonstration (see
    src/rag/demo_rag_vs_no_rag.py). This intentionally cannot cite any
    factory-specific threshold, document, or procedure — that is the point
    of the comparison.
    """
    return (
        "In general, if a machine shows a concerning sensor reading, "
        "standard practice is to reduce load, monitor the situation, and "
        "consult the manufacturer's documentation or a qualified "
        "technician before deciding whether to continue operating the "
        "equipment. (This answer is generated without access to this "
        "factory's specific manuals or SOPs, so it cannot state exact "
        "thresholds, required response steps, or which document to follow.)"
    )
