"""Stage V — Knowledge / RAG Agent.

Thin wrapper around Stage IV's existing `RAGPipeline` - reuses the same
vector store, retriever, and generator built in Stage IV. Does not create
a second index or a second embedding/vector-store pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.rag.rag_pipeline import RAGPipeline
from src.agents.schemas import error_result

_pipeline = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def formulate_query(vision_result: dict, pm_result: dict) -> str:
    """Build a knowledge query from the other agents' actual findings -
    never a fixed string, so the question genuinely reflects the current
    situation.
    """
    parts = []
    if pm_result.get("status") == "ok" and pm_result["risk_level"] in ("HIGH", "MEDIUM"):
        signal = pm_result["important_signals"][0] if pm_result.get("important_signals") else "sensor readings"
        parts.append(f"What should the operator do if {signal.replace('_', ' ')} becomes abnormal or dangerously high?")
    if vision_result.get("status") == "ok" and vision_result.get("defect_detected"):
        parts.append("What is the recommended procedure when a product defect is detected during inspection?")
    if not parts:
        parts.append("What is the standard preventive maintenance procedure for this machine?")
    return parts[0]  # the RAG pipeline answers one question per call


def get_knowledge(question: str, machine_id: str) -> dict:
    try:
        pipeline = _get_pipeline()
        result = pipeline.answer(question, machine_id=machine_id, top_k=4)
    except Exception as exc:
        return error_result(f"RAG pipeline failed: {exc}")

    return {
        "status": "ok",
        "question": result["question"],
        "answer": result["answer"],
        "grounded": result["grounded"],
        "used_live_llm": result["used_live_llm"],
        "sources": result["sources"],
        "source": "src/rag/rag_pipeline.py (existing Stage IV RAG pipeline)",
    }
