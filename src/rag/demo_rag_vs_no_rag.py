"""Stage IV — required demonstration: RAG vs. unsupported generation.

Asks the same factory question two ways:
  1. WITHOUT RAG: generate_unsupported_answer() has no access to the
     knowledge base at all.
  2. WITH RAG: RAGPipeline.answer() retrieves evidence first, then
     generates a grounded, sourced answer.

Saves the side-by-side comparison to
reports/stage4_rag_vs_no_rag_demo.json.

Run:
    python -m src.rag.demo_rag_vs_no_rag
"""
from __future__ import annotations

import json

from config.config import REPORTS_DIR
from src.rag.generator import generate_unsupported_answer
from src.rag.rag_pipeline import RAGPipeline

DEMO_QUESTION = "What should the operator do if machine vibration becomes dangerously high?"


def run_demo(question: str = DEMO_QUESTION) -> dict:
    without_rag_answer = generate_unsupported_answer(question)

    pipeline = RAGPipeline()
    with_rag_result = pipeline.answer(question)

    return {
        "question": question,
        "without_rag": {
            "answer": without_rag_answer,
            "sources": [],
            "note": "Generated with no access to the factory knowledge base — cannot cite a specific document, threshold value, or required step.",
        },
        "with_rag": with_rag_result,
    }


def main() -> None:
    result = run_demo()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "stage4_rag_vs_no_rag_demo.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
