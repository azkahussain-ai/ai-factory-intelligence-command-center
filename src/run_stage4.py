"""Stage IV entrypoint — GenAI + RAG.

Runs the full Stage IV pipeline end to end and writes reports:
  1. Generate knowledge-base PDFs if missing.
  2. Build (or reuse) the vector index.
  3. Answer a machine-specific question, showing sourced evidence and
     Stage II/III context integration.
  4. Run the required RAG-vs-no-RAG demonstration.

Run:
    python -m src.run_stage4
"""
from __future__ import annotations

import json

from config.config import REPORTS_DIR
from src.rag.build_knowledge_base_pdfs import KB_DIR, main as build_pdfs
from src.rag.demo_rag_vs_no_rag import run_demo
from src.rag.rag_pipeline import RAGPipeline, build_index


def main() -> None:
    if not any(KB_DIR.glob("*.pdf")):
        print("No knowledge-base PDFs found — generating them now...")
        build_pdfs()

    pipeline = RAGPipeline()
    index_was_missing = not (pipeline._index_dir / "vectors.npy").exists()
    if index_was_missing:
        ingestion_summary = build_index()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "stage4_ingestion_summary.json").write_text(
            json.dumps(ingestion_summary, indent=2)
        )
        print(f"Ingested knowledge base: {ingestion_summary}")
    pipeline.ensure_index()

    sample_result = pipeline.answer(
        "What should the operator do if machine vibration becomes dangerously high?",
        machine_id="M-01",
    )

    demo_result = run_demo()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "stage4_structured_output.json").write_text(
        json.dumps(sample_result, indent=2)
    )
    (REPORTS_DIR / "stage4_rag_vs_no_rag_demo.json").write_text(
        json.dumps(demo_result, indent=2)
    )

    print("=== Stage IV sample RAG answer (machine M-01) ===")
    print(json.dumps(sample_result, indent=2))
    print("\nSaved reports/stage4_structured_output.json")
    print("Saved reports/stage4_rag_vs_no_rag_demo.json")


if __name__ == "__main__":
    main()
