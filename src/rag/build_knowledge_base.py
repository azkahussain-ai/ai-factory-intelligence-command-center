"""Stage IV — build the RAG index from the knowledge base PDFs.

Run:
    python -m src.rag.build_knowledge_base
"""
from __future__ import annotations

import json

from config.config import REPORTS_DIR
from src.rag.build_knowledge_base_pdfs import KB_DIR
from src.rag.rag_pipeline import build_index


def main() -> None:
    if not any(KB_DIR.glob("*.pdf")):
        raise SystemExit(
            f"No PDFs found in {KB_DIR}. Run "
            f"'python -m src.rag.build_knowledge_base_pdfs' first."
        )

    summary = build_index()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "stage4_ingestion_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
