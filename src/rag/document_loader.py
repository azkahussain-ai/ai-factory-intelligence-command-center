"""Stage IV — PDF document ingestion.

Reads every PDF in ``knowledge_base/`` and returns per-page text with
traceable metadata (document name, page number). Section titles are
attached later in ``chunking.py`` once the text is split into chunks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from config.config import PROJECT_ROOT

KB_DIR = PROJECT_ROOT / "knowledge_base"


@dataclass
class PageText:
    document: str      # PDF file name, e.g. "machine_operation_manual.pdf"
    page: int           # 1-indexed page number
    text: str


class DocumentLoadError(RuntimeError):
    """Raised when the knowledge base cannot be read at all."""


def load_knowledge_base_pdfs(kb_dir: Path = KB_DIR) -> list[PageText]:
    """Extract per-page text from every PDF in ``kb_dir``.

    Raises DocumentLoadError if the directory is missing or contains no
    PDFs, and skips (with a logged warning, not a silent drop) any PDF that
    fails to extract text, so a single corrupt file doesn't take down the
    whole knowledge base.
    """
    if not kb_dir.exists():
        raise DocumentLoadError(
            f"Knowledge base directory not found: {kb_dir}. Run "
            f"'python -m src.rag.build_knowledge_base_pdfs' first."
        )

    pdf_paths = sorted(kb_dir.glob("*.pdf"))
    if not pdf_paths:
        raise DocumentLoadError(
            f"No PDF documents found in {kb_dir}. Run "
            f"'python -m src.rag.build_knowledge_base_pdfs' first."
        )

    pages: list[PageText] = []
    for pdf_path in pdf_paths:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if len(pdf.pages) == 0:
                    print(f"[document_loader] WARNING: {pdf_path.name} has 0 pages, skipping.")
                    continue
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if not text.strip():
                        print(f"[document_loader] WARNING: {pdf_path.name} page {i} extracted empty text.")
                        continue
                    pages.append(PageText(document=pdf_path.name, page=i, text=text))
        except Exception as exc:  # noqa: BLE001 - deliberately broad, this is ingestion-time I/O
            print(f"[document_loader] ERROR: failed to read {pdf_path.name}: {exc}")
            continue

    if not pages:
        raise DocumentLoadError(
            "Every PDF in the knowledge base failed to extract usable text."
        )
    return pages
