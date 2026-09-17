"""Stage IV — chunking with traceable metadata.

Splits each page's text into paragraph-based chunks and attaches the
section heading that precedes them (when detectable), so every chunk can be
cited back to a document name, page number, and section title.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.rag.document_loader import PageText

# A "Section N: Title" or "Section N" heading line, as used consistently
# across all three generated knowledge-base PDFs.
_SECTION_HEADING_RE = re.compile(r"^Section\s+\d+[:\-]?\s*.*$")

# Target chunk size in words. Small enough to keep retrieval precise (each
# chunk maps to one procedural idea), large enough to keep context intact
# for the generator.
CHUNK_WORDS = 90
CHUNK_OVERLAP_WORDS = 20


@dataclass
class Chunk:
    chunk_id: str
    document: str
    page: int
    section: str | None
    text: str
    metadata: dict = field(default_factory=dict)


def _split_into_paragraphs(text: str) -> list[str]:
    # The generated PDFs render each Paragraph() as its own line in
    # extracted text; treat blank-free single lines as paragraph units and
    # re-merge on sentence boundaries for chunking below.
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return lines


def _current_section(lines: list[str], up_to_index: int) -> str | None:
    for line in reversed(lines[: up_to_index + 1]):
        if _SECTION_HEADING_RE.match(line):
            return line
    return None


def chunk_pages(pages: list[PageText]) -> list[Chunk]:
    """Turn per-page text into overlapping word-count-bounded chunks.

    Each chunk keeps the nearest preceding "Section ..." heading on its own
    page as its section label (None if the page has no heading before that
    point, e.g. a title page).
    """
    chunks: list[Chunk] = []
    chunk_counter = 0

    for page in pages:
        lines = _split_into_paragraphs(page.text)
        # Build a flat word stream but remember, for each word, which line
        # index it came from, so we can look up the nearest section heading.
        word_stream: list[tuple[str, int]] = []
        for line_idx, line in enumerate(lines):
            for word in line.split():
                word_stream.append((word, line_idx))

        if not word_stream:
            continue

        start = 0
        while start < len(word_stream):
            end = min(start + CHUNK_WORDS, len(word_stream))
            window = word_stream[start:end]
            chunk_text = " ".join(w for w, _ in window)
            last_line_idx = window[-1][1]
            section = _current_section(lines, last_line_idx)

            chunk_counter += 1
            chunks.append(Chunk(
                chunk_id=f"chunk_{chunk_counter:04d}",
                document=page.document,
                page=page.page,
                section=section,
                text=chunk_text,
                metadata={"word_count": len(window)},
            ))

            if end == len(word_stream):
                break
            start = end - CHUNK_OVERLAP_WORDS

    return chunks
