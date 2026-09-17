"""Stage IV — semantic retrieval."""
from __future__ import annotations

from dataclasses import dataclass

from src.rag.chunking import Chunk
from src.rag.embeddings import TfidfEmbedder
from src.rag.vector_store import NumpyVectorStore


@dataclass
class RetrievedEvidence:
    document: str
    page: int
    section: str | None
    evidence: str
    score: float


class Retriever:
    def __init__(self, embedder: TfidfEmbedder, store: NumpyVectorStore):
        self._embedder = embedder
        self._store = store

    def retrieve(self, query: str, top_k: int = 4, min_score: float = 0.05) -> list[RetrievedEvidence]:
        """Embed the query and return the top-k most relevant chunks.

        min_score filters out near-zero-similarity matches (i.e. the query
        shares essentially no vocabulary with the knowledge base) so the
        pipeline can honestly report "no relevant evidence" instead of
        returning an unrelated chunk just to fill top_k.
        """
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        query_vector = self._embedder.embed_query(query)
        results = self._store.search(query_vector, top_k=top_k)
        evidence = [
            RetrievedEvidence(
                document=chunk.document,
                page=chunk.page,
                section=chunk.section,
                evidence=chunk.text,
                score=score,
            )
            for chunk, score in results
            if score >= min_score
        ]
        return evidence
