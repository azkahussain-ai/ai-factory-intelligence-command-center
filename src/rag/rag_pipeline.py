"""Stage IV — end-to-end RAG pipeline orchestrator.

Documents -> extraction -> chunking -> embeddings -> vector store ->
retrieval -> generation -> structured, sourced answer.
"""
from __future__ import annotations

from pathlib import Path

from config.config import PROJECT_ROOT
from src.rag.chunking import chunk_pages
from src.rag.context_builder import format_machine_context, load_machine_context
from src.rag.document_loader import load_knowledge_base_pdfs
from src.rag.embeddings import TfidfEmbedder
from src.rag.generator import generate_grounded_answer
from src.rag.retriever import Retriever
from src.rag.vector_store import NumpyVectorStore, VectorStoreError

INDEX_DIR = PROJECT_ROOT / "knowledge_base" / "index"
EMBEDDER_PATH = INDEX_DIR / "tfidf_vectorizer.joblib"


def build_index(index_dir: Path = INDEX_DIR) -> dict:
    """Run the full ingestion pipeline and persist the index to disk.

    Returns a small summary dict (document count, chunk count) for logging
    / the Stage IV ingestion report — never fabricated, always the actual
    counts from this run.
    """
    pages = load_knowledge_base_pdfs()
    chunks = chunk_pages(pages)
    if not chunks:
        raise VectorStoreError("Chunking produced zero chunks from the knowledge base.")

    embedder = TfidfEmbedder()
    vectors = embedder.fit_transform([c.text for c in chunks])

    store = NumpyVectorStore()
    store.build(vectors, chunks)

    index_dir.mkdir(parents=True, exist_ok=True)
    store.save(index_dir)
    embedder.save(index_dir / "tfidf_vectorizer.joblib")

    documents = sorted({p.document for p in pages})
    return {
        "documents_ingested": documents,
        "pages_extracted": len(pages),
        "chunks_created": len(chunks),
    }


def load_pipeline(index_dir: Path = INDEX_DIR) -> Retriever:
    store = NumpyVectorStore.load(index_dir)
    embedder = TfidfEmbedder.load(index_dir / "tfidf_vectorizer.joblib")
    return Retriever(embedder, store)


class RAGPipeline:
    def __init__(self, index_dir: Path = INDEX_DIR):
        self._index_dir = index_dir
        self._retriever: Retriever | None = None

    def ensure_index(self) -> None:
        if not (self._index_dir / "vectors.npy").exists():
            build_index(self._index_dir)
        if self._retriever is None:
            self._retriever = load_pipeline(self._index_dir)

    def answer(self, question: str, machine_id: str | None = None, top_k: int = 4) -> dict:
        """Return the Stage IV structured RAG result.

        {
          "question": ...,
          "answer": ...,
          "grounded": bool,
          "used_live_llm": bool,
          "machine_context": str | None,
          "sources": [{"document", "page", "section", "evidence", "score"}]
        }
        """
        self.ensure_index()
        evidence = self._retriever.retrieve(question, top_k=top_k)
        generation = generate_grounded_answer(question, evidence)

        machine_context = None
        if machine_id:
            record = load_machine_context(machine_id)
            machine_context = format_machine_context(record)

        return {
            "question": question,
            "answer": generation["answer"],
            "grounded": len(evidence) > 0,
            "used_live_llm": generation["used_live_llm"],
            "machine_context": machine_context,
            "sources": [
                {
                    "document": e.document,
                    "page": e.page,
                    "section": e.section,
                    "evidence": e.evidence,
                    "score": round(e.score, 4),
                }
                for e in evidence
            ],
        }
