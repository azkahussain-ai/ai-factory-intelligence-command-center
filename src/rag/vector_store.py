"""Stage IV — vector store.

Why a from-scratch NumPy store instead of FAISS/Chroma
--------------------------------------------------------
Neither ``faiss-cpu`` nor ``chromadb`` can be installed in this sandbox
(no PyPI access on the network egress allowlist — verified before writing
this file). Per the Stage IV brief ("choose the simplest reliable option
compatible with the existing project... do not introduce unnecessary
infrastructure"), this implements exactly the interface Stage IV needs
(store embeddings + metadata, similarity search, persist/load locally)
using NumPy cosine similarity and a JSON sidecar for metadata. For a
knowledge base of this size (tens of chunks), brute-force cosine similarity
is exact and fast — no approximate-nearest-neighbor index is needed.

This mirrors the project's existing, documented pattern of substituting a
from-scratch implementation when the "standard" library is not installable
(see src/deep_learning/gru_numpy.py and src/computer_vision/cnn_numpy.py
from Stage II/III).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.rag.chunking import Chunk


class VectorStoreError(RuntimeError):
    pass


class NumpyVectorStore:
    def __init__(self):
        self._vectors: np.ndarray | None = None
        self._chunks: list[Chunk] = []

    def build(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if vectors.shape[0] != len(chunks):
            raise VectorStoreError(
                f"Vector count ({vectors.shape[0]}) does not match chunk count ({len(chunks)})."
            )
        # L2-normalize once at build time so search is a plain dot product.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vectors = vectors / norms
        self._chunks = chunks

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[tuple[Chunk, float]]:
        if self._vectors is None or len(self._chunks) == 0:
            raise VectorStoreError("Vector store is empty — call build() or load() first.")
        norm = np.linalg.norm(query_vector)
        if norm == 0:
            # Query shares no vocabulary at all with the knowledge base.
            return []
        q = query_vector / norm
        scores = self._vectors @ q
        top_k = min(top_k, len(self._chunks))
        top_indices = np.argsort(-scores)[:top_k]
        return [(self._chunks[i], float(scores[i])) for i in top_indices]

    def save(self, dir_path: Path) -> None:
        if self._vectors is None:
            raise VectorStoreError("Nothing to save — build the store first.")
        dir_path.mkdir(parents=True, exist_ok=True)
        np.save(dir_path / "vectors.npy", self._vectors)
        metadata = [asdict(c) for c in self._chunks]
        (dir_path / "chunks.json").write_text(json.dumps(metadata, indent=2))

    @classmethod
    def load(cls, dir_path: Path) -> "NumpyVectorStore":
        vectors_path = dir_path / "vectors.npy"
        chunks_path = dir_path / "chunks.json"
        if not vectors_path.exists() or not chunks_path.exists():
            raise VectorStoreError(
                f"No persisted index found at {dir_path}. Run "
                f"'python -m src.rag.build_knowledge_base' first."
            )
        instance = cls()
        instance._vectors = np.load(vectors_path)
        raw_chunks = json.loads(chunks_path.read_text())
        instance._chunks = [Chunk(**c) for c in raw_chunks]
        return instance

    def __len__(self) -> int:
        return len(self._chunks)
