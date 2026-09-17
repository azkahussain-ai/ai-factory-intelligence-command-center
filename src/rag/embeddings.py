"""Stage IV — embedding model.

Why TF-IDF instead of a sentence-transformer
---------------------------------------------
This sandbox has no general network access (confirmed by inspection: PyPI,
Hugging Face, and general internet hosts are not on the egress allowlist —
the same constraint that made Stage II/III implement the GRU/CNN from
scratch instead of using TensorFlow/PyTorch). A sentence-transformer model
cannot be downloaded here, so this uses scikit-learn's TfidfVectorizer
(already a project dependency since Stage III's NLP component) as a
lightweight, fully local embedding model. This is a legitimate "practical
lightweight embedding model suitable for a normal student laptop" per the
Stage IV brief, not a placeholder — TF-IDF + cosine similarity is a
standard, well-understood retrieval baseline.

If this project is later run in an environment with network access, this
class can be swapped for a sentence-transformer embedder without changing
any other Stage IV file, since ``vector_store.py`` and ``retriever.py`` only
depend on the ``embed_texts`` / ``embed_query`` interface below.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    def __init__(self, max_features: int = 4096):
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True,
        )
        self._fitted = False

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        matrix = self._vectorizer.fit_transform(texts)
        self._fitted = True
        return matrix.toarray().astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder must be fit before embedding queries.")
        return self._vectorizer.transform([query]).toarray().astype(np.float32)[0]

    def save(self, path: Path) -> None:
        joblib.dump(self._vectorizer, path)

    @classmethod
    def load(cls, path: Path) -> "TfidfEmbedder":
        instance = cls()
        instance._vectorizer = joblib.load(path)
        instance._fitted = True
        return instance
