"""Process-wide embedder and vector store."""

from __future__ import annotations

from app.config import get_settings
from app.db import get_engine
from app.retrieval.embeddings import Embedder, build_embedder
from app.retrieval.store import VectorStore, build_vector_store

_embedder: Embedder | None = None
_store: VectorStore | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        settings = get_settings()
        _embedder = build_embedder(
            settings.embedding_backend,
            settings.embedding_model,
            settings.embedding_cache_dir,
        )
    return _embedder


def get_store() -> VectorStore:
    global _store
    if _store is None:
        settings = get_settings()
        _store = build_vector_store(
            settings.resolved_vector_backend,
            settings.chroma_dir,
            get_engine(),
        )
    return _store


def reset_runtime() -> None:
    global _embedder, _store
    _embedder = None
    _store = None
