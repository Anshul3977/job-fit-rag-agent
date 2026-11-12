"""Vector stores. Chroma locally, pgvector when Postgres is selected."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sqlalchemy import text
from sqlalchemy.engine import Engine

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


@dataclass
class VectorHit:
    chunk_id: str
    section: str
    score: float


class VectorStore(Protocol):
    name: str

    def upsert(self, items: list[dict]) -> None: ...

    def query(self, resume_id: str, embedding: list[float], k: int) -> list[VectorHit]: ...

    def reset(self) -> None: ...

    def ping(self) -> None: ...


class MemoryVectorStore:
    """Brute-force cosine store. Used by unit tests that do not need Chroma."""

    name = "memory"

    def __init__(self) -> None:
        self._items: list[dict] = []

    def upsert(self, items: list[dict]) -> None:
        by_id = {item["chunk_id"]: item for item in self._items}
        for item in items:
            by_id[item["chunk_id"]] = item
        self._items = list(by_id.values())

    def query(self, resume_id: str, embedding: list[float], k: int) -> list[VectorHit]:
        import numpy as np

        query_vec = np.array(embedding, dtype=np.float64)
        norm = float(np.linalg.norm(query_vec))
        if norm == 0:
            return []
        query_vec = query_vec / norm
        scored: list[VectorHit] = []
        for item in self._items:
            if item["resume_id"] != resume_id:
                continue
            vec = np.array(item["embedding"], dtype=np.float64)
            denom = float(np.linalg.norm(vec))
            if denom == 0:
                continue
            score = float(np.dot(query_vec, vec / denom))
            scored.append(VectorHit(chunk_id=item["chunk_id"], section=item["section"], score=score))
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:k]

    def reset(self) -> None:
        self._items = []

    def ping(self) -> None:
        return None


class ChromaVectorStore:
    name = "chroma"

    def __init__(self, path: str, collection_name: str = "resume_chunks") -> None:
        self.path = path
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def _open(self):
        if self._collection is not None:
            return self._collection
        import chromadb

        Path(self.path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.path)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def upsert(self, items: list[dict]) -> None:
        if not items:
            return
        collection = self._open()
        collection.upsert(
            ids=[item["chunk_id"] for item in items],
            embeddings=[item["embedding"] for item in items],
            documents=[item.get("text") or "" for item in items],
            metadatas=[
                {
                    "resume_id": item["resume_id"],
                    "section": item["section"],
                    "chunk_id": item["chunk_id"],
                }
                for item in items
            ],
        )

    def query(self, resume_id: str, embedding: list[float], k: int) -> list[VectorHit]:
        collection = self._open()
        count = collection.count()
        if count == 0 or k <= 0:
            return []
        try:
            result = collection.query(
                query_embeddings=[embedding],
                n_results=min(k, count),
                where={"resume_id": resume_id},
                include=["metadatas", "distances"],
            )
        except Exception:
            return []
        ids = (result.get("ids") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits: list[VectorHit] = []
        for chunk_id, meta, distance in zip(ids, metas, distances):
            # Cosine distance in Chroma is 1 - cosine similarity.
            similarity = 1.0 - float(distance)
            hits.append(
                VectorHit(
                    chunk_id=str(chunk_id),
                    section=str((meta or {}).get("section") or "preamble"),
                    score=max(-1.0, min(1.0, similarity)),
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits

    def reset(self) -> None:
        if self._client is None:
            self._open()
        assert self._client is not None
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = None

    def ping(self) -> None:
        self._open().count()


class PgVectorStore:
    name = "pgvector"

    def __init__(self, engine: Engine, dim: int = 384) -> None:
        self.engine = engine
        self.dim = dim
        self._ready = False

    def setup(self) -> None:
        if self._ready:
            return
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS chunk_vectors (
                        chunk_id TEXT PRIMARY KEY,
                        resume_id TEXT NOT NULL,
                        section TEXT NOT NULL,
                        embedding vector({self.dim})
                    )
                    """
                )
            )
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS chunk_vectors_resume_idx ON chunk_vectors (resume_id)")
            )
        self._ready = True

    def upsert(self, items: list[dict]) -> None:
        if not items:
            return
        self.setup()
        statement = text(
            """
            INSERT INTO chunk_vectors (chunk_id, resume_id, section, embedding)
            VALUES (:chunk_id, :resume_id, :section, CAST(:embedding AS vector))
            ON CONFLICT (chunk_id) DO UPDATE
            SET resume_id = EXCLUDED.resume_id,
                section = EXCLUDED.section,
                embedding = EXCLUDED.embedding
            """
        )
        with self.engine.begin() as conn:
            for item in items:
                if len(item["embedding"]) != self.dim:
                    raise ValueError(
                        f"Expected {self.dim}-d embeddings for pgvector, got {len(item['embedding'])}."
                    )
                literal = "[" + ",".join(f"{value:.7f}" for value in item["embedding"]) + "]"
                conn.execute(
                    statement,
                    {
                        "chunk_id": item["chunk_id"],
                        "resume_id": item["resume_id"],
                        "section": item["section"],
                        "embedding": literal,
                    },
                )

    def query(self, resume_id: str, embedding: list[float], k: int) -> list[VectorHit]:
        self.setup()
        literal = "[" + ",".join(f"{value:.7f}" for value in embedding) + "]"
        statement = text(
            """
            SELECT chunk_id, section,
                   1 - (embedding <=> CAST(:embedding AS vector)) AS score
            FROM chunk_vectors
            WHERE resume_id = :resume_id
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :k
            """
        )
        with self.engine.connect() as conn:
            rows = conn.execute(
                statement,
                {"embedding": literal, "resume_id": resume_id, "k": k},
            ).all()
        return [VectorHit(chunk_id=row.chunk_id, section=row.section, score=float(row.score)) for row in rows]

    def reset(self) -> None:
        self.setup()
        with self.engine.begin() as conn:
            conn.execute(text("DELETE FROM chunk_vectors"))

    def ping(self) -> None:
        self.setup()
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1 FROM chunk_vectors LIMIT 1"))


def build_vector_store(backend: str, chroma_dir: str, engine: Engine) -> VectorStore:
    if backend == "memory":
        return MemoryVectorStore()
    if backend == "pgvector":
        return PgVectorStore(engine)
    if backend == "chroma":
        return ChromaVectorStore(chroma_dir)
    raise ValueError(f"Unknown vector backend: {backend}")
