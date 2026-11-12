"""Retrieve resume chunks for a query or a whole job post."""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.agent.lexicon import skill_in_text, skills_in
from app.errors import AppError
from app.ingest.chunking import SECTION_WEIGHTS, split_sections
from app.models import Chunk
from app.retrieval.embeddings import Embedder
from app.retrieval.store import VectorStore

logger = logging.getLogger("jobfit")


def _embed(embedder: Embedder, text: str) -> list[float]:
    try:
        vectors = embedder.embed([text])
    except AppError:
        raise
    except Exception as exc:
        logger.exception("embedding failed")
        raise AppError(
            503,
            "EMBEDDING_UNAVAILABLE",
            "The local embedding model could not be loaded.",
        ) from exc
    if not vectors:
        raise AppError(503, "EMBEDDING_UNAVAILABLE", "The embedder returned no vector.")
    return vectors[0]


def _hydrate(db: Session, hits: list, via: str | None = None) -> list[dict]:
    if not hits:
        return []
    ids = [hit.chunk_id for hit in hits]
    rows = db.query(Chunk).filter(Chunk.id.in_(ids)).all()
    by_id = {row.id: row for row in rows}
    hydrated: list[dict] = []
    for hit in hits:
        row = by_id.get(hit.chunk_id)
        if row is None:
            continue
        hydrated.append(
            {
                "chunk_id": row.id,
                "text": row.text,
                "section": row.section,
                "score": float(hit.score),
                "page": row.page,
                "char_start": row.char_start,
                "char_end": row.char_end,
                "via": via,
            }
        )
    return hydrated


def _lexical_backfill(db: Session, resume_id: str, query: str, already: list[dict]) -> list[dict]:
    """If a named skill is in the query but not in the dense hits, pull that chunk.

    Dense retrieval carries paraphrases. Exact tool names are not left to chance:
    a resume line that says Docker has to be visible when the job says Docker.
    """
    needed = skills_in(query)
    if not needed:
        return []
    covered = {skill for skill in needed if any(skill_in_text(skill, item["text"]) for item in already)}
    missing = [skill for skill in needed if skill not in covered]
    if not missing:
        return []
    rows = db.query(Chunk).filter(Chunk.doc_id == resume_id, Chunk.doc_type == "resume").all()
    extra: list[dict] = []
    seen = {item["chunk_id"] for item in already}
    for skill in missing:
        for row in rows:
            if row.id in seen or not skill_in_text(skill, row.text):
                continue
            seen.add(row.id)
            extra.append(
                {
                    "chunk_id": row.id,
                    "text": row.text,
                    "section": row.section,
                    "score": 0.35,
                    "page": row.page,
                    "char_start": row.char_start,
                    "char_end": row.char_end,
                    "via": "lexical",
                }
            )
            break
    return extra


def retrieve_chunks(
    db: Session,
    store: VectorStore,
    embedder: Embedder,
    resume_id: str,
    query: str,
    k: int = 4,
) -> list[dict]:
    embedding = _embed(embedder, query[:2000])
    hits = store.query(resume_id, embedding, k)
    found = _hydrate(db, hits)
    found.extend(_lexical_backfill(db, resume_id, query, found))
    found.sort(key=lambda item: item["score"], reverse=True)
    # Keep every lexical save even if that means going one past k, capped at k+3.
    limit = max(k, min(len(found), k + 3))
    return found[:limit]


def retrieve_for_job(
    db: Session,
    store: VectorStore,
    embedder: Embedder,
    resume_id: str,
    job_text: str,
    k: int = 8,
) -> list[dict]:
    """Section-weighted union. Requirements outrank the about-the-role blurb."""
    queries: list[tuple[str, str, float]] = [("full", job_text[:2500], SECTION_WEIGHTS["full"])]
    for section in split_sections(job_text):
        if not section.text.strip():
            continue
        weight = SECTION_WEIGHTS.get(section.name, 0.6)
        queries.append((section.name, section.text[:2500], weight))

    merged: dict[str, dict] = {}
    for name, text, weight in queries:
        for item in retrieve_chunks(db, store, embedder, resume_id, text, k):
            weighted = max(0.0, item["score"]) * weight
            current = merged.get(item["chunk_id"])
            if current is None or weighted > current["score"]:
                merged[item["chunk_id"]] = {**item, "score": weighted, "via": name}
    ranked = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
    return ranked[:k]
