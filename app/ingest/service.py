"""Persist a resume or job and index resume chunks."""

from __future__ import annotations

import re
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import Settings
from app.errors import AppError, ExtractError
from app.ids import new_id
from app.ingest.chunking import chunk_text, match_header
from app.ingest.extract import ALLOWED_EXTENSIONS, extract_upload, usable_text
from app.models import Chunk, Job, Resume
from app.retrieval.embeddings import Embedder
from app.retrieval.store import VectorStore

_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str, default: str) -> str:
    base = Path(name or "").name.strip() or default
    cleaned = _FILENAME.sub("_", base).strip("._") or default
    return cleaned[:180]


def extension_of(filename: str) -> str:
    if "." not in filename:
        return ""
    return "." + filename.lower().rsplit(".", 1)[-1]


def assert_allowed(filename: str) -> None:
    if extension_of(filename) not in ALLOWED_EXTENSIONS:
        raise AppError(
            422,
            "UNSUPPORTED_FILE_TYPE",
            "Upload a PDF, DOCX, TXT, or MD file.",
            {"allowed": sorted(ALLOWED_EXTENSIONS)},
        )


def assert_size(data: bytes, limit: int) -> None:
    if len(data) > limit:
        raise AppError(
            422,
            "FILE_TOO_LARGE",
            f"File is larger than {limit} bytes.",
            {"max_upload_bytes": limit},
        )
    if len(data) == 0:
        raise AppError(400, "EMPTY_DOCUMENT", "The upload was empty.")


def require_text(text: str) -> str:
    cleaned = text.strip()
    letters = sum(1 for char in cleaned if char.isalpha())
    if letters < 40:
        raise AppError(
            400,
            "EMPTY_DOCUMENT",
            "No usable text could be extracted. The file may be scanned or empty.",
        )
    return cleaned


def infer_title(text: str, explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()[:200]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or match_header(stripped):
            continue
        if stripped.startswith(("-", "*", "•")):
            continue
        if len(stripped) <= 80:
            return stripped[:200]
        break
    return "Untitled role"


def _save_chunks(db: Session, doc_id: str, doc_type: str, drafts: list) -> list[Chunk]:
    rows: list[Chunk] = []
    for draft in drafts:
        row = Chunk(
            id=new_id("chk"),
            doc_id=doc_id,
            doc_type=doc_type,
            section=draft.section,
            text=draft.text,
            page=draft.page,
            char_start=draft.char_start,
            char_end=draft.char_end,
            token_estimate=draft.token_estimate,
        )
        rows.append(row)
        db.add(row)
    return rows


def ingest_resume_bytes(
    db: Session,
    settings: Settings,
    store: VectorStore,
    embedder: Embedder,
    filename: str,
    content_type: str,
    data: bytes,
) -> Resume:
    filename = safe_filename(filename, "resume.txt")
    assert_allowed(filename)
    assert_size(data, settings.max_upload_bytes)
    try:
        extracted = extract_upload(filename, data)
    except ExtractError as exc:
        raise AppError(400, "EXTRACT_FAILED", str(exc)) from exc
    text = require_text(extracted.text)
    drafts = chunk_text(text, extracted.page_spans)
    if not drafts:
        raise AppError(400, "EMPTY_DOCUMENT", "The document produced no chunks.")

    resume_id = new_id("res")
    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored = upload_dir / f"{resume_id}_{filename}"
    stored.write_bytes(data)

    resume = Resume(
        id=resume_id,
        filename=filename,
        content_type=content_type or "application/octet-stream",
        raw_text=text,
        byte_size=len(data),
        stored_path=str(stored),
    )
    db.add(resume)
    chunks = _save_chunks(db, resume_id, "resume", drafts)
    try:
        vectors = embedder.embed([chunk.text for chunk in chunks])
        store.upsert(
            [
                {
                    "chunk_id": chunk.id,
                    "resume_id": resume_id,
                    "section": chunk.section,
                    "text": chunk.text,
                    "embedding": vector,
                }
                for chunk, vector in zip(chunks, vectors)
            ]
        )
        db.commit()
    except AppError:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise AppError(500, "INDEX_FAILED", "The resume was read but could not be indexed.") from exc
    db.refresh(resume)
    return resume


def ingest_resume_text(
    db: Session,
    settings: Settings,
    store: VectorStore,
    embedder: Embedder,
    text: str,
    filename: str,
) -> Resume:
    return ingest_resume_bytes(
        db,
        settings,
        store,
        embedder,
        filename=filename if filename.endswith((".txt", ".md")) else f"{filename}.txt",
        content_type="text/plain",
        data=text.encode("utf-8"),
    )


def ingest_job_text(
    db: Session,
    text: str,
    title: str | None = None,
    company: str | None = None,
    source_filename: str | None = None,
) -> Job:
    cleaned = require_text(text)
    if len(cleaned) > 20_000:
        raise AppError(422, "VALIDATION_ERROR", "Job text is longer than 20000 characters.")
    drafts = chunk_text(cleaned, None)
    job = Job(
        id=new_id("job"),
        title=infer_title(cleaned, title),
        company=company.strip()[:200] if company else None,
        raw_text=cleaned,
        source_filename=source_filename,
    )
    db.add(job)
    _save_chunks(db, job.id, "job", drafts)
    db.commit()
    db.refresh(job)
    return job


def chunk_count(db: Session, doc_id: str) -> int:
    return db.query(Chunk).filter(Chunk.doc_id == doc_id).count()


def usable_text_preview(text: str, limit: int = 400) -> str:
    compact = usable_text(text)
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"
