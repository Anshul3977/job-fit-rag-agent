"""HTTP routes."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from sqlalchemy.orm import Session

from app.agent.llm import build_provider
from app.agent.service import fit_response_from_run, run_fit
from app.config import Settings, get_settings
from app.db import get_db, ping_db
from app.errors import AppError, ExtractError
from app.ingest.extract import extract_upload
from app.ingest.service import (
    assert_allowed,
    assert_size,
    chunk_count,
    ingest_job_text,
    ingest_resume_bytes,
    ingest_resume_text,
    safe_filename,
    usable_text_preview,
)
from app.models import Job, Resume, Run
from app.rate_limit import get_rate_limiter
from app.retrieval.service import retrieve_for_job
from app.runtime import get_embedder, get_store
from app.samples import load_samples
from app.schemas import (
    BatchFitRequest,
    BatchFitResponse,
    DebugRetrieveResponse,
    FitRequest,
    FitResponse,
    HealthResponse,
    IngestJobResponse,
    IngestResumeResponse,
    JobIn,
    JobOut,
    ReadyResponse,
    ResumeOut,
    RetrievedChunkOut,
    SamplesResponse,
)

router = APIRouter()


def _iso(value: datetime | None) -> str:
    if value is None:
        return datetime.now(UTC).isoformat()
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(status="ok", version=settings.version, service=settings.app_name)


@router.get("/ready", response_model=ReadyResponse, tags=["meta"])
def ready(settings: Settings = Depends(get_settings)) -> ReadyResponse:
    try:
        ping_db()
    except Exception as exc:
        raise AppError(503, "NOT_READY", "Database ping failed.") from exc
    store = get_store()
    try:
        store.ping()
    except Exception as exc:
        raise AppError(503, "NOT_READY", "Vector store ping failed.") from exc
    model = settings.groq_model if settings.resolved_llm_provider == "groq" else "deterministic-v1"
    return ReadyResponse(
        status="ok",
        database="ok",
        vector_store=store.name,
        embedding_backend=settings.embedding_backend,
        embedding_model=settings.embedding_model if settings.embedding_backend == "fastembed" else "hash-384",
        llm_provider=settings.resolved_llm_provider,
        llm_model=model,
    )


@router.get("/samples", response_model=SamplesResponse, tags=["demo"])
def samples() -> SamplesResponse:
    return load_samples()


@router.post("/ingest/resume", response_model=IngestResumeResponse, tags=["ingest"])
async def ingest_resume(
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    filename: str | None = Form(default=None),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> IngestResumeResponse:
    if file is None and not (text and text.strip()):
        raise AppError(422, "VALIDATION_ERROR", "Provide a resume file or pasted text.")
    if text and len(text) > 100_000:
        raise AppError(422, "VALIDATION_ERROR", "Pasted resume text is too long.")
    if file is not None and file.filename:
        data = await file.read(settings.max_upload_bytes + 1)
        resume = ingest_resume_bytes(
            db,
            settings,
            get_store(),
            get_embedder(),
            filename=file.filename,
            content_type=file.content_type or "application/octet-stream",
            data=data,
        )
    else:
        resume = ingest_resume_text(
            db,
            settings,
            get_store(),
            get_embedder(),
            text=text or "",
            filename=safe_filename(filename or "pasted-resume.txt", "pasted-resume.txt"),
        )
    return IngestResumeResponse(
        resume_id=resume.id,
        filename=resume.filename,
        chunk_count=chunk_count(db, resume.id),
        char_count=len(resume.raw_text),
    )


@router.post("/ingest/job", response_model=IngestJobResponse, tags=["ingest"])
def ingest_job(body: JobIn, db: Session = Depends(get_db)) -> IngestJobResponse:
    job = ingest_job_text(db, text=body.text, title=body.title, company=body.company)
    return IngestJobResponse(
        job_id=job.id,
        title=job.title,
        chunk_count=chunk_count(db, job.id),
        char_count=len(job.raw_text),
    )


@router.post("/ingest/job/file", response_model=IngestJobResponse, tags=["ingest"])
async def ingest_job_file(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> IngestJobResponse:
    filename = safe_filename(file.filename or "job.txt", "job.txt")
    assert_allowed(filename)
    data = await file.read(settings.max_upload_bytes + 1)
    assert_size(data, settings.max_upload_bytes)
    try:
        extracted = extract_upload(filename, data)
    except ExtractError as exc:
        raise AppError(400, "EXTRACT_FAILED", str(exc)) from exc
    job = ingest_job_text(db, text=extracted.text, title=title, source_filename=filename)
    return IngestJobResponse(
        job_id=job.id,
        title=job.title,
        chunk_count=chunk_count(db, job.id),
        char_count=len(job.raw_text),
    )


@router.get("/resumes/{resume_id}", response_model=ResumeOut, tags=["ingest"])
def get_resume(resume_id: str, db: Session = Depends(get_db)) -> ResumeOut:
    resume = db.get(Resume, resume_id)
    if resume is None:
        raise AppError(404, "RESUME_NOT_FOUND", "No resume with that id.")
    return ResumeOut(
        resume_id=resume.id,
        filename=resume.filename,
        chunk_count=chunk_count(db, resume.id),
        char_count=len(resume.raw_text),
        preview=usable_text_preview(resume.raw_text),
        created_at=_iso(resume.created_at),
    )


@router.get("/jobs/{job_id}", response_model=JobOut, tags=["ingest"])
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobOut:
    job = db.get(Job, job_id)
    if job is None:
        raise AppError(404, "JOB_NOT_FOUND", "No job with that id.")
    return JobOut(
        job_id=job.id,
        title=job.title,
        company=job.company,
        text=job.raw_text,
        chunk_count=chunk_count(db, job.id),
        created_at=_iso(job.created_at),
    )


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()[:80]
    if request.client is None:
        return "unknown"
    return request.client.host


@router.post("/fit", response_model=FitResponse, tags=["fit"])
def fit(
    body: FitRequest,
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> FitResponse:
    get_rate_limiter(settings.rate_limit_fit_per_minute).check(_client_key(request))
    if body.job_id:
        job = db.get(Job, body.job_id)
        if job is None:
            raise AppError(404, "JOB_NOT_FOUND", "No job with that id.")
    else:
        job = ingest_job_text(db, text=body.job_text or "", title=body.job_title)
    return run_fit(
        db,
        resume_id=body.resume_id,
        job=job,
        provider=build_provider(settings),
        store=get_store(),
        embedder=get_embedder(),
    )


@router.post("/fit/batch", response_model=BatchFitResponse, tags=["fit"])
def fit_batch(
    body: BatchFitRequest,
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> BatchFitResponse:
    get_rate_limiter(settings.rate_limit_fit_per_minute).check(_client_key(request))
    results: list[FitResponse] = []
    for job_id in body.job_ids:
        job = db.get(Job, job_id)
        if job is None:
            raise AppError(404, "JOB_NOT_FOUND", "No job with that id.", {"job_id": job_id})
        results.append(
            run_fit(
                db,
                resume_id=body.resume_id,
                job=job,
                provider=build_provider(settings),
                store=get_store(),
                embedder=get_embedder(),
            )
        )
    return BatchFitResponse(results=results)


@router.get("/runs/{run_id}", response_model=FitResponse, tags=["fit"])
def get_run(run_id: str, db: Session = Depends(get_db)) -> FitResponse:
    run = db.get(Run, run_id)
    if run is None:
        raise AppError(404, "RUN_NOT_FOUND", "No run with that id.")
    job = db.get(Job, run.job_id)
    title = job.title if job is not None else "Role"
    return fit_response_from_run(run, title)


@router.get("/debug/retrieve", response_model=DebugRetrieveResponse, tags=["debug"])
def debug_retrieve(
    resume_id: str = Query(min_length=4, max_length=40),
    job_id: str = Query(min_length=4, max_length=40),
    k: int = Query(default=8, ge=1, le=20),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> DebugRetrieveResponse:
    if settings.env != "dev" or not settings.debug_routes:
        raise AppError(404, "NOT_FOUND", "Not found.")
    if db.get(Resume, resume_id) is None:
        raise AppError(404, "RESUME_NOT_FOUND", "No resume with that id.")
    job = db.get(Job, job_id)
    if job is None:
        raise AppError(404, "JOB_NOT_FOUND", "No job with that id.")
    chunks = retrieve_for_job(db, get_store(), get_embedder(), resume_id, job.raw_text, k)
    return DebugRetrieveResponse(
        resume_id=resume_id,
        job_id=job_id,
        k=k,
        chunks=[
            RetrievedChunkOut(
                chunk_id=item["chunk_id"],
                section=item["section"],
                score=round(float(item["score"]), 4),
                page=item["page"],
                text=item["text"],
                via=item.get("via"),
            )
            for item in chunks
        ],
    )
