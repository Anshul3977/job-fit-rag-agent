"""Run the fit graph and store the report."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.agent.graph import compile_fit_graph
from app.agent.llm import LLMProvider
from app.agent.pricing import estimate_cost_usd
from app.config import get_settings
from app.errors import AppError
from app.guardrails import sanitize, scan
from app.ids import new_id
from app.models import Job, Resume, Run
from app.retrieval.embeddings import Embedder
from app.retrieval.service import retrieve_chunks
from app.retrieval.store import VectorStore
from app.runtime import get_reranker
from app.schemas import FitReport, FitResponse


def run_fit(
    db: Session,
    *,
    resume_id: str,
    job: Job,
    provider: LLMProvider,
    store: VectorStore,
    embedder: Embedder,
) -> FitResponse:
    resume = db.get(Resume, resume_id)
    if resume is None:
        raise AppError(404, "RESUME_NOT_FOUND", "No resume with that id.")

    settings = get_settings()
    reranker = get_reranker() if settings.retrieval_mode == "hybrid" else None

    def retrieve(query: str, k: int) -> list[dict]:
        return retrieve_chunks(
            db,
            store,
            embedder,
            resume_id,
            query,
            k,
            mode=settings.retrieval_mode,
            reranker=reranker,
            shortlist=settings.rerank_shortlist,
        )

    job_text = job.raw_text
    guardrail_flags: list[str] = []
    if settings.guardrails:
        job_text, job_flags = sanitize(job.raw_text)
        guardrail_flags = [f.describe("Resume") for f in scan(resume.raw_text)] + [
            f.describe("Job post") for f in job_flags
        ]

    provider.reset_usage()
    graph = compile_fit_graph(provider, retrieve)
    started = time.perf_counter()
    state = graph.invoke(
        {
            "resume_id": resume_id,
            "job_text": job_text,
            "job_title": job.title,
            "timings": {},
        }
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    timings = state.get("timings") or {}
    retrieve_ms = int(timings.get("retrieve_evidence") or 0)
    generate_ms = int(
        sum(
            timings.get(name) or 0
            for name in ("extract_requirements", "judge_matches", "self_check", "score", "explain")
        )
    )
    report = FitReport.model_validate(state["report"])
    report.guardrail_flags = guardrail_flags
    prompt_tokens, completion_tokens = provider.usage()
    priced = getattr(provider, "priced_as_groq", None)
    if priced is not None and priced():
        price_model = getattr(provider, "price_model", lambda: provider.model_name)()
        cost = estimate_cost_usd(price_model, prompt_tokens, completion_tokens)
        cost_usd = float(cost or 0.0)
    else:
        cost_usd = 0.0

    run = Run(
        id=new_id("run"),
        resume_id=resume_id,
        job_id=job.id,
        score=report.overall_score,
        verdict=report.verdict,
        model=report.model,
        provider=report.provider,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        retrieve_ms=retrieve_ms,
        generate_ms=generate_ms,
        cost_usd=cost_usd,
        report_json=report.model_dump_json(),
        created_at=datetime.now(UTC),
    )
    db.add(run)
    db.commit()
    return FitResponse(
        run_id=run.id,
        resume_id=resume_id,
        job_id=job.id,
        job_title=job.title,
        latency_ms=latency_ms,
        retrieve_ms=retrieve_ms,
        generate_ms=generate_ms,
        model=report.model,
        provider=report.provider,
        provider_note=report.provider_note,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        report=report,
    )


def fit_response_from_run(run: Run, job_title: str) -> FitResponse:
    report = FitReport.model_validate_json(run.report_json)
    return FitResponse(
        run_id=run.id,
        resume_id=run.resume_id,
        job_id=run.job_id,
        job_title=job_title,
        latency_ms=run.latency_ms,
        retrieve_ms=run.retrieve_ms,
        generate_ms=run.generate_ms,
        model=run.model,
        provider=run.provider,
        provider_note=report.provider_note,
        prompt_tokens=run.prompt_tokens,
        completion_tokens=run.completion_tokens,
        cost_usd=run.cost_usd,
        report=report,
    )
