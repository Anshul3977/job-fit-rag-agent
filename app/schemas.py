"""Public request and response models."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Verdict = Literal["strong", "maybe", "skip"]
RequirementKind = Literal["required", "preferred"]
MatchStatus = Literal["match", "partial", "gap"]
Severity = Literal["high", "medium", "low"]


class ErrorBody(BaseModel):
    error_code: str
    message: str
    details: dict = Field(default_factory=dict)
    request_id: str


class HealthResponse(BaseModel):
    status: str
    version: str
    service: str


class ReadyResponse(BaseModel):
    status: str
    database: str
    vector_store: str
    embedding_backend: str
    embedding_model: str
    llm_provider: str
    llm_model: str


class IngestResumeResponse(BaseModel):
    resume_id: str
    filename: str
    chunk_count: int
    char_count: int


class IngestJobResponse(BaseModel):
    job_id: str
    title: str
    chunk_count: int
    char_count: int


class JobIn(BaseModel):
    text: str = Field(min_length=20, max_length=20_000)
    title: str | None = Field(default=None, max_length=200)
    company: str | None = Field(default=None, max_length=200)


class JobOut(BaseModel):
    job_id: str
    title: str
    company: str | None
    text: str
    chunk_count: int
    created_at: str


class ResumeOut(BaseModel):
    resume_id: str
    filename: str
    chunk_count: int
    char_count: int
    preview: str
    created_at: str


class EvidenceQuote(BaseModel):
    chunk_id: str
    quote: str
    section: str
    score: float
    page: int | None = None


class RequirementResult(BaseModel):
    requirement: str
    kind: RequirementKind
    status: MatchStatus
    severity: Severity
    rationale: str
    evidence: EvidenceQuote | None = None


class MatchingSkill(BaseModel):
    skill: str
    evidence_chunk_id: str
    quote: str


class Gap(BaseModel):
    requirement: str
    why_it_is_a_gap: str
    severity: Severity


class Citation(BaseModel):
    chunk_id: str
    quote: str
    section: str | None = None


class FitReport(BaseModel):
    overall_score: int = Field(ge=0, le=100)
    verdict: Verdict
    summary: str
    matching_skills: list[MatchingSkill]
    gaps: list[Gap]
    stretch_flags: list[str]
    recommended_resume_edits: list[str]
    citations: list[Citation]
    requirement_results: list[RequirementResult]
    model: str
    provider: Literal["groq", "deterministic"]
    provider_note: str = ""


class FitRequest(BaseModel):
    resume_id: str = Field(min_length=4, max_length=40)
    job_id: str | None = Field(default=None, min_length=4, max_length=40)
    job_text: str | None = Field(default=None, max_length=20_000)
    job_title: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def job_source(self) -> "FitRequest":
        has_text = bool(self.job_text and self.job_text.strip())
        if not self.job_id and not has_text:
            raise ValueError("Provide job_id or job_text")
        if has_text and self.job_text and len(self.job_text.strip()) < 20:
            raise ValueError("job_text must be at least 20 characters")
        return self


class FitResponse(BaseModel):
    run_id: str
    resume_id: str
    job_id: str
    job_title: str
    latency_ms: int
    retrieve_ms: int
    generate_ms: int
    model: str
    provider: str
    provider_note: str = ""
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    report: FitReport


class BatchFitRequest(BaseModel):
    resume_id: str = Field(min_length=4, max_length=40)
    job_ids: list[str] = Field(min_length=1, max_length=5)


class BatchFitResponse(BaseModel):
    results: list[FitResponse]


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    section: str
    score: float
    page: int | None
    text: str
    via: str | None = None


class DebugRetrieveResponse(BaseModel):
    resume_id: str
    job_id: str
    k: int
    chunks: list[RetrievedChunkOut]


class SampleJob(BaseModel):
    title: str
    text: str


class SamplesResponse(BaseModel):
    label: Literal["synthetic"]
    note: str
    resume_filename: str
    resume_text: str
    jobs: list[SampleJob]
