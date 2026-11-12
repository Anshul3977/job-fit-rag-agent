import io

from docx import Document
from fpdf import FPDF
from tests.conftest import BACKEND_JOB, BACKEND_RESUME, PLATFORM_JOB


def _pdf(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, text)
    return bytes(pdf.output())


def _docx(paragraphs: list[str]) -> bytes:
    document = Document()
    document.add_heading("EXPERIENCE", level=1)
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_ingest_txt_and_job_then_fit_round_trip(client):
    resume = client.post(
        "/ingest/resume",
        files={"file": ("resume.txt", BACKEND_RESUME.encode(), "text/plain")},
    )
    assert resume.status_code == 200
    body = resume.json()
    assert body["chunk_count"] > 0
    assert body["resume_id"].startswith("res_")

    job = client.post("/ingest/job", json={"text": BACKEND_JOB, "title": "Backend Engineer"})
    assert job.status_code == 200
    job_id = job.json()["job_id"]

    fetched = client.get(f"/jobs/{job_id}")
    assert fetched.status_code == 200
    assert "FastAPI" in fetched.json()["text"]

    fit = client.post("/fit", json={"resume_id": body["resume_id"], "job_id": job_id})
    assert fit.status_code == 200
    report = fit.json()["report"]
    assert report["provider"] == "deterministic"
    assert report["citations"]
    skills = " ".join(item["skill"] for item in report["matching_skills"])
    assert "FastAPI" in skills
    for match in report["matching_skills"]:
        assert match["quote"]
        assert match["evidence_chunk_id"].startswith("chk_")

    run = client.get(f"/runs/{fit.json()['run_id']}")
    assert run.status_code == 200
    assert run.json()["report"]["overall_score"] == report["overall_score"]


def test_kubernetes_is_a_gap(client):
    resume = client.post(
        "/ingest/resume",
        files={"file": ("resume.txt", BACKEND_RESUME.encode(), "text/plain")},
    )
    job = client.post("/ingest/job", json={"text": PLATFORM_JOB, "title": "Platform Engineer"})
    fit = client.post(
        "/fit",
        json={"resume_id": resume.json()["resume_id"], "job_id": job.json()["job_id"]},
    )
    assert fit.status_code == 200
    gaps = " ".join(item["requirement"] for item in fit.json()["report"]["gaps"])
    assert "Kubernetes" in gaps
    matched = " ".join(item["skill"] for item in fit.json()["report"]["matching_skills"])
    assert "Kubernetes" not in matched


def test_batch_fit(client):
    resume = client.post(
        "/ingest/resume",
        files={"file": ("resume.txt", BACKEND_RESUME.encode(), "text/plain")},
    )
    first = client.post("/ingest/job", json={"text": BACKEND_JOB, "title": "Backend"})
    second = client.post("/ingest/job", json={"text": PLATFORM_JOB, "title": "Platform"})
    batch = client.post(
        "/fit/batch",
        json={
            "resume_id": resume.json()["resume_id"],
            "job_ids": [first.json()["job_id"], second.json()["job_id"]],
        },
    )
    assert batch.status_code == 200
    assert len(batch.json()["results"]) == 2


def test_pdf_and_docx_ingest(client):
    pdf = client.post(
        "/ingest/resume",
        files={"file": ("resume.pdf", _pdf(BACKEND_RESUME), "application/pdf")},
    )
    assert pdf.status_code == 200
    assert pdf.json()["chunk_count"] > 0

    docx = client.post(
        "/ingest/resume",
        files={
            "file": (
                "resume.docx",
                _docx(["Built FastAPI services on PostgreSQL and tested them with pytest."]),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert docx.status_code == 200
    assert docx.json()["char_count"] > 40


def test_empty_pdf_and_bad_type(client):
    empty = client.post(
        "/ingest/resume",
        files={"file": ("blank.pdf", _pdf("   "), "application/pdf")},
    )
    assert empty.status_code == 400
    assert empty.json()["error_code"] == "EMPTY_DOCUMENT"

    corrupt = client.post(
        "/ingest/resume",
        files={"file": ("broken.pdf", b"not a pdf at all", "application/pdf")},
    )
    assert corrupt.status_code == 400
    assert corrupt.json()["error_code"] == "EXTRACT_FAILED"

    rejected = client.post(
        "/ingest/resume",
        files={"file": ("notes.exe", b"hello", "application/octet-stream")},
    )
    assert rejected.status_code == 422
    assert rejected.json()["error_code"] == "UNSUPPORTED_FILE_TYPE"


def test_validation_and_oversize(client):
    short = client.post("/ingest/job", json={"text": "too short"})
    assert short.status_code == 422
    assert short.json()["error_code"] == "VALIDATION_ERROR"
    assert short.json()["details"]["fields"]

    huge = b"A" * 5_000_001
    over = client.post("/ingest/resume", files={"file": ("resume.txt", huge, "text/plain")})
    assert over.status_code == 422
    assert over.json()["error_code"] == "FILE_TOO_LARGE"

    missing = client.post("/fit", json={"resume_id": "res_missing000"})
    assert missing.status_code == 422


def test_rate_limit(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_FIT_PER_MINUTE", "1")
    from app.config import get_settings
    from app.rate_limit import reset_rate_limiter

    get_settings.cache_clear()
    reset_rate_limiter()
    resume = client.post(
        "/ingest/resume",
        files={"file": ("resume.txt", BACKEND_RESUME.encode(), "text/plain")},
    )
    job = client.post("/ingest/job", json={"text": BACKEND_JOB, "title": "Backend"})
    payload = {"resume_id": resume.json()["resume_id"], "job_id": job.json()["job_id"]}
    assert client.post("/fit", json=payload).status_code == 200
    limited = client.post("/fit", json=payload)
    assert limited.status_code == 429
    assert limited.json()["error_code"] == "RATE_LIMITED"
    get_settings.cache_clear()
    reset_rate_limiter()


def test_samples_are_marked_synthetic(client):
    response = client.get("/samples")
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "synthetic"
    assert "Synthetic" in body["resume_text"] or "synthetic" in body["resume_text"]
    assert len(body["jobs"]) == 2


def test_retrieval_finds_fastapi_and_ranks_it_above_unrelated(client):
    resume = client.post(
        "/ingest/resume",
        files={"file": ("resume.txt", BACKEND_RESUME.encode(), "text/plain")},
    )
    job = client.post("/ingest/job", json={"text": BACKEND_JOB, "title": "Backend"})
    from app.config import get_settings

    monkey_env = get_settings()
    del monkey_env
    # Enable debug for this call only.
    import os

    os.environ["ENV"] = "dev"
    os.environ["DEBUG_ROUTES"] = "true"
    get_settings.cache_clear()
    found = client.get(
        "/debug/retrieve",
        params={"resume_id": resume.json()["resume_id"], "job_id": job.json()["job_id"], "k": 4},
    )
    assert found.status_code == 200
    chunks = found.json()["chunks"]
    assert chunks
    assert any("FastAPI" in chunk["text"] for chunk in chunks)
    top_score = chunks[0]["score"]

    unrelated = """
Unrelated role
REQUIREMENTS
- underwater basket weaving
- ceramic kiln maintenance
- glaze chemistry
"""
    other = client.post("/ingest/job", json={"text": unrelated, "title": "Ceramics"})
    ceramics = client.get(
        "/debug/retrieve",
        params={
            "resume_id": resume.json()["resume_id"],
            "job_id": other.json()["job_id"],
            "k": 4,
        },
    )
    assert ceramics.status_code == 200
    ceramic_scores = [chunk["score"] for chunk in ceramics.json()["chunks"]]
    assert ceramic_scores
    assert max(ceramic_scores) < top_score
    os.environ["ENV"] = "test"
    os.environ["DEBUG_ROUTES"] = "false"
    get_settings.cache_clear()
