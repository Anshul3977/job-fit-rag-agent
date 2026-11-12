"""Point the app at a temp SQLite file and the hash embedder before import."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

ROOT = Path(tempfile.mkdtemp(prefix="jobfit-test-"))
os.environ["DATABASE_URL"] = f"sqlite:///{ROOT / 'test.db'}"
os.environ["CHROMA_DIR"] = str(ROOT / "chroma")
os.environ["VECTOR_BACKEND"] = "chroma"
os.environ["EMBEDDING_BACKEND"] = "hash"
os.environ["EMBEDDING_CACHE_DIR"] = str(ROOT / "models")
os.environ["LLM_PROVIDER"] = "deterministic"
os.environ["GROQ_API_KEY"] = ""
os.environ["ENV"] = "test"
os.environ["DEBUG_ROUTES"] = "false"
os.environ["RATE_LIMIT_FIT_PER_MINUTE"] = "10000"
os.environ["LOG_LEVEL"] = "WARNING"

from app.config import get_settings  # noqa: E402
from app.db import get_engine, reset_engine  # noqa: E402
from app.models import Base  # noqa: E402
from app.rate_limit import reset_rate_limiter  # noqa: E402
from app.runtime import get_store, reset_runtime  # noqa: E402

get_settings.cache_clear()
reset_engine()
reset_runtime()


@pytest.fixture(autouse=True)
def clean_state():
    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    get_store().reset()
    reset_rate_limiter()
    yield


@pytest.fixture(scope="session")
def client():
    from app.main import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


BACKEND_RESUME = """
Avery Chen
SUMMARY
Backend engineer who ships Python services.
EXPERIENCE
- Built REST endpoints with FastAPI and PostgreSQL, covered by pytest.
- Packaged the API with Docker.
SKILLS
Python, FastAPI, PostgreSQL, Docker, pytest
"""

PLATFORM_JOB = """
Platform Engineer
REQUIREMENTS
- Kubernetes for deploying services
- Go for internal tooling
NICE TO HAVE
- Terraform
"""

BACKEND_JOB = """
Backend Engineer
REQUIREMENTS
- Python as the main language
- FastAPI for HTTP APIs
- PostgreSQL for the primary datastore
NICE TO HAVE
- Docker
"""
