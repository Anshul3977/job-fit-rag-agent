def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["service"] == "job-fit-rag-agent"
    assert response.headers.get("X-Request-ID")


def test_ready(client):
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["vector_store"] == "chroma"
    assert body["llm_provider"] == "deterministic"
    assert body["embedding_backend"] == "hash"


def test_debug_hidden_unless_dev(client, monkeypatch):
    hidden = client.get("/debug/retrieve", params={"resume_id": "res_abc", "job_id": "job_abc"})
    assert hidden.status_code == 404
    assert hidden.json()["error_code"] == "NOT_FOUND"

    monkeypatch.setenv("ENV", "dev")
    monkeypatch.setenv("DEBUG_ROUTES", "true")
    from app.config import get_settings

    get_settings.cache_clear()
    missing = client.get("/debug/retrieve", params={"resume_id": "res_missing1", "job_id": "job_missing1"})
    assert missing.status_code == 404
    assert missing.json()["error_code"] == "RESUME_NOT_FOUND"
    get_settings.cache_clear()
