"""Groq JSON-validation and rate-limit fallbacks, with a mocked HTTP client."""

import json

import httpx
import pytest
from app.agent.llm import GROQ_MAX_COMPLETION_TOKENS, GroqProvider
from app.errors import AppError

FASTAPI_CHUNK = {
    "chunk_id": "chk_api",
    "text": "Built internal REST endpoints with FastAPI and PostgreSQL, covered by pytest.",
    "section": "experience",
    "score": 0.8,
    "page": None,
}


def _provider(handler, model="openai/gpt-oss-20b", sleeps=None):
    recorded = sleeps if sleeps is not None else []

    def sleeper(seconds: float) -> None:
        recorded.append(seconds)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = GroqProvider(
        api_key="test-key",
        model=model,
        timeout=2,
        client=client,
        sleeper=sleeper,
    )
    return provider, recorded


def _ok(content: dict | str, prompt: int = 11, completion: int = 7) -> httpx.Response:
    if not isinstance(content, str):
        content = json.dumps(content)
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        },
    )


def _json_fail() -> httpx.Response:
    return httpx.Response(
        400,
        json={
            "error": {
                "message": "Failed to generate JSON",
                "type": "invalid_request_error",
                "code": "json_validate_failed",
            }
        },
    )


def _rate(retry_after: str) -> httpx.Response:
    return httpx.Response(
        429,
        headers={"retry-after": retry_after},
        json={"error": {"message": "Rate limit", "code": "rate_limit_exceeded"}},
    )


def _judge_body():
    return {
        "judgments": [
            {
                "requirement_id": "req_1",
                "status": "match",
                "chunk_id": "chk_api",
                "quote": "Built internal REST endpoints with FastAPI",
                "rationale": "The chunk names FastAPI.",
            }
        ]
    }


def test_gpt_oss_raises_token_budget_and_sets_low_reasoning_effort():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return _ok(_judge_body())

    provider, sleeps = _provider(handler)
    judgments = provider.judge(
        [{"id": "req_1", "text": "FastAPI", "kind": "required"}],
        {"req_1": [FASTAPI_CHUNK]},
    )
    assert judgments[0]["status"] == "match"
    assert seen["max_tokens"] == GROQ_MAX_COMPLETION_TOKENS
    assert seen["max_completion_tokens"] == GROQ_MAX_COMPLETION_TOKENS
    assert seen["max_tokens"] > 1400
    assert seen["reasoning_effort"] == "low"
    assert seen["temperature"] == 0
    assert sleeps == []
    assert provider.provider_id == "groq"
    assert provider.provider_note == ""


def test_other_models_do_not_send_reasoning_effort():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return _ok(_judge_body())

    provider, _sleeps = _provider(handler, model="llama-3.1-8b-instant")
    provider.judge(
        [{"id": "req_1", "text": "FastAPI", "kind": "required"}],
        {"req_1": [FASTAPI_CHUNK]},
    )
    assert "reasoning_effort" not in seen
    assert seen["max_completion_tokens"] == GROQ_MAX_COMPLETION_TOKENS


def test_json_validation_retries_once_then_accepts_the_second_response():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return _json_fail()
        return _ok(_judge_body())

    provider, sleeps = _provider(handler)
    judgments = provider.judge(
        [{"id": "req_1", "text": "FastAPI", "kind": "required"}],
        {"req_1": [FASTAPI_CHUNK]},
    )
    assert calls["n"] == 2
    assert sleeps == []
    assert judgments[0]["quote"].startswith("Built internal REST endpoints")
    assert provider.provider_id == "groq"
    assert provider.model_name == "openai/gpt-oss-20b"
    assert provider.provider_note == ""


def test_json_validation_falls_back_to_the_deterministic_judge():
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _json_fail()

    provider, _sleeps = _provider(handler)
    judgments = provider.judge(
        [{"id": "req_1", "text": "FastAPI for HTTP APIs", "kind": "required"}],
        {"req_1": [FASTAPI_CHUNK]},
    )
    assert calls["n"] == 2
    assert judgments[0]["status"] == "match"
    assert "FastAPI" in judgments[0]["quote"]
    assert provider.provider_id == "deterministic"
    assert provider.model_name == "deterministic-v1"
    assert "deterministic scorer" in provider.provider_note
    assert "invalid JSON" in provider.provider_note


def test_empty_message_is_treated_as_a_json_failure_and_falls_back():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": ""}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 0},
            },
        )

    provider, _sleeps = _provider(handler)
    judgments = provider.judge(
        [{"id": "req_1", "text": "FastAPI for HTTP APIs", "kind": "required"}],
        {"req_1": [FASTAPI_CHUNK]},
    )
    assert judgments[0]["status"] == "match"
    assert provider.provider_id == "deterministic"
    assert provider.usage() == (8, 0)


def test_unrelated_400_is_still_a_503():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "bad model", "code": "model_not_found"}},
        )

    provider, _sleeps = _provider(handler)
    with pytest.raises(AppError) as caught:
        provider.judge(
            [{"id": "req_1", "text": "FastAPI", "kind": "required"}],
            {"req_1": [FASTAPI_CHUNK]},
        )
    assert caught.value.status_code == 503
    assert caught.value.error_code == "LLM_UNAVAILABLE"
    assert provider.provider_note == ""


def test_rate_limit_honors_retry_after_then_succeeds():
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return _rate("2.5")
        return _ok(_judge_body())

    provider, sleeps = _provider(handler)
    judgments = provider.judge(
        [{"id": "req_1", "text": "FastAPI", "kind": "required"}],
        {"req_1": [FASTAPI_CHUNK]},
    )
    assert calls["n"] == 2
    assert sleeps == [2.5]
    assert judgments[0]["status"] == "match"
    assert provider.provider_id == "groq"
    assert provider.provider_note == ""


def test_rate_limit_caps_retry_after_and_falls_back():
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _rate("30")

    provider, sleeps = _provider(handler)
    judgments = provider.judge(
        [{"id": "req_1", "text": "FastAPI for HTTP APIs", "kind": "required"}],
        {"req_1": [FASTAPI_CHUNK]},
    )
    assert calls["n"] == 4
    assert sleeps == [8.0, 8.0, 8.0]
    assert judgments[0]["status"] == "match"
    assert provider.provider_id == "deterministic"
    assert "rate-limited" in provider.provider_note
    assert "deterministic scorer" in provider.provider_note


def test_rate_limit_on_extract_skips_later_groq_calls():
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _rate("1")

    provider, sleeps = _provider(handler)
    requirements = provider.extract_requirements("Requirements\n- FastAPI experience in production services")
    assert requirements
    assert calls["n"] == 4
    judgments = provider.judge(requirements, {requirements[0]["id"]: [FASTAPI_CHUNK]})
    summary = provider.write_summary("Backend", 10, "skip", [])
    assert calls["n"] == 4
    assert sleeps == [1.0, 1.0, 1.0]
    assert provider.provider_id == "deterministic"
    assert "requirement extraction" in provider.provider_note
    assert "judge step" in provider.provider_note
    assert "summary" in provider.provider_note
    assert judgments
    assert summary


def test_missing_retry_after_uses_exponential_backoff():
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"error": {"message": "slow"}})
        return _ok(_judge_body())

    provider, sleeps = _provider(handler)
    provider.judge(
        [{"id": "req_1", "text": "FastAPI", "kind": "required"}],
        {"req_1": [FASTAPI_CHUNK]},
    )
    assert sleeps == [0.5, 1.0]
    assert provider.provider_id == "groq"
