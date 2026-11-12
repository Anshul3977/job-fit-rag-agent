from app.agent.llm import DeterministicProvider, GroqProvider, ground_requirements
from app.agent.pricing import estimate_cost_usd
from app.agent.scoring import assemble_report, judge_requirement, self_check


def _chunk(chunk_id: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "section": "experience",
        "score": 0.8,
        "page": None,
    }


def test_negated_skill_is_a_gap():
    evidence = [
        _chunk(
            "chk_no",
            "The demo server was a short Flask script. I have not shipped a FastAPI service.",
        )
    ]
    decision = judge_requirement("FastAPI for HTTP APIs", evidence)
    assert decision["status"] == "gap"
    assert "absent" in decision["rationale"]
    flask = judge_requirement("Flask", evidence)
    assert flask["status"] == "match"
    assert "not shipped" not in flask["quote"]


def test_rather_than_does_not_require_the_contrast():
    evidence = [_chunk("chk_ts", "Shipped the settings page in React and TypeScript.")]
    decision = judge_requirement("TypeScript rather than untyped JavaScript", evidence)
    assert decision["status"] == "match"


def test_fastapi_match_and_kubernetes_gap():
    evidence = [
        _chunk(
            "chk_api",
            "Built internal REST endpoints with FastAPI and PostgreSQL, covered by pytest.",
        )
    ]
    match = judge_requirement("FastAPI for HTTP APIs", evidence)
    assert match["status"] == "match"
    assert match["quote"] in evidence[0]["text"]
    gap = judge_requirement("Kubernetes for deploying services", evidence)
    assert gap["status"] == "gap"


def test_self_check_strips_invented_quote():
    evidence = {"req_1": [_chunk("chk_api", "Built internal REST endpoints with FastAPI.")]}
    judgments = [
        {
            "requirement_id": "req_1",
            "status": "match",
            "chunk_id": "chk_api",
            "quote": "Invented Kubernetes platform ownership",
            "rationale": "The model claimed a skill.",
        }
    ]
    checked = self_check(judgments, evidence)
    assert checked[0]["status"] == "gap"
    assert "Self-check" in checked[0]["rationale"]
    report = assemble_report(
        requirements=[{"id": "req_1", "text": "Kubernetes", "kind": "required"}],
        judgments=checked,
        evidence=evidence,
        summary="",
        model="test",
        provider="deterministic",
    )
    assert report.matching_skills == []
    assert report.gaps


def test_missing_required_skill_is_not_strong():
    requirements = [
        {"id": "req_1", "text": "Python", "kind": "required"},
        {"id": "req_2", "text": "FastAPI", "kind": "required"},
    ]
    evidence = {
        "req_1": [_chunk("chk_1", "Daily work is Python services.")],
        "req_2": [_chunk("chk_1", "Daily work is Python services.")],
    }
    provider = DeterministicProvider()
    judgments = self_check(provider.judge(requirements, evidence), evidence)
    report = assemble_report(
        requirements=requirements,
        judgments=judgments,
        evidence=evidence,
        summary="",
        model="deterministic-v1",
        provider="deterministic",
    )
    assert report.verdict != "strong"
    assert any(gap.requirement == "FastAPI" for gap in report.gaps)


def test_ground_requirements_drop_invented_skills():
    job = "Requirements\n- Python and FastAPI"
    grounded = ground_requirements(
        [
            {"text": "Python and FastAPI", "kind": "required"},
            {"text": "Kubernetes", "kind": "required"},
        ],
        job,
    )
    texts = [item["text"] for item in grounded]
    assert "Kubernetes" not in texts
    assert any("FastAPI" in text for text in texts)


def test_groq_price_table_is_explicit():
    assert estimate_cost_usd("openai/gpt-oss-20b", 1_000_000, 2_000_000) == 0.075 + 0.60
    assert estimate_cost_usd("not-a-real-model", 10, 10) is None


def test_groq_judge_quote_is_stripped_by_self_check():
    def handler(request):
        import json

        import httpx

        body = json.loads(request.content)
        user = body["messages"][1]["content"]
        if "judgments" in body["messages"][0]["content"]:
            content = json.dumps(
                {
                    "judgments": [
                        {
                            "requirement_id": "req_1",
                            "status": "match",
                            "chunk_id": "chk_api",
                            "quote": "Expert in Kubernetes for five years",
                            "rationale": "unsupported",
                        }
                    ]
                }
            )
        elif "requirements" in body["messages"][0]["content"] and "Kubernetes" not in user:
            content = json.dumps({"requirements": [{"text": "FastAPI", "kind": "required"}]})
        else:
            content = json.dumps({"summary": "maybe fit at 10/100"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            },
        )

    import httpx

    provider = GroqProvider(
        api_key="test",
        model="openai/gpt-oss-20b",
        timeout=2,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    requirements = provider.extract_requirements("Requirements\n- FastAPI")
    evidence = {"req_1": [_chunk("chk_api", "Built internal REST endpoints with FastAPI.")]}
    checked = self_check(provider.judge(requirements, evidence), evidence)
    assert checked[0]["status"] == "gap"
    assert provider.usage() == (22, 14)
