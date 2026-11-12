"""LLM providers.

Groq is used when a key is configured. Otherwise a deterministic provider
extracts bullets and judges them from retrieved quotes. Both go through the
same graph, and both still have to survive the citation self-check.
"""

from __future__ import annotations

import json
import re
import time
from typing import Protocol

import httpx

from app.agent.scoring import deterministic_summary, judge_requirement
from app.config import Settings
from app.errors import AppError
from app.ingest.chunking import match_header

BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+(?P<body>\S.*?)\s*$")
MAX_REQUIREMENTS = 8


def _clean_bullet(body: str) -> str:
    body = re.sub(r"\s+", " ", body).strip(" -*")
    if len(body) > 240:
        body = body[:237].rstrip() + "..."
    return body


def extract_requirements_deterministic(job_text: str) -> list[dict]:
    """Pull required and preferred bullets. Requirements sections win over the rest."""
    current = "preamble"
    scoped: list[tuple[str, str]] = []
    fallback: list[str] = []
    for raw_line in job_text.splitlines():
        header = match_header(raw_line)
        if header:
            current = header
            continue
        match = BULLET_RE.match(raw_line)
        if not match:
            continue
        body = _clean_bullet(match.group("body"))
        if len(body) < 8:
            continue
        if current == "requirements":
            scoped.append(("required", body))
        elif current == "preferred":
            scoped.append(("preferred", body))
        else:
            fallback.append(body)

    chosen = scoped or [("required", body) for body in fallback]
    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for kind, body in chosen:
        key = body.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((kind, body))
    required = [item for item in deduped if item[0] == "required"]
    preferred = [item for item in deduped if item[0] == "preferred"]
    trimmed = (required + preferred)[:MAX_REQUIREMENTS]
    if not trimmed:
        trimmed = [("required", "Relevant experience for this role")]
    return [
        {"id": f"req_{index}", "text": text, "kind": kind}
        for index, (kind, text) in enumerate(trimmed, start=1)
    ]


def _parse_json_content(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AppError(
            503,
            "LLM_BAD_RESPONSE",
            "The model did not return JSON.",
        ) from exc
    if not isinstance(data, dict):
        raise AppError(503, "LLM_BAD_RESPONSE", "The model returned a JSON value that is not an object.")
    return data


def ground_requirements(requirements: list[dict], job_text: str) -> list[dict]:
    """Drop requirements the model did not take from the posting."""
    haystack = job_text.casefold()
    grounded: list[dict] = []
    for requirement in requirements:
        text = re.sub(r"\s+", " ", requirement.get("text") or "").strip()
        if len(text) < 3:
            continue
        words = re.findall(r"[A-Za-z][A-Za-z0-9.+#-]{1,}", text)
        if text.casefold() in haystack or (words and all(word.casefold() in haystack for word in words)):
            kind = requirement.get("kind")
            grounded.append(
                {
                    "id": f"req_{len(grounded) + 1}",
                    "text": text[:240],
                    "kind": "preferred" if kind == "preferred" else "required",
                }
            )
        if len(grounded) >= MAX_REQUIREMENTS:
            break
    return grounded


class LLMProvider(Protocol):
    provider_id: str
    model_name: str

    def reset_usage(self) -> None: ...

    def usage(self) -> tuple[int, int]: ...

    def extract_requirements(self, job_text: str) -> list[dict]: ...

    def judge(self, requirements: list[dict], evidence: dict[str, list[dict]]) -> list[dict]: ...

    def write_summary(self, job_title: str, score: int, verdict: str, results: list[dict]) -> str: ...


class DeterministicProvider:
    provider_id = "deterministic"
    model_name = "deterministic-v1"

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def reset_usage(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def usage(self) -> tuple[int, int]:
        return self.prompt_tokens, self.completion_tokens

    def extract_requirements(self, job_text: str) -> list[dict]:
        self.prompt_tokens += max(1, len(job_text) // 4)
        requirements = extract_requirements_deterministic(job_text)
        self.completion_tokens += max(1, sum(len(item["text"]) for item in requirements) // 4)
        return requirements

    def judge(self, requirements: list[dict], evidence: dict[str, list[dict]]) -> list[dict]:
        judgments = []
        for requirement in requirements:
            chunks = evidence.get(requirement["id"], [])
            self.prompt_tokens += max(1, sum(len(chunk["text"]) for chunk in chunks) // 4)
            decision = judge_requirement(requirement["text"], chunks)
            decision["requirement_id"] = requirement["id"]
            judgments.append(decision)
            self.completion_tokens += 40
        return judgments

    def write_summary(self, job_title: str, score: int, verdict: str, results: list[dict]) -> str:
        del job_title
        from app.schemas import RequirementResult

        models = [RequirementResult.model_validate(item) for item in results]
        summary = deterministic_summary(score, verdict, models)
        self.completion_tokens += max(1, len(summary) // 4)
        return summary


class GroqProvider:
    provider_id = "groq"

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: float,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.model_name = model
        self.timeout = timeout
        self._client = client
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def reset_usage(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def usage(self) -> tuple[int, int]:
        return self.prompt_tokens, self.completion_tokens

    def _complete(self, system: str, user: str) -> dict:
        payload = {
            "model": self.model_name,
            "temperature": 0,
            "max_tokens": 1400,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                if self._client is not None:
                    response = self._client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                else:
                    with httpx.Client(timeout=httpx.Timeout(self.timeout, connect=5.0)) as client:
                        response = client.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers=headers,
                            json=payload,
                        )
                if response.status_code in {429, 500, 502, 503, 504} and attempt == 0:
                    time.sleep(0.4)
                    continue
                if response.status_code >= 400:
                    raise AppError(
                        503,
                        "LLM_UNAVAILABLE",
                        "The Groq API rejected the request.",
                        {"status": response.status_code},
                    )
                data = response.json()
                usage = data.get("usage") or {}
                self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                self.completion_tokens += int(usage.get("completion_tokens") or 0)
                content = data["choices"][0]["message"].get("content") or ""
                if not content.strip():
                    raise AppError(503, "LLM_BAD_RESPONSE", "The model returned an empty message.")
                return _parse_json_content(content)
            except AppError:
                raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt == 0:
                    continue
        raise AppError(
            503,
            "LLM_UNAVAILABLE",
            "The Groq API did not respond in time.",
        ) from last_error

    def extract_requirements(self, job_text: str) -> list[dict]:
        data = self._complete(
            (
                "Extract hiring requirements from a job post. Return JSON only: "
                '{"requirements": [{"text": string, "kind": "required"|"preferred"}]}. '
                "Use the posting's own wording. Do not add skills that are not written there. "
                "Put must-haves in required and nice-to-haves in preferred. At most 8 items."
            ),
            job_text[:8000],
        )
        raw = data.get("requirements") or []
        if not isinstance(raw, list):
            raise AppError(503, "LLM_BAD_RESPONSE", "Requirements were not a list.")
        grounded = ground_requirements(raw, job_text)
        if not grounded:
            return extract_requirements_deterministic(job_text)
        return grounded

    def judge(self, requirements: list[dict], evidence: dict[str, list[dict]]) -> list[dict]:
        packet = []
        for requirement in requirements:
            chunks = []
            for chunk in evidence.get(requirement["id"], [])[:4]:
                chunks.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "section": chunk["section"],
                        "text": chunk["text"][:900],
                    }
                )
            packet.append(
                {
                    "requirement_id": requirement["id"],
                    "text": requirement["text"],
                    "kind": requirement["kind"],
                    "chunks": chunks,
                }
            )
        data = self._complete(
            (
                "Judge whether each job requirement is supported by the resume chunks. "
                'Return JSON only: {"judgments": [{"requirement_id": string, '
                '"status": "match"|"partial"|"gap", "chunk_id": string, '
                '"quote": string, "rationale": string}]}. '
                "quote MUST be an exact substring of the chosen chunk. "
                "If the chunks do not support the requirement, status is gap and quote is empty. "
                "Never infer an employer, a year, or a skill that is not written in a chunk. "
                "partial means a related or incomplete signal, not a full hit."
            ),
            json.dumps({"requirements": packet}),
        )
        raw = data.get("judgments") or []
        if not isinstance(raw, list):
            raise AppError(503, "LLM_BAD_RESPONSE", "Judgments were not a list.")
        by_id = {}
        for item in raw:
            if isinstance(item, dict) and item.get("requirement_id"):
                by_id[item["requirement_id"]] = item
        judgments = []
        for requirement in requirements:
            item = by_id.get(requirement["id"], {})
            status = item.get("status")
            if status not in {"match", "partial", "gap"}:
                status = "gap"
            judgments.append(
                {
                    "requirement_id": requirement["id"],
                    "status": status,
                    "chunk_id": item.get("chunk_id") or "",
                    "quote": item.get("quote") or "",
                    "rationale": (item.get("rationale") or "No rationale returned.")[:500],
                }
            )
        return judgments

    def write_summary(self, job_title: str, score: int, verdict: str, results: list[dict]) -> str:
        data = self._complete(
            (
                'Write a two-sentence fit summary. Return JSON: {"summary": string}. '
                "Use only the supplied results. Do not add skills, employers, or years. "
                f"The score is {score}/100 and the verdict is {verdict}. Mention both."
            ),
            json.dumps({"job_title": job_title, "results": results})[:6000],
        )
        summary = data.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            from app.schemas import RequirementResult

            models = [RequirementResult.model_validate(item) for item in results]
            return deterministic_summary(score, verdict, models)
        return summary.strip()[:800]


def build_provider(settings: Settings) -> LLMProvider:
    if settings.resolved_llm_provider == "groq":
        if not settings.groq_api_key.strip():
            raise AppError(
                503,
                "LLM_NOT_CONFIGURED",
                "LLM_PROVIDER=groq but GROQ_API_KEY is empty.",
            )
        return GroqProvider(
            api_key=settings.groq_api_key.strip(),
            model=settings.groq_model,
            timeout=settings.groq_timeout_seconds,
        )
    if settings.resolved_llm_provider != "deterministic":
        raise AppError(
            500,
            "CONFIG_ERROR",
            "LLM_PROVIDER must be auto, groq, or deterministic.",
        )
    return DeterministicProvider()
