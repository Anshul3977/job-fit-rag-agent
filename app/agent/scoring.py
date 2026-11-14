"""Pure scoring and citation checks.

The model, or the deterministic stand-in, only proposes judgments. The score
is a function of those judgments after quotes have been checked against the
retrieved resume text. A required miss blocks a "strong" verdict even when
the numeric score is high.
"""

from __future__ import annotations

import math
import re

from app.agent.lexicon import (
    ADJACENT,
    aliases_for,
    requirement_skills,
    significant_words,
    skill_in_text,
    skill_is_claimed,
    skills_in,
    term_in_text,
)
from app.schemas import (
    Citation,
    EvidenceQuote,
    FitReport,
    Gap,
    MatchingSkill,
    RequirementResult,
)

MATCH_CREDIT = {"match": 1.0, "partial": 0.45, "gap": 0.0}
STRONG_COVERAGE = 0.999  # every required item must be a full match
MAYBE_COVERAGE = 0.45
STRONG_SCORE = 75
MAYBE_SCORE = 45


def normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def snap_quote(quote: str, text: str) -> str | None:
    """Return an exact substring of `text` or None when the quote is unsupported."""
    if not quote or not text:
        return None
    stripped = quote.strip()
    if stripped and stripped in text:
        return stripped
    collapsed = normalize_ws(quote)
    if len(collapsed) < 8:
        return None
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    for sentence in sentences:
        if collapsed in normalize_ws(sentence) and sentence in text:
            return sentence
    return None


def exact_window(text: str, needle: str, radius: int = 160) -> str | None:
    match = re.search(rf"\b{re.escape(needle)}\b", text, re.I)
    if match is None and needle.casefold() in text.casefold():
        start = text.casefold().find(needle.casefold())
        end = start + len(needle)
    elif match is not None:
        start, end = match.start(), match.end()
    else:
        return None
    left = start
    right = end
    while left > 0 and text[left - 1] not in "\n.?,;" and start - left < radius:
        left -= 1
    while right < len(text) and text[right] not in "\n.?,;" and right - end < radius:
        right += 1
    if right < len(text) and text[right] in ".?":
        right += 1
    snippet = text[left:right].strip()
    if snippet and snippet in text:
        return snippet
    return text[start:end]


def _years_required(requirement: str) -> int | None:
    match = re.search(r"(\d+)\+?\s+years?", requirement, re.I)
    if not match:
        return None
    return int(match.group(1))


def _years_met(text: str, years: int) -> bool:
    for match in re.finditer(r"(\d+)\+?\s+years?", text, re.I):
        if int(match.group(1)) >= years:
            return True
    return False


def judge_requirement(requirement: str, evidence: list[dict]) -> dict:
    """Decide match, partial, or gap using only retrieved chunk text."""
    needed = requirement_skills(requirement)
    if not evidence:
        return {
            "status": "gap",
            "chunk_id": "",
            "quote": "",
            "rationale": "No resume chunk was retrieved for this requirement.",
        }

    if not needed:
        words = significant_words(requirement)
        best: tuple[float, dict, list[str]] | None = None
        for chunk in evidence:
            overlap = [word for word in words if term_in_text(word, chunk["text"])]
            ratio = (len(overlap) / len(words)) if words else 0.0
            rank = (ratio, chunk["score"])
            if best is None or rank > (best[0], best[1]["score"]):
                best = (ratio, chunk, overlap)
        assert best is not None
        ratio, chunk, overlap = best
        if ratio >= 0.8 and overlap:
            quote = exact_window(chunk["text"], overlap[0]) or ""
            return {
                "status": "match",
                "chunk_id": chunk["chunk_id"],
                "quote": quote,
                "rationale": f"Retrieved {chunk['section']} text overlaps this requirement.",
            }
        if ratio >= 0.5 and overlap:
            quote = exact_window(chunk["text"], overlap[0]) or ""
            return {
                "status": "partial",
                "chunk_id": chunk["chunk_id"],
                "quote": quote,
                "rationale": "Only some of the wording shows up in the retrieved resume text.",
            }
        return {
            "status": "gap",
            "chunk_id": "",
            "quote": "",
            "rationale": "The retrieved resume chunks do not state this requirement.",
        }

    or_mode = " or " in requirement.casefold()
    best_skill: tuple[float, float, dict, list[str]] | None = None
    negated: list[str] = []
    for chunk in evidence:
        found = [skill for skill in needed if skill_is_claimed(skill, chunk["text"])]
        for skill in needed:
            if skill not in found and skill_in_text(skill, chunk["text"]) and skill not in negated:
                negated.append(skill)
        if or_mode and found:
            ratio = 1.0
        else:
            ratio = len(found) / len(needed)
        rank = (ratio, chunk["score"])
        if best_skill is None or rank > (best_skill[0], best_skill[1]):
            best_skill = (ratio, chunk["score"], chunk, found)
    assert best_skill is not None
    ratio, _score, chunk, found = best_skill
    missing = [skill for skill in needed if skill not in found]
    quote = ""
    for skill in found:
        for needle in aliases_for(skill):
            if not skill_is_claimed(skill, chunk["text"]):
                continue
            window = exact_window(chunk["text"], needle) or ""
            if window and skill_is_claimed(skill, window):
                quote = window
                break
        if quote:
            break

    if ratio == 1.0 and quote:
        status = "match"
        rationale = f"Retrieved {chunk['section']} text mentions {', '.join(found)}."
        years = _years_required(requirement)
        if years is not None and not _years_met(chunk["text"], years):
            status = "partial"
            rationale = f"The resume shows {', '.join(found)} but does not state {years}+ years of that work."
    elif ratio >= 0.5 and quote:
        status = "partial"
        rationale = (
            f"Retrieved {chunk['section']} text mentions {', '.join(found)}. "
            f"Not found there: {', '.join(missing)}."
        )
    else:
        status = "gap"
        quote = ""
        denied = [skill for skill in needed if skill in negated]
        if denied:
            rationale = (
                "Retrieved text mentions "
                + ", ".join(denied)
                + " only to say it is absent, so it is not counted."
            )
        else:
            rationale = "None of the retrieved resume chunks mention " + ", ".join(needed) + "."
        chunk = {"chunk_id": "", "section": ""}

    return {
        "status": status,
        "chunk_id": chunk["chunk_id"],
        "quote": quote,
        "rationale": rationale,
    }


def self_check(judgments: list[dict], evidence: dict[str, list[dict]]) -> list[dict]:
    """Drop any match or partial whose quote is not an exact retrieved substring."""
    checked: list[dict] = []
    for judgment in judgments:
        status = judgment["status"]
        if status not in {"match", "partial"}:
            checked.append({**judgment, "status": "gap", "quote": "", "chunk_id": ""})
            continue
        chunks = {item["chunk_id"]: item for item in evidence.get(judgment["requirement_id"], [])}
        chunk = chunks.get(judgment.get("chunk_id") or "")
        snapped = snap_quote(judgment.get("quote") or "", chunk["text"]) if chunk else None
        if snapped is None:
            rationale = judgment.get("rationale") or ""
            note = " Self-check removed this claim because the quote is not in the retrieved resume text."
            checked.append(
                {
                    **judgment,
                    "status": "gap",
                    "quote": "",
                    "chunk_id": "",
                    "rationale": (rationale + note).strip(),
                }
            )
            continue
        checked.append({**judgment, "quote": snapped})
    return checked


def coverage_and_score(rows: list[tuple[str, str]]) -> tuple[float, int]:
    """rows are (kind, status). Coverage ignores preferred items."""
    required = [(kind, status) for kind, status in rows if kind == "required"]
    preferred = [(kind, status) for kind, status in rows if kind == "preferred"]
    if required:
        coverage = sum(MATCH_CREDIT[status] for _kind, status in required) / len(required)
    else:
        coverage = 0.0
    req_credit = coverage
    if preferred:
        pref_credit = sum(MATCH_CREDIT[status] for _kind, status in preferred) / len(preferred)
        raw = 0.85 * req_credit + 0.15 * pref_credit
    else:
        raw = req_credit
    # Half-up, so 92.5 becomes 93. Python's round() uses bankers' rounding.
    score = max(0, min(100, int(math.floor(100 * raw + 0.5))))
    return coverage, score


def verdict_for(coverage: float, score: int, rows: list[tuple[str, str]]) -> str:
    required_statuses = [status for kind, status in rows if kind == "required"]
    all_required_match = bool(required_statuses) and all(status == "match" for status in required_statuses)
    if all_required_match and score >= STRONG_SCORE and coverage >= STRONG_COVERAGE:
        return "strong"
    if coverage >= MAYBE_COVERAGE and score >= MAYBE_SCORE:
        return "maybe"
    return "skip"


def severity_for(kind: str, status: str) -> str:
    if status == "match":
        return "low"
    if kind == "required" and status == "gap":
        return "high"
    if kind == "required" and status == "partial":
        return "medium"
    return "low"


def stretch_flags(
    requirements: list[dict],
    judgments: list[dict],
    evidence: dict[str, list[dict]],
) -> list[str]:
    by_id = {item["requirement_id"]: item for item in judgments}
    flags: list[str] = []
    pool_text = "\n".join(chunk["text"] for chunks in evidence.values() for chunk in chunks)
    for requirement in requirements:
        judgment = by_id.get(requirement["id"])
        if judgment is None:
            continue
        if judgment["status"] == "partial":
            flags.append(f"Partial cover: {requirement['text']}")
        if judgment["status"] != "gap":
            continue
        for skill in skills_in(requirement["text"]):
            for neighbor in ADJACENT.get(skill, []):
                if skill_in_text(neighbor, pool_text) and not skill_in_text(skill, pool_text):
                    flags.append(
                        f"Adjacent, not a match: the job asks for {skill}; "
                        f"retrieved text mentions {neighbor}."
                    )
    # Preserve order while dropping duplicates.
    seen: set[str] = set()
    unique: list[str] = []
    for flag in flags:
        if flag not in seen:
            seen.add(flag)
            unique.append(flag)
    return unique[:6]


def recommended_edits(results: list[RequirementResult]) -> list[str]:
    edits: list[str] = []
    for result in results:
        if result.kind != "required" or result.status == "match":
            continue
        label = result.requirement
        if len(label) > 90:
            label = label[:87].rstrip() + "..."
        edits.append(
            f"If you have done this, add one bullet that names “{label}” and what you shipped. "
            "Leave it off if you have not."
        )
        if len(edits) == 4:
            break
    return edits


def deterministic_summary(score: int, verdict: str, results: list[RequirementResult]) -> str:
    required = [item for item in results if item.kind == "required"]
    preferred = [item for item in results if item.kind == "preferred"]
    required_hits = sum(1 for item in required if item.status == "match")
    preferred_hits = sum(1 for item in preferred if item.status == "match")
    sentence = (
        f"{verdict.capitalize()} fit at {score}/100. "
        f"The resume supports {required_hits} of {len(required)} required items"
    )
    if preferred:
        sentence += f" and {preferred_hits} of {len(preferred)} preferred items."
    else:
        sentence += "."
    sentence += " Quotes are copied from retrieved resume chunks. Anything without that support is a gap."
    return sentence


def build_results(
    requirements: list[dict],
    judgments: list[dict],
    evidence: dict[str, list[dict]],
) -> list[RequirementResult]:
    by_id = {item["requirement_id"]: item for item in judgments}
    results: list[RequirementResult] = []
    for requirement in requirements:
        judgment = by_id.get(
            requirement["id"],
            {
                "status": "gap",
                "chunk_id": "",
                "quote": "",
                "rationale": "The judge did not return this requirement.",
            },
        )
        status = judgment["status"]
        quote_model = None
        if status in {"match", "partial"} and judgment.get("quote") and judgment.get("chunk_id"):
            chunks = {item["chunk_id"]: item for item in evidence.get(requirement["id"], [])}
            chunk = chunks.get(judgment["chunk_id"])
            if chunk is not None:
                quote_model = EvidenceQuote(
                    chunk_id=chunk["chunk_id"],
                    quote=judgment["quote"],
                    section=chunk["section"],
                    score=round(float(chunk["score"]), 4),
                    page=chunk.get("page"),
                )
        results.append(
            RequirementResult(
                requirement=requirement["text"],
                kind=requirement["kind"],
                status=status,
                severity=severity_for(requirement["kind"], status),  # type: ignore[arg-type]
                rationale=judgment.get("rationale") or "",
                evidence=quote_model,
            )
        )
    return results


def assemble_report(
    *,
    requirements: list[dict],
    judgments: list[dict],
    evidence: dict[str, list[dict]],
    summary: str,
    model: str,
    provider: str,
    provider_note: str = "",
) -> FitReport:
    results = build_results(requirements, judgments, evidence)
    rows = [(item.kind, item.status) for item in results]
    coverage, score = coverage_and_score(rows)
    verdict = verdict_for(coverage, score, rows)
    # Summary from the caller was built with a preliminary score. Rebuild so
    # the paragraph cannot disagree with the number, unless the caller already
    # passed a Groq paragraph. Groq summaries are kept when they are non-empty
    # and do not contain a conflicting "N/100" with a different N.
    if provider == "deterministic" or not summary.strip():
        summary = deterministic_summary(score, verdict, results)
    else:
        summary = _align_summary_score(summary, score, verdict, results)

    matching = [
        MatchingSkill(
            skill=item.requirement,
            evidence_chunk_id=item.evidence.chunk_id,
            quote=item.evidence.quote,
        )
        for item in results
        if item.status == "match" and item.evidence is not None
    ]
    gaps = [
        Gap(requirement=item.requirement, why_it_is_a_gap=item.rationale, severity=item.severity)
        for item in results
        if item.status == "gap"
    ]
    citations: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for item in results:
        if item.evidence is None:
            continue
        key = (item.evidence.chunk_id, item.evidence.quote)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                chunk_id=item.evidence.chunk_id,
                quote=item.evidence.quote,
                section=item.evidence.section,
            )
        )
    return FitReport(
        overall_score=score,
        verdict=verdict,  # type: ignore[arg-type]
        summary=summary,
        matching_skills=matching,
        gaps=gaps,
        stretch_flags=stretch_flags(requirements, judgments, evidence),
        recommended_resume_edits=recommended_edits(results),
        citations=citations,
        requirement_results=results,
        model=model,
        provider=provider,  # type: ignore[arg-type]
        provider_note=provider_note,
    )


def _align_summary_score(
    summary: str,
    score: int,
    verdict: str,
    results: list[RequirementResult],
) -> str:
    """If the model wrote a different score, prepend the computed one."""
    match = re.search(r"\b(\d{1,3})\s*/\s*100\b", summary)
    if match and int(match.group(1)) == score and verdict in summary.casefold():
        return summary.strip()
    lead = deterministic_summary(score, verdict, results)
    return lead
