"""LangGraph fit pipeline.

extract requirements -> retrieve evidence per requirement -> judge ->
self-check citations -> score -> explain.

Score and self-check are code, not a prompt, so an uncited skill cannot
survive into the report.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent.llm import LLMProvider
from app.agent.scoring import assemble_report, self_check


class FitState(TypedDict, total=False):
    resume_id: str
    job_text: str
    job_title: str
    requirements: list[dict]
    evidence: dict[str, list[dict]]
    judgments: list[dict]
    results: list[dict]
    score: int
    verdict: str
    summary: str
    report: dict
    timings: dict[str, int]


RetrieveFn = Callable[[str, int], list[dict]]


def _stamp(state: FitState, node: str, started: float, update: dict) -> dict:
    timings = dict(state.get("timings") or {})
    timings[node] = int((time.perf_counter() - started) * 1000)
    update["timings"] = timings
    return update


def compile_fit_graph(provider: LLMProvider, retrieve: RetrieveFn):
    def extract_requirements(state: FitState) -> dict:
        started = time.perf_counter()
        requirements = provider.extract_requirements(state["job_text"])
        return _stamp(state, "extract_requirements", started, {"requirements": requirements})

    def retrieve_evidence(state: FitState) -> dict:
        started = time.perf_counter()
        evidence: dict[str, list[dict]] = {}
        for requirement in state.get("requirements") or []:
            evidence[requirement["id"]] = retrieve(requirement["text"], 4)
        return _stamp(state, "retrieve_evidence", started, {"evidence": evidence})

    def judge_matches(state: FitState) -> dict:
        started = time.perf_counter()
        judgments = provider.judge(state.get("requirements") or [], state.get("evidence") or {})
        return _stamp(state, "judge_matches", started, {"judgments": judgments})

    def check_citations(state: FitState) -> dict:
        started = time.perf_counter()
        checked = self_check(state.get("judgments") or [], state.get("evidence") or {})
        return _stamp(state, "self_check", started, {"judgments": checked})

    def score(state: FitState) -> dict:
        started = time.perf_counter()
        report = assemble_report(
            requirements=state.get("requirements") or [],
            judgments=state.get("judgments") or [],
            evidence=state.get("evidence") or {},
            summary="",
            model=provider.model_name,
            provider=provider.provider_id,
        )
        results = [item.model_dump() for item in report.requirement_results]
        return _stamp(
            state,
            "score",
            started,
            {
                "results": results,
                "score": report.overall_score,
                "verdict": report.verdict,
                "report": report.model_dump(),
            },
        )

    def explain(state: FitState) -> dict:
        started = time.perf_counter()
        summary = provider.write_summary(
            state.get("job_title") or "Role",
            int(state.get("score") or 0),
            str(state.get("verdict") or "skip"),
            state.get("results") or [],
        )
        report = assemble_report(
            requirements=state.get("requirements") or [],
            judgments=state.get("judgments") or [],
            evidence=state.get("evidence") or {},
            summary=summary,
            model=provider.model_name,
            provider=provider.provider_id,
        )
        return _stamp(
            state,
            "explain",
            started,
            {"summary": report.summary, "report": report.model_dump()},
        )

    graph = StateGraph(FitState)
    graph.add_node("extract_requirements", extract_requirements)
    graph.add_node("retrieve_evidence", retrieve_evidence)
    graph.add_node("judge_matches", judge_matches)
    graph.add_node("self_check", check_citations)
    graph.add_node("score", score)
    graph.add_node("explain", explain)
    graph.add_edge(START, "extract_requirements")
    graph.add_edge("extract_requirements", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "judge_matches")
    graph.add_edge("judge_matches", "self_check")
    graph.add_edge("self_check", "score")
    graph.add_edge("score", "explain")
    graph.add_edge("explain", END)
    return graph.compile()
