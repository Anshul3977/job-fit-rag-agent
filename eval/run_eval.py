"""Run the labeled synthetic set and write measured metrics.

Usage:
    python eval/run_eval.py

The deterministic scorer is forced unless EVAL_USE_GROQ=1. Embeddings default
to fastembed (local all-MiniLM-L6-v2). Set EMBEDDING_BACKEND=hash only for a
smoke check; do not publish those numbers as the model eval.

This script calls the HTTP handlers in-process (Starlette TestClient), so the
latency figures exclude a real network hop and a separate browser.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LABELS_PATH = Path(__file__).resolve().parent / "labels.json"
DATASET = Path(__file__).resolve().parent / "dataset"
# EVAL_OUT_DIR lets CI write somewhere other than the published results files.
OUT_DIR = Path(os.environ.get("EVAL_OUT_DIR") or Path(__file__).resolve().parent)
RESULTS_JSON = OUT_DIR / "results.json"
RESULTS_MD = OUT_DIR / "results.md"


def _prepare_env() -> str:
    work = tempfile.mkdtemp(prefix="jobfit-eval-")
    os.environ["DATABASE_URL"] = f"sqlite:///{work}/eval.db"
    os.environ["CHROMA_DIR"] = str(Path(work) / "chroma")
    os.environ["VECTOR_BACKEND"] = "chroma"
    os.environ["ENV"] = "dev"
    os.environ["DEBUG_ROUTES"] = "false"
    os.environ["RATE_LIMIT_FIT_PER_MINUTE"] = "100000"
    os.environ.setdefault("EMBEDDING_BACKEND", "fastembed")
    os.environ.setdefault("EMBEDDING_CACHE_DIR", str((ROOT / "data" / "models").resolve()))
    os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    if os.environ.get("EVAL_USE_GROQ") != "1":
        os.environ["LLM_PROVIDER"] = "deterministic"
        os.environ["GROQ_API_KEY"] = ""
    return work


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (pct / 100)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def _adjacent(predicted: str, label: str) -> float:
    order = ["skip", "maybe", "strong"]
    if predicted == label:
        return 1.0
    if abs(order.index(predicted) - order.index(label)) == 1:
        return 0.5
    return 0.0


def _term_in_chunks(term: str, chunks: list[dict]) -> bool:
    needle = term.casefold()
    return any(needle in chunk["text"].casefold() for chunk in chunks)


def main() -> None:
    work = _prepare_env()
    from app.agent.lexicon import skill_in_text
    from app.config import get_settings

    get_settings.cache_clear()
    from app.db import get_sessionmaker, init_db, reset_engine
    from app.runtime import get_embedder, get_store, reset_runtime

    reset_engine()
    reset_runtime()

    from app.main import create_app
    from app.models import Chunk
    from app.retrieval.service import retrieve_chunks, retrieve_for_job
    from fastapi.testclient import TestClient

    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    if not labels["meta"].get("synthetic"):
        raise SystemExit("Refusing to run: eval/labels.json is not marked synthetic.")

    settings = get_settings()
    init_db()
    app = create_app()
    started_wall = time.perf_counter()
    pair_rows: list[dict] = []
    failures = 0
    resume_ids: dict[str, str] = {}
    job_ids: dict[str, str] = {}

    with TestClient(app) as client:
        for resume in labels["resumes"]:
            text = (DATASET / resume["file"]).read_text(encoding="utf-8")
            response = client.post(
                "/ingest/resume",
                files={"file": (f"{resume['id']}.txt", text.encode(), "text/plain")},
            )
            if response.status_code != 200:
                raise SystemExit(f"Resume ingest failed for {resume['id']}: {response.text}")
            resume_ids[resume["id"]] = response.json()["resume_id"]
        for job in labels["jobs"]:
            text = (DATASET / job["file"]).read_text(encoding="utf-8")
            response = client.post("/ingest/job", json={"text": text, "title": job["id"]})
            if response.status_code != 200:
                raise SystemExit(f"Job ingest failed for {job['id']}: {response.text}")
            job_ids[job["id"]] = response.json()["job_id"]

        db = get_sessionmaker()()
        store = get_store()
        embedder = get_embedder()
        lexical_hits = 0
        lexical_total = 0
        try:
            for pair in labels["pairs"]:
                resume_key = pair["resume"]
                job_key = pair["job"]
                job_text = (DATASET / f"jobs/{job_key}.txt").read_text(encoding="utf-8")
                chunks = retrieve_for_job(db, store, embedder, resume_ids[resume_key], job_text, k=5)
                for term in pair["evidence_terms"]:
                    lexical_total += 1
                    if _term_in_chunks(term, chunks):
                        lexical_hits += 1

                t0 = time.perf_counter()
                response = client.post(
                    "/fit",
                    json={"resume_id": resume_ids[resume_key], "job_id": job_ids[job_key]},
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if response.status_code != 200:
                    failures += 1
                    pair_rows.append(
                        {
                            "id": pair["id"],
                            "resume": resume_key,
                            "job": job_key,
                            "label": pair["verdict"],
                            "error": response.text,
                            "ok": False,
                        }
                    )
                    continue
                body = response.json()
                report = body["report"]
                hits_found = 0
                for skill in pair["expected_hits"]:
                    if _skill_supported(report, skill, skill_in_text):
                        hits_found += 1
                misses_held = 0
                for skill in pair["expected_misses"]:
                    if _miss_respected(report, skill, skill_in_text):
                        misses_held += 1
                citation_total = 0
                citation_valid = 0
                for match in report["matching_skills"]:
                    citation_total += 1
                    chunk = db.get(Chunk, match["evidence_chunk_id"])
                    quote = match["quote"]
                    if chunk is not None and quote and quote in chunk.text:
                        citation_valid += 1
                pair_rows.append(
                    {
                        "id": pair["id"],
                        "resume": resume_key,
                        "job": job_key,
                        "label": pair["verdict"],
                        "predicted": report["verdict"],
                        "score": report["overall_score"],
                        "ok": True,
                        "latency_ms": body["latency_ms"],
                        "client_latency_ms": round(elapsed_ms, 1),
                        "retrieve_ms": body["retrieve_ms"],
                        "generate_ms": body["generate_ms"],
                        "prompt_tokens": body["prompt_tokens"],
                        "completion_tokens": body["completion_tokens"],
                        "cost_usd": body["cost_usd"],
                        "hits_found": hits_found,
                        "hits_total": len(pair["expected_hits"]),
                        "misses_held": misses_held,
                        "misses_total": len(pair["expected_misses"]),
                        "citation_valid": citation_valid,
                        "citation_total": citation_total,
                        "model": body["model"],
                        "provider": body["provider"],
                    }
                )

            probe_rows = []
            probe_found = 0
            probe_total = 0
            for probe in labels["retrieval_probes"]:
                k = int(probe.get("k") or 5)
                chunks = retrieve_chunks(
                    db,
                    store,
                    embedder,
                    resume_ids[probe["resume"]],
                    probe["query"],
                    k,
                )
                term_hits = []
                for term in probe["expect_terms"]:
                    probe_total += 1
                    found = _term_in_chunks(term, chunks)
                    if found:
                        probe_found += 1
                    term_hits.append({"term": term, "found": found})
                probe_rows.append(
                    {
                        "id": probe["id"],
                        "resume": probe["resume"],
                        "query": probe["query"],
                        "k": k,
                        "terms": term_hits,
                    }
                )
        finally:
            db.close()

    ok_rows = [row for row in pair_rows if row.get("ok")]
    n = len(labels["pairs"])
    exact = sum(1 for row in ok_rows if row["predicted"] == row["label"])
    adjacent = sum(_adjacent(row["predicted"], row["label"]) for row in ok_rows)
    hit_found = sum(row["hits_found"] for row in ok_rows)
    hit_total = sum(row["hits_total"] for row in ok_rows)
    miss_held = sum(row["misses_held"] for row in ok_rows)
    miss_total = sum(row["misses_total"] for row in ok_rows)
    cite_valid = sum(row["citation_valid"] for row in ok_rows)
    cite_total = sum(row["citation_total"] for row in ok_rows)
    latencies = [row["latency_ms"] for row in ok_rows]
    retrieve_lat = [row["retrieve_ms"] for row in ok_rows]
    generate_lat = [row["generate_ms"] for row in ok_rows]
    prompt_tokens = [row["prompt_tokens"] for row in ok_rows]
    completion_tokens = [row["completion_tokens"] for row in ok_rows]
    costs = [row["cost_usd"] for row in ok_rows]
    labels_only = [pair["verdict"] for pair in labels["pairs"]]
    majority = max(set(labels_only), key=labels_only.count)
    majority_correct = sum(1 for verdict in labels_only if verdict == majority)

    def rate(num: int, den: int) -> float | None:
        if den == 0:
            return None
        return num / den

    provider = ok_rows[0]["provider"] if ok_rows else settings.resolved_llm_provider
    model = ok_rows[0]["model"] if ok_rows else ""
    metrics = {
        "synthetic": True,
        "pair_count": n,
        "scored_pairs": len(ok_rows),
        "failures": failures,
        "provider": provider,
        "model": model,
        "embedding_backend": settings.embedding_backend,
        "embedding_model": embedder.name,
        "vector_store": get_store().name,
        "verdict_accuracy": rate(exact, len(ok_rows)),
        "adjacent_agreement": (adjacent / len(ok_rows)) if ok_rows else None,
        "majority_class": majority,
        "majority_class_accuracy": majority_correct / n,
        "skill_hit_rate": rate(hit_found, hit_total),
        "unsupported_skill_rejection_rate": rate(miss_held, miss_total),
        "citation_validity": rate(cite_valid, cite_total),
        "lexical_recall_at_5": rate(lexical_hits, lexical_total),
        "paraphrase_recall_at_5": rate(probe_found, probe_total),
        "latency_ms_p50": _percentile(latencies, 50),
        "latency_ms_p95": _percentile(latencies, 95),
        "retrieve_ms_p50": _percentile(retrieve_lat, 50),
        "generate_ms_p50": _percentile(generate_lat, 50),
        "prompt_tokens_mean": (sum(prompt_tokens) / len(prompt_tokens)) if prompt_tokens else 0,
        "completion_tokens_mean": (sum(completion_tokens) / len(completion_tokens))
        if completion_tokens
        else 0,
        "cost_usd_mean": (sum(costs) / len(costs)) if costs else 0,
        "wall_clock_s": round(time.perf_counter() - started_wall, 2),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "workdir": work,
        "note": (
            "In-process TestClient on CPU. No browser and no extra network hop. "
            "cost_usd is 0 for the deterministic provider because it does not call Groq. "
            "Token counts for that provider are size estimates (characters/4), not API usage."
        ),
    }
    payload = {"metrics": metrics, "pairs": pair_rows, "probes": probe_rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    RESULTS_MD.write_text(_markdown(metrics, pair_rows, probe_rows), encoding="utf-8")
    print(_markdown(metrics, pair_rows, probe_rows))


def _skill_supported(report: dict, skill: str, skill_in_text) -> bool:
    for result in report["requirement_results"]:
        if result["status"] not in {"match", "partial"}:
            continue
        quote = ""
        if result.get("evidence"):
            quote = result["evidence"].get("quote") or ""
        if skill_in_text(skill, result["requirement"]) or skill_in_text(skill, quote):
            return True
    return False


def _miss_respected(report: dict, skill: str, skill_in_text) -> bool:
    for result in report["requirement_results"]:
        if result["status"] != "match":
            continue
        quote = ""
        if result.get("evidence"):
            quote = result["evidence"].get("quote") or ""
        if skill_in_text(skill, result["requirement"]) or skill_in_text(skill, quote):
            return False
    return True


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _ms(value: float) -> str:
    return f"{value:.1f} ms"


def _markdown(metrics: dict, pairs: list[dict], probes: list[dict]) -> str:
    lines = [
        "# Eval results",
        "",
        "Synthetic labeled set. These numbers were produced by `python eval/run_eval.py` and are not targets.",
        "",
        f"- Pairs: {metrics['pair_count']} scored {metrics['scored_pairs']}, failures {metrics['failures']}",
        f"- Provider: `{metrics['provider']}` / `{metrics['model']}`",
        f"- Embeddings: `{metrics['embedding_backend']}` / `{metrics['embedding_model']}`",
        f"- Vector store: `{metrics['vector_store']}`",
        f"- Machine: {metrics['platform']}; processor `{metrics['processor']}`; CPUs {metrics['cpu_count']}; Python {metrics['python']}",
        f"- Wall clock: {metrics['wall_clock_s']} s",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Verdict accuracy | {_pct(metrics['verdict_accuracy'])} |",
        f"| Adjacent agreement (off-by-one = 0.5) | {_pct(metrics['adjacent_agreement'])} |",
        f"| Majority-class baseline ({metrics['majority_class']}) | {_pct(metrics['majority_class_accuracy'])} |",
        f"| Required-skill hit rate | {_pct(metrics['skill_hit_rate'])} |",
        f"| Unsupported-skill rejection rate | {_pct(metrics['unsupported_skill_rejection_rate'])} |",
        f"| Citation validity | {_pct(metrics['citation_validity'])} |",
        f"| Lexical recall@5 | {_pct(metrics['lexical_recall_at_5'])} |",
        f"| Paraphrase-probe recall@5 | {_pct(metrics['paraphrase_recall_at_5'])} |",
        f"| Fit latency p50 | {_ms(metrics['latency_ms_p50'])} |",
        f"| Fit latency p95 | {_ms(metrics['latency_ms_p95'])} |",
        f"| Retrieve p50 | {_ms(metrics['retrieve_ms_p50'])} |",
        f"| Generate p50 | {_ms(metrics['generate_ms_p50'])} |",
        f"| Mean prompt tokens | {metrics['prompt_tokens_mean']:.1f} |",
        f"| Mean completion tokens | {metrics['completion_tokens_mean']:.1f} |",
        f"| Mean cost | ${metrics['cost_usd_mean']:.6f} |",
        "",
        metrics["note"],
        "",
        "Groq list prices used when a Groq run reports tokens (checked 2026-11-12 on the Groq models page): `openai/gpt-oss-20b` is $0.075 / 1M input tokens and $0.30 / 1M output tokens. The free tier bills $0. This run's cost column is the estimator in `app/agent/pricing.py`, which returns $0 for the deterministic provider.",
        "",
        "## Pairs",
        "",
        "| Id | Resume | Job | Label | Predicted | Score |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in pairs:
        if not row.get("ok"):
            lines.append(f"| {row['id']} | {row['resume']} | {row['job']} | {row['label']} | ERROR |  |")
            continue
        lines.append(
            f"| {row['id']} | {row['resume']} | {row['job']} | {row['label']} | {row['predicted']} | {row['score']} |"
        )
    lines.extend(["", "## Paraphrase probes", ""])
    for probe in probes:
        found = ", ".join(f"{item['term']}={'yes' if item['found'] else 'no'}" for item in probe["terms"])
        lines.append(f"- `{probe['id']}` k={probe['k']}: {found}. Query: {probe['query']}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
