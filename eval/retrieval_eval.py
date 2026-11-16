"""Retrieval benchmark: dense vs BM25 vs hybrid (RRF) vs hybrid + cross-encoder.

Usage:
    python eval/retrieval_eval.py            # writes eval/retrieval_results.{json,md}
    python eval/retrieval_eval.py --check    # also fails if metrics drop below eval/retrieval_floor.json

Queries come from the existing synthetic set in eval/labels.json:

* requirement queries: every requirement bullet the deterministic extractor pulls
  from a job post, for each of the 20 labeled resume/job pairs. A resume chunk is
  relevant when it states one of the skills that bullet names (alias-aware, the
  same lexicon the scorer uses). Bullets the resume cannot satisfy are skipped.
* paraphrase probes: the 8 hand-written probes in labels.json ("orchestrating
  containers across a cluster" -> the Kubernetes line). A chunk is relevant when
  it contains one of the probe's expected terms.

Two settings: per-resume (what /fit does: one resume's ~6 chunks) and pooled
(all resumes' chunks in one index, relevant chunks only from the target resume),
which is a harder, larger candidate set.

Requirement relevance is derived from skill names, which favors lexical methods;
the probes are the paraphrase check. Both are reported separately.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.lexicon import requirement_skills, skill_in_text, skills_in  # noqa: E402
from app.agent.llm import extract_requirements_deterministic  # noqa: E402
from app.ingest.chunking import chunk_text  # noqa: E402
from app.retrieval.embeddings import FastEmbedder  # noqa: E402
from app.retrieval.hybrid import CrossEncoderReranker, bm25_scores, rank_by, rrf  # noqa: E402

EVAL = Path(__file__).resolve().parent
LABELS = json.loads((EVAL / "labels.json").read_text(encoding="utf-8"))
CACHE = os.environ.get("EMBEDDING_CACHE_DIR", str(ROOT / "data" / "models"))
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.environ.get("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
KS = (1, 3, 4)  # /fit retrieves k=4 chunks per requirement


def load_chunks() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for resume in LABELS["resumes"]:
        text = (EVAL / "dataset" / resume["file"]).read_text(encoding="utf-8")
        out[resume["id"]] = [
            {"id": f"{resume['id']}#{i}", "resume": resume["id"], "text": c.text, "section": c.section}
            for i, c in enumerate(chunk_text(text))
        ]
    return out


def build_queries(chunks: dict[str, list[dict]]) -> list[dict]:
    queries: list[dict] = []
    for pair in LABELS["pairs"]:
        job = next(j for j in LABELS["jobs"] if j["id"] == pair["job"])
        job_text = (EVAL / "dataset" / job["file"]).read_text(encoding="utf-8")
        for req in extract_requirements_deterministic(job_text):
            skills = requirement_skills(req["text"])
            if not skills:
                continue
            relevant = {
                c["id"] for c in chunks[pair["resume"]] if any(skill_in_text(s, c["text"]) for s in skills)
            }
            if relevant:
                queries.append(
                    {
                        "kind": "requirement",
                        "id": f"{pair['id']}:{req['id']}",
                        "resume": pair["resume"],
                        "text": req["text"],
                        "relevant": relevant,
                    }
                )
    for probe in LABELS["retrieval_probes"]:
        terms = [t.casefold() for t in probe["expect_terms"]]
        relevant = {c["id"] for c in chunks[probe["resume"]] if any(t in c["text"].casefold() for t in terms)}
        queries.append(
            {
                "kind": "probe",
                "id": probe["id"],
                "resume": probe["resume"],
                "text": probe["query"],
                "relevant": relevant,
            }
        )
    return queries


class Methods:
    def __init__(self, chunks: list[dict], embedder: FastEmbedder, reranker: CrossEncoderReranker) -> None:
        self.chunks = chunks
        self.ids = [c["id"] for c in chunks]
        self.texts = [c["text"] for c in chunks]
        vecs = np.array(embedder.embed(self.texts))
        self.vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        self.embedder = embedder
        self.reranker = reranker
        self.rerank_ms: list[float] = []

    def _cos(self, query: str) -> np.ndarray:
        q = np.array(self.embedder.embed([query[:2000]])[0])
        return self.vecs @ (q / np.linalg.norm(q))

    def dense(self, query: str) -> list[str]:
        return rank_by(self._cos(query).tolist(), self.ids)

    def bm25(self, query: str) -> list[str]:
        # Zero-score docs go last in their original order so every method ranks every chunk.
        ranked = rank_by(bm25_scores(query, self.texts), self.ids, drop_zero=True)
        return ranked + [i for i in self.ids if i not in ranked]

    def dense_backfill(self, query: str, k: int = 4) -> list[str]:
        """The pre-upgrade production path: dense top-k, then a chunk for any named skill not yet covered."""
        cos = self._cos(query)
        order = rank_by(cos.tolist(), self.ids)
        found = [(cos[self.ids.index(i)], i) for i in order[:k]]
        by_id = dict(zip(self.ids, self.texts))
        for skill in skills_in(query):
            if any(skill_in_text(skill, by_id[i]) for _, i in found):
                continue
            for i in self.ids:
                if i not in {x for _, x in found} and skill_in_text(skill, by_id[i]):
                    found.append((0.35, i))
                    break
        found.sort(key=lambda p: p[0], reverse=True)
        head = [i for _, i in found]
        return head + [i for i in order if i not in head]

    def hybrid(self, query: str) -> list[str]:
        return [i for i, _ in rrf([self.dense(query), self.bm25(query)])]

    def hybrid_rerank(self, query: str, shortlist: int = 10) -> list[str]:
        fused = self.hybrid(query)
        head = fused[:shortlist]
        by_id = dict(zip(self.ids, self.texts))
        t0 = time.perf_counter()
        scores = self.reranker.scores(query, [by_id[i] for i in head])
        self.rerank_ms.append((time.perf_counter() - t0) * 1000)
        return rank_by(scores, head) + fused[shortlist:]


METHODS = ["dense", "bm25", "dense_backfill", "hybrid", "hybrid_rerank"]


def metrics(rankings: list[tuple[list[str], set[str]]]) -> dict:
    out: dict[str, float] = {}
    for k in KS:
        out[f"recall@{k}"] = float(np.mean([len(set(r[:k]) & rel) / len(rel) for r, rel in rankings]))
        out[f"hit@{k}"] = float(np.mean([1.0 if set(r[:k]) & rel else 0.0 for r, rel in rankings]))
    out["mrr"] = float(
        np.mean([next((1 / (i + 1) for i, d in enumerate(r) if d in rel), 0.0) for r, rel in rankings])
    )
    out["n"] = len(rankings)
    return out


def run() -> dict:
    chunks = load_chunks()
    queries = build_queries(chunks)
    embedder = FastEmbedder(EMBED_MODEL, CACHE)
    reranker = CrossEncoderReranker(RERANK_MODEL, CACHE)
    per_resume = {rid: Methods(cs, embedder, reranker) for rid, cs in chunks.items()}
    pooled = Methods([c for cs in chunks.values() for c in cs], embedder, reranker)

    results: dict = {}
    for setting, pick in (("per_resume", lambda q: per_resume[q["resume"]]), ("pooled", lambda q: pooled)):
        results[setting] = {}
        for kind in ("requirement", "probe", "all"):
            qs = [q for q in queries if kind == "all" or q["kind"] == kind]
            results[setting][kind] = {
                m: metrics([(getattr(pick(q), m)(q["text"]), q["relevant"]) for q in qs]) for m in METHODS
            }
    rerank_ms = per_resume[next(iter(per_resume))].rerank_ms + pooled.rerank_ms
    return {
        "meta": {
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "synthetic": True,
            "queries": {
                "requirement": sum(q["kind"] == "requirement" for q in queries),
                "probe": sum(q["kind"] == "probe" for q in queries),
            },
            "chunks_per_resume": {rid: len(cs) for rid, cs in chunks.items()},
            "embedding_model": EMBED_MODEL,
            "reranker_model": RERANK_MODEL,
            "rrf_k": 60,
            "rerank_shortlist": 10,
            "machine": f"{platform.platform()}, {os.cpu_count()} CPUs, Python {platform.python_version()}",
            "rerank_ms_sample_p50": float(np.median(rerank_ms)) if rerank_ms else None,
        },
        "results": results,
    }


def to_markdown(data: dict) -> str:
    m = data["meta"]
    lines = [
        "# Retrieval benchmark",
        "",
        f"Generated by `python eval/retrieval_eval.py` on {m['date']} (UTC). Synthetic data: "
        f"{m['queries']['requirement']} requirement queries from the 20 labeled pairs and {m['queries']['probe']} paraphrase probes.",
        f"Embeddings `{m['embedding_model']}`, reranker `{m['reranker_model']}` on the top {m['rerank_shortlist']} fused chunks, RRF k={m['rrf_k']}. {m['machine']}.",
        "",
        "`dense_backfill` is the retrieval that shipped before this change (dense top-k plus a lexical save for named skills).",
        "",
    ]
    for setting, title in (
        ("per_resume", "Per resume (what /fit searches, ~6 chunks)"),
        ("pooled", "Pooled index (all resumes, 30 chunks)"),
    ):
        for kind in ("requirement", "probe", "all"):
            rows = data["results"][setting][kind]
            n = rows["dense"]["n"]
            lines += [
                f"### {title}: {kind} queries (n={n})",
                "",
                "| Method | Recall@1 | Recall@3 | Recall@4 | Hit@1 | MRR |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for name, r in rows.items():
                lines.append(
                    f"| {name} | {r['recall@1']:.3f} | {r['recall@3']:.3f} | {r['recall@4']:.3f} | {r['hit@1']:.3f} | {r['mrr']:.3f} |"
                )
            lines.append("")
    if m.get("rerank_ms_sample_p50") is not None:
        lines.append(
            f"Cross-encoder rerank of 10 chunks: p50 {m['rerank_ms_sample_p50']:.1f} ms on this machine."
        )
    return "\n".join(lines) + "\n"


def check(data: dict, floor_path: Path) -> list[str]:
    floor = json.loads(floor_path.read_text(encoding="utf-8"))
    failures = []
    for key, minimum in floor["min"].items():
        setting, kind, method, metric = key.split(".")
        got = data["results"][setting][kind][method][metric]
        if got + 1e-9 < minimum:
            failures.append(f"{key}: {got:.3f} < floor {minimum:.3f}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="fail when metrics fall below eval/retrieval_floor.json"
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    data = run()
    if not args.no_write:
        (EVAL / "retrieval_results.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        (EVAL / "retrieval_results.md").write_text(to_markdown(data), encoding="utf-8")
    print(to_markdown(data))
    if args.check:
        failures = check(data, EVAL / "retrieval_floor.json")
        if failures:
            print("Retrieval regression:\n  " + "\n  ".join(failures))
            sys.exit(1)
        print("Retrieval floor: OK")


if __name__ == "__main__":
    main()
