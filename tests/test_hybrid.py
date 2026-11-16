"""BM25, rank fusion and the hybrid retrieval path."""

from __future__ import annotations

from app.retrieval.hybrid import bm25_scores, build_reranker, rank_by, rrf


def test_bm25_prefers_rare_exact_terms():
    docs = [
        "Built services in Python and Django.",
        "Wrote network services in Go with gRPC.",
        "Python scripts for data cleaning.",
    ]
    scores = bm25_scores("Go gRPC services", docs)
    assert max(range(3), key=lambda i: scores[i]) == 1
    assert bm25_scores("kubernetes", docs) == [0.0, 0.0, 0.0]


def test_rank_by_can_drop_zero_scores():
    assert rank_by([0.0, 2.0, 1.0], ["a", "b", "c"]) == ["b", "c", "a"]
    assert rank_by([0.0, 2.0, 1.0], ["a", "b", "c"], drop_zero=True) == ["b", "c"]


def test_rrf_rewards_agreement_between_rankings():
    fused = [doc for doc, _ in rrf([["a", "b", "c"], ["b", "c", "a"]])]
    assert fused[0] == "b"
    # an item found by only one ranking still appears
    assert "d" in dict(rrf([["a"], ["d"]]))


def test_reranker_is_off_unless_named():
    assert build_reranker("", "/tmp") is None
    assert build_reranker("none", "/tmp") is None
    assert build_reranker("Xenova/ms-marco-MiniLM-L-6-v2", "/tmp").name == "Xenova/ms-marco-MiniLM-L-6-v2"


class FlipReranker:
    name = "flip"

    def scores(self, query, docs):
        return [float(i) for i in range(len(docs))]  # last doc wins


def test_hybrid_retrieval_finds_exact_skill_and_honors_reranker(client):
    from app.db import get_sessionmaker
    from app.retrieval.service import retrieve_chunks
    from app.runtime import get_embedder, get_store

    text = (
        "Avery Chen\n\nSUMMARY\nBackend engineer.\n\nEXPERIENCE\n- Built REST APIs with FastAPI and PostgreSQL.\n\n"
        "PROJECTS\n- Command-line tool written in Go for log shipping.\n\nSKILLS\nPython, SQL, Docker\n"
    )
    resume_id = client.post("/ingest/resume", data={"text": text}).json()["resume_id"]
    db = get_sessionmaker()()
    try:
        hits = retrieve_chunks(db, get_store(), get_embedder(), resume_id, "Go", 2, mode="hybrid")
        assert any("Go" in h["text"] for h in hits)
        assert all("rrf" in h for h in hits if h["via"] != "lexical")
        plain = retrieve_chunks(
            db, get_store(), get_embedder(), resume_id, "APIs in Python", 3, mode="hybrid"
        )
        flipped = retrieve_chunks(
            db,
            get_store(),
            get_embedder(),
            resume_id,
            "APIs in Python",
            3,
            mode="hybrid",
            reranker=FlipReranker(),
            shortlist=3,
        )
        assert [h["chunk_id"] for h in flipped[:3]] == [h["chunk_id"] for h in plain[:3]][::-1]
    finally:
        db.close()
