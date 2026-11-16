"""Lexical scoring, rank fusion and optional cross-encoder reranking.

Dense MiniLM search is good at paraphrase ("orchestrating containers" finds a
Kubernetes line) and weak at rare exact tokens ("Go", "pytest"). BM25 is the
opposite. Reciprocal rank fusion combines the two rankings without having to
calibrate cosine scores against BM25 scores. A cross-encoder can then reorder
the fused shortlist by reading query and chunk together.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from app.retrieval.embeddings import tokenize

RRF_K = 60


def bm25_scores(query: str, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Okapi BM25 of `query` against each doc. IDF is computed over `docs` only."""
    tokenized = [tokenize(doc) for doc in docs]
    n = len(tokenized)
    if n == 0:
        return []
    avgdl = sum(len(t) for t in tokenized) / n or 1.0
    df: Counter[str] = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    terms = set(tokenize(query))
    scores: list[float] = []
    for tokens in tokenized:
        tf = Counter(tokens)
        dl = len(tokens)
        score = 0.0
        for term in terms:
            freq = tf.get(term, 0)
            if not freq:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * dl / avgdl))
        scores.append(score)
    return scores


def rank_by(scores: Sequence[float], ids: Sequence[str], drop_zero: bool = False) -> list[str]:
    pairs = [(s, i) for s, i in zip(scores, ids) if not (drop_zero and s <= 0)]
    pairs.sort(key=lambda p: p[0], reverse=True)
    return [i for _, i in pairs]


def rrf(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda p: p[1], reverse=True)


class Reranker(Protocol):
    name: str

    def scores(self, query: str, docs: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """fastembed cross-encoder (ONNX). Loaded on first use."""

    def __init__(self, model_name: str, cache_dir: str) -> None:
        self.name = model_name
        self.cache_dir = cache_dir
        self._model = None

    def scores(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=self.name, cache_dir=self.cache_dir)
        return [float(s) for s in self._model.rerank(query, docs)]


def build_reranker(model_name: str, cache_dir: str) -> Reranker | None:
    name = (model_name or "").strip()
    if not name or name.lower() in {"none", "off", "false"}:
        return None
    return CrossEncoderReranker(name, cache_dir)
