"""Local embeddings.

fastembed runs all-MiniLM-L6-v2 as ONNX, which keeps the image smaller than
torch. The hash embedder is a test double: shared words land near each other,
but it is not a semantic model and must not be used for published eval numbers.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.#-]{1,}")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Feature hashing of unigrams and bigrams. Deterministic and offline."""

    dim = 384
    name = "hash-384"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float64)
            tokens = tokenize(text)
            for token in tokens:
                digest = hashlib.sha256(token.encode()).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[index] += sign
            for left, right in zip(tokens, tokens[1:]):
                digest = hashlib.sha256(f"{left}_{right}".encode()).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[index] += 0.5 * sign
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            vectors.append(vec.tolist())
        return vectors


class FastEmbedder:
    dim = 384

    def __init__(self, model_name: str, cache_dir: str) -> None:
        self.name = model_name
        self.cache_dir = cache_dir
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.name, cache_dir=self.cache_dir)
        vectors = []
        for vector in self._model.embed(texts):
            as_list = vector.tolist()
            self.dim = len(as_list)
            vectors.append(as_list)
        return vectors


def build_embedder(backend: str, model_name: str, cache_dir: str) -> Embedder:
    if backend == "hash":
        return HashEmbedder()
    if backend == "fastembed":
        return FastEmbedder(model_name, cache_dir)
    raise ValueError(f"Unknown embedding backend: {backend}")
