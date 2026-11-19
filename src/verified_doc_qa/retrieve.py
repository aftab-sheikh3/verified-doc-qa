"""Two-stage retrieval: candidates (dense | bm25 | hybrid) -> optional cross-encoder rerank -> top n.

Hybrid = weighted Reciprocal Rank Fusion: score(d) = sum_i w_i / (k + rank_i(d)), with k=60.
RRF only uses ranks, so BM25 scores and cosine similarities never need to be put on one scale.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.documents import Document

from .store import ChunkStore

log = logging.getLogger(__name__)
RRF_K = 60


def rrf(rankings: list[list[Document]], weights: tuple[float, ...] | None = None, k: int = RRF_K) -> list[Document]:
    weights = weights or tuple(1.0 for _ in rankings)
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, doc in enumerate(ranking, 1):
            cid = doc.metadata["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank)
            docs.setdefault(cid, doc)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [Document(page_content=docs[c].page_content, metadata={**docs[c].metadata, "score": scores[c]})
            for c in ordered]


@lru_cache(maxsize=2)
def _cross_encoder(model: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model)


class Retriever:
    def __init__(self, store: ChunkStore, mode: str = "hybrid", reranker: str = "none",
                 reranker_model: str = "BAAI/bge-reranker-base", candidates_k: int = 20, top_n: int = 5,
                 weights: tuple[float, ...] = (0.4, 0.6)):
        if mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"retrieval mode must be dense, bm25 or hybrid, not {mode!r}")
        self.store, self.mode, self.candidates_k, self.top_n, self.weights = store, mode, candidates_k, top_n, weights
        self.reranker_model = reranker_model
        self.reranker = reranker
        if reranker == "cross-encoder":
            try:
                import sentence_transformers  # noqa: F401
            except ImportError:
                log.warning("sentence-transformers is not installed (pip install '.[rerank]'): reranking disabled")
                self.reranker = "none"

    def candidates(self, query: str) -> list[Document]:
        k = self.candidates_k
        if self.mode == "dense":
            return self.store.dense(query, k)
        if self.mode == "bm25":
            return self.store.bm25(query, k)
        return rrf([self.store.bm25(query, k), self.store.dense(query, k)], self.weights)[:k]

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        if self.reranker != "cross-encoder" or not docs:
            return docs
        scores = _cross_encoder(self.reranker_model).predict([(query, d.page_content) for d in docs])
        ranked = sorted(zip(docs, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [Document(page_content=d.page_content, metadata={**d.metadata, "rerank_score": float(s)})
                for d, s in ranked]

    def invoke(self, query: str) -> list[Document]:
        return self.rerank(query, self.candidates(query))[: self.top_n]
