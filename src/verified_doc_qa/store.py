"""Chunk store: Qdrant for dense vectors, plus an in-process BM25 index (own ~25 lines) over the same chunks.

Qdrant is the single source of truth (the text and metadata live in the point payload). The
BM25 index is rebuilt from it lazily after any change, which is fine up to roughly 100k chunks.
For more, use Qdrant's own sparse vectors instead.

    QDRANT_URL=http://localhost:6333   a Qdrant server (docker compose)
    INDEX_DIR=data/index               embedded on-disk Qdrant (one process at a time)
    INDEX_DIR=:memory:                 tests
"""
from __future__ import annotations

import math
import re
import uuid
from collections import Counter

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = set("a an and are as at be by can do does for from how i in is it many may much of on or the to what "
                "when where which who why with you your".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    """Okapi BM25 with Lucene's IDF, log(1 + (N - n + 0.5) / (n + 0.5)), which stays positive even for
    terms found in most documents (the classic formula hits zero or below on small corpora)."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(doc) for doc in corpus]
        self.lengths = [len(doc) for doc in corpus]
        self.avg_len = (sum(self.lengths) / len(corpus)) if corpus else 0.0
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(corpus)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def get_scores(self, query: list[str]) -> list[float]:
        scores = []
        for tf, length in zip(self.tfs, self.lengths, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
            scores.append(sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in query if t in tf))
        return scores


def _point_id(chunk_id: str) -> str:
    return str(uuid.UUID(hex=chunk_id[:32].ljust(32, "0")))


def _to_doc(payload: dict, score: float | None = None) -> Document:
    meta = {k: v for k, v in payload.items() if k != "text"}
    if score is not None:
        meta["score"] = float(score)
    return Document(page_content=payload["text"], metadata=meta)


class ChunkStore:
    def __init__(self, embeddings: Embeddings, collection: str = "docs", url: str | None = None,
                 path: str = ":memory:"):
        self.embeddings = embeddings
        self.collection = collection
        self.is_server = bool(url)
        if url:
            self.client = QdrantClient(url=url)
        elif path == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(path=path)
        self._bm25: tuple[BM25, list[Document]] | None = None

    # ------------------------------------------------------------------ writes
    def _ensure_collection(self, dim: int) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(self.collection, vectors_config=models.VectorParams(
                size=dim, distance=models.Distance.COSINE))
            if self.is_server:  # embedded Qdrant filters without indexes
                self.client.create_payload_index(self.collection, "source", models.PayloadSchemaType.KEYWORD)

    def add(self, chunks: list[Document], batch_size: int = 64) -> int:
        """Embed and upsert chunks; chunks of a source that is re-ingested replace the old ones."""
        if not chunks:
            return 0
        for source in {c.metadata["source"] for c in chunks}:
            self.delete_source(source)
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors = self.embeddings.embed_documents([c.page_content for c in batch])
            self._ensure_collection(len(vectors[0]))
            self.client.upsert(self.collection, points=[
                models.PointStruct(id=_point_id(c.metadata["chunk_id"]), vector=v,
                                   payload={"text": c.page_content, **c.metadata})
                for c, v in zip(batch, vectors, strict=True)])
        self._bm25 = None
        return len(chunks)

    def delete_source(self, source: str) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete(self.collection, points_selector=models.FilterSelector(filter=models.Filter(
                must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))])))
            self._bm25 = None

    def reset(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self._bm25 = None

    # ------------------------------------------------------------------ reads
    def all_chunks(self) -> list[Document]:
        if not self.client.collection_exists(self.collection):
            return []
        docs, offset = [], None
        while True:
            points, offset = self.client.scroll(self.collection, limit=512, offset=offset, with_payload=True)
            docs += [_to_doc(p.payload) for p in points]
            if offset is None:
                return docs

    def sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.all_chunks():
            counts[d.metadata["source"]] = counts.get(d.metadata["source"], 0) + 1
        return dict(sorted(counts.items()))

    def dense(self, query: str, k: int) -> list[Document]:
        if not self.client.collection_exists(self.collection):
            return []
        hits = self.client.query_points(self.collection, query=self.embeddings.embed_query(query), limit=k,
                                        with_payload=True).points
        return [_to_doc(h.payload, h.score) for h in hits]

    def bm25(self, query: str, k: int) -> list[Document]:
        if self._bm25 is None:
            docs = self.all_chunks()
            if not docs:
                return []
            self._bm25 = (BM25([tokenize(d.page_content) for d in docs]), docs)
        index, docs = self._bm25
        scores = index.get_scores(tokenize(query))
        ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:k]
        return [Document(page_content=docs[i].page_content, metadata={**docs[i].metadata, "score": float(scores[i])})
                for i in ranked if scores[i] > 0]
