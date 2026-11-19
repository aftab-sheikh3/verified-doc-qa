from pathlib import Path

import pytest

from verified_doc_qa.config import Settings
from verified_doc_qa.ingest import index_paths
from verified_doc_qa.llm import HashingEmbeddings
from verified_doc_qa.retrieve import Retriever
from verified_doc_qa.store import ChunkStore

CORPUS = Path(__file__).resolve().parents[1] / "corpus"


@pytest.fixture
def settings() -> Settings:
    return Settings(provider="fake", index_dir=":memory:", reranker="none", max_attempts=2)


@pytest.fixture(scope="session")
def corpus_store() -> ChunkStore:
    """The bundled corpus indexed in memory with hashing embeddings (no model, no network)."""
    store = ChunkStore(HashingEmbeddings(), "test", path=":memory:")
    index_paths(store, [CORPUS])
    return store


@pytest.fixture
def retriever(corpus_store) -> Retriever:
    return Retriever(corpus_store, "hybrid", "none", top_n=4)
