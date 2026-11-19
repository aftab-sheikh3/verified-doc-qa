"""Settings, read from environment variables (and a .env file if present)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHAT_MODELS = {"ollama": "qwen3:8b", "openai": "gpt-4o-mini", "anthropic": "claude-sonnet-5-5", "fake": "fake"}
DEFAULT_EMBED_MODELS = {"ollama": "nomic-embed-text", "openai": "text-embedding-3-small", "fake": "hashing-512"}


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _floats(name: str, default: str) -> tuple[float, ...]:
    return tuple(float(x) for x in _env(name, default).split(","))


@dataclass(frozen=True)
class Settings:
    provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "ollama").lower() or "ollama")
    model: str | None = field(default_factory=lambda: os.getenv("LLM_MODEL") or None)
    # Embeddings default to the chat provider (Anthropic has no embedding API -> Ollama).
    embed_provider: str | None = field(default_factory=lambda: os.getenv("EMBED_PROVIDER") or None)
    embed_model: str | None = field(default_factory=lambda: os.getenv("EMBED_MODEL") or None)
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    # Qdrant: a server URL, or an embedded on-disk store at index_dir (":memory:" for tests).
    qdrant_url: str | None = field(default_factory=lambda: os.getenv("QDRANT_URL") or None)
    index_dir: str = field(default_factory=lambda: _env("INDEX_DIR", str(ROOT / "data" / "index")))
    collection: str = field(default_factory=lambda: _env("COLLECTION", "docs"))
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "100")))
    contextual: bool = field(default_factory=lambda: _env("CONTEXTUAL_CHUNKS", "0") == "1")
    candidates_k: int = field(default_factory=lambda: int(_env("CANDIDATES_K", "20")))
    top_n: int = field(default_factory=lambda: int(_env("TOP_N", "5")))
    hybrid_weights: tuple[float, ...] = field(default_factory=lambda: _floats("HYBRID_WEIGHTS", "0.4,0.6"))
    retrieval: str = field(default_factory=lambda: _env("RETRIEVAL", "hybrid"))  # dense | bm25 | hybrid
    reranker: str = field(default_factory=lambda: _env("RERANKER", "cross-encoder"))  # cross-encoder | none
    reranker_model: str = field(default_factory=lambda: _env("RERANKER_MODEL", "BAAI/bge-reranker-base"))
    max_attempts: int = field(default_factory=lambda: int(_env("MAX_ATTEMPTS", "2")))

    @property
    def model_name(self) -> str:
        return self.model or DEFAULT_CHAT_MODELS.get(self.provider, "")

    @property
    def embedding_provider(self) -> str:
        if self.embed_provider:
            return self.embed_provider
        return "ollama" if self.provider == "anthropic" else self.provider

    @property
    def embedding_model(self) -> str:
        return self.embed_model or DEFAULT_EMBED_MODELS.get(self.embedding_provider, "")


def get_settings() -> Settings:
    return Settings()
