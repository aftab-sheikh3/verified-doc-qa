"""Chat-model and embedding factories with a provider switch, plus offline fakes for tests.

    LLM_PROVIDER=ollama     (default) local model via Ollama - `ollama pull qwen3:8b`
    LLM_PROVIDER=openai     OPENAI_API_KEY
    LLM_PROVIDER=anthropic  ANTHROPIC_API_KEY
    LLM_PROVIDER=fake       offline scripted model (tests, CI) - not meant to give real answers

Embeddings: ollama (nomic-embed-text) | openai (text-embedding-3-small) | fake (HashingEmbeddings:
a deterministic bag-of-words hash, so offline tests still retrieve sensibly).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from .config import Settings, get_settings


def get_chat_model(settings: Settings | None = None, temperature: float = 0.0) -> BaseChatModel:
    s = settings or get_settings()
    if s.provider == "ollama":
        from langchain_ollama import ChatOllama

        return _traced(ChatOllama(model=s.model_name, base_url=s.ollama_url, temperature=temperature))
    if s.provider == "openai":
        from langchain_openai import ChatOpenAI

        return _traced(ChatOpenAI(model=s.model_name, temperature=temperature))
    if s.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return _traced(ChatAnthropic(model=s.model_name, temperature=temperature, max_tokens=4096))
    if s.provider == "fake":
        return ScriptedChatModel()
    raise ValueError(f"Unknown LLM_PROVIDER {s.provider!r}: use ollama, openai, anthropic or fake")


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    s = settings or get_settings()
    provider = s.embedding_provider
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=s.embedding_model, base_url=s.ollama_url)
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=s.embedding_model)
    if provider == "fake":
        return HashingEmbeddings()
    raise ValueError(f"Unknown EMBED_PROVIDER {provider!r}: use ollama, openai or fake")


class HashingEmbeddings(Embeddings):
    """Feature-hashed bag of words (with light suffix stripping), L2-normalised. No model, no network."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            token = re.sub(r"(ing|ed|es|s)$", "", token) if len(token) > 4 else token
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def _traced(model: BaseChatModel) -> BaseChatModel:
    """Attach a Langfuse callback when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are set."""
    import os

    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse.langchain import CallbackHandler

            model.callbacks = [*(model.callbacks or []), CallbackHandler()]
        except ImportError:
            pass
    return model


# --------------------------------------------------------------------------- fake model
def _dummy(annotation: Any) -> Any:
    from typing import get_args, get_origin
    origin = get_origin(annotation)
    if origin in (list, tuple, set):
        return []
    if origin is dict:
        return {}
    if origin is not None:  # Optional / Union / Literal
        args = [a for a in get_args(annotation) if a is not type(None)]
        return _dummy(args[0]) if args else None
    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return dummy_instance(annotation)
        for kind, value in ((bool, True), (int, 1), (float, 1.0), (str, "fake")):
            if issubclass(annotation, kind):
                return value
    return annotation if isinstance(annotation, str) else "fake"  # Literal values


def dummy_instance(schema: type[BaseModel]) -> BaseModel:
    values = {n: _dummy(f.annotation) for n, f in schema.model_fields.items() if f.is_required()}
    try:
        return schema(**values)
    except Exception:
        return schema.model_construct(**values)


class ScriptedChatModel(BaseChatModel):
    """Offline model. `script` items are consumed in order: a str = text reply, a dict
    {"tool": name, "args": {...}} = a tool call. With an empty script it calls the first bound
    tool once per human message, then answers with canned text. `structured` maps a schema name
    to the object `with_structured_output(schema)` should return; a list is consumed one item per
    call (the last item repeats). `calls` records the schema name of every structured call."""

    script: list[Any] = []
    structured: dict[str, Any] = {}
    tool_schemas: list[dict] = []
    calls: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, dict):
                call = {"name": item["tool"], "args": item.get("args", {}), "id": f"call_{uuid.uuid4().hex[:8]}"}
                message = AIMessage(content="", tool_calls=[call])
            else:
                message = AIMessage(content=str(item))
            return ChatResult(generations=[ChatGeneration(message=message)])
        last = next((m for m in reversed(messages) if m.type != "system"), None)
        if self.tool_schemas and last is not None and last.type == "human":
            schema = self.tool_schemas[0]["function"]
            props = schema.get("parameters", {}).get("properties", {})
            args = {k: {"integer": 1, "number": 1.0, "boolean": True}.get(v.get("type"), "test") for k, v in props.items()}
            call = {"name": schema["name"], "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Fake answer (LLM_PROVIDER=fake)."))])

    def bind_tools(self, tools, tool_choice=None, **kwargs):
        from langchain_core.utils.function_calling import convert_to_openai_tool

        self.tool_schemas = [convert_to_openai_tool(t) for t in tools]  # shared script: mutate, don't copy
        return self

    def with_structured_output(self, schema, **kwargs):
        def build(_input, **_ignored):
            name = getattr(schema, "__name__", "")
            self.calls.append(name)
            if name in self.structured:
                value = self.structured[name]
                if isinstance(value, list):
                    value = value.pop(0) if len(value) > 1 else value[0]
                return schema(**value) if isinstance(value, dict) else value
            return dummy_instance(schema) if isinstance(schema, type) else json.loads('{"result": "fake"}')

        return RunnableLambda(build)
