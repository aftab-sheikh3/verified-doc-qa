"""HTTP API.

    uvicorn verified_doc_qa.api:app --reload      # docs at http://127.0.0.1:8000/docs
    curl -F files=@corpus/leave-policy.md localhost:8000/ingest
    curl localhost:8000/ask -H 'Content-Type: application/json' -d '{"question": "How much annual leave do I get?"}'
"""
from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile

from . import __version__
from .graph import DocQA
from .ingest import SUPPORTED, load_file
from .schemas import AskRequest, AskResponse

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(title="Verified Doc QA", version=__version__,
              description="Agentic RAG that checks relevance, cites its sources and verifies every answer.")


@lru_cache(maxsize=1)
def qa() -> DocQA:
    return DocQA()


@app.get("/health")
def health() -> dict:
    s = qa().settings
    return {"status": "ok", "provider": s.provider, "model": s.model_name, "embeddings": s.embedding_model,
            "retrieval": s.retrieval, "reranker": qa().retriever.reranker}


@app.get("/documents")
def documents() -> dict[str, int]:
    return qa().store.sources()


@app.post("/ingest")
async def ingest(files: list[UploadFile]) -> dict[str, int]:
    s, counts = qa().settings, {}
    for upload in files:
        name = Path(upload.filename or "").name
        if Path(name).suffix.lower() not in SUPPORTED:
            raise HTTPException(415, f"{name!r}: supported types are {', '.join(sorted(SUPPORTED))}")
        data = await upload.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"{name!r} is larger than {MAX_UPLOAD_BYTES // 2**20} MB")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_bytes(data)
            counts[name] = qa().store.add(load_file(path, s.chunk_size, s.chunk_overlap, name=name))
    return counts


@app.delete("/documents/{source}")
def delete_document(source: str) -> dict:
    if source not in qa().store.sources():
        raise HTTPException(404, f"{source!r} is not indexed")
    qa().store.delete_source(source)
    return {"deleted": source}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if not qa().store.sources():
        raise HTTPException(409, "No documents indexed yet: POST /ingest or run `docqa ingest corpus/`")
    try:
        return qa().ask(request.question)
    except Exception as e:  # model server down, invalid key ...
        raise HTTPException(502, f"Model error: {type(e).__name__}: {e}") from e
