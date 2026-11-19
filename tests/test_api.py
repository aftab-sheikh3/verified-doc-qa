import pytest
from fastapi.testclient import TestClient

from verified_doc_qa import api
from verified_doc_qa.graph import DocQA
from verified_doc_qa.llm import HashingEmbeddings, ScriptedChatModel
from verified_doc_qa.retrieve import Retriever
from verified_doc_qa.store import ChunkStore


@pytest.fixture
def client(settings, monkeypatch):
    store = ChunkStore(HashingEmbeddings(), "api", path=":memory:")
    llm = ScriptedChatModel(structured={
        "Relevance": {"label": "CAN_ANSWER"}, "Verification": {"supported": True, "relevant": True},
        "CitedAnswer": {"answerable": True, "answer": "The hotel limit in London is 250 EUR [1].", "citations": [1]}})
    app = DocQA(settings, llm=llm, store=store, retriever=Retriever(store, "hybrid", "none", top_n=3))
    monkeypatch.setattr(api, "qa", lambda: app)
    return TestClient(api.app)


def test_ask_before_ingest_is_409(client):
    assert client.post("/ask", json={"question": "hotel limit?"}).status_code == 409


def test_ingest_ask_delete(client):
    text = b"# Travel\n\n## Hotels\nThe nightly hotel limit is 180 EUR, and 250 EUR in London."
    r = client.post("/ingest", files=[("files", ("travel.md", text, "text/markdown"))])
    assert r.status_code == 200 and r.json() == {"travel.md": 1}
    assert client.get("/documents").json() == {"travel.md": 1}
    body = client.post("/ask", json={"question": "What is the hotel limit in London?"}).json()
    assert body["status"] == "answered" and body["citations"][0]["source"] == "travel.md"
    assert body["verification"]["passed"] is True
    assert client.delete("/documents/travel.md").status_code == 200
    assert client.get("/documents").json() == {}


def test_rejects_bad_uploads_and_questions(client):
    assert client.post("/ingest", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))]).status_code == 415
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.delete("/documents/nope.md").status_code == 404
