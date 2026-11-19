import pytest
from langchain_core.documents import Document

from verified_doc_qa.retrieve import RRF_K, Retriever, rrf


def d(cid):
    return Document(page_content=cid, metadata={"chunk_id": cid})


def test_rrf_maths():
    fused = rrf([[d("a"), d("b")], [d("b"), d("c")]], (1.0, 1.0))
    assert [x.metadata["chunk_id"] for x in fused] == ["b", "a", "c"]
    assert fused[0].metadata["score"] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))


def test_rrf_weights_change_the_winner():
    lists = [[d("a"), d("b")], [d("b"), d("a")]]
    assert rrf(lists, (0.9, 0.1))[0].metadata["chunk_id"] == "a"
    assert rrf(lists, (0.1, 0.9))[0].metadata["chunk_id"] == "b"


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_modes_find_the_right_document(corpus_store, mode):
    r = Retriever(corpus_store, mode, "none", top_n=3)
    assert r.invoke("What is the hotel limit in London?")[0].metadata["source"] == "travel.md"


def test_missing_reranker_falls_back(corpus_store, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert Retriever(corpus_store, "hybrid", "cross-encoder").reranker == "none"


def test_bad_mode():
    with pytest.raises(ValueError):
        Retriever(None, "magic")
