from langchain_core.documents import Document

from tests.conftest import CORPUS
from verified_doc_qa.ingest import iter_files, load_file, split_text
from verified_doc_qa.llm import HashingEmbeddings
from verified_doc_qa.store import ChunkStore, tokenize


def test_split_keeps_sections_and_stable_ids():
    text = "# Policy\n\n## Leave\nYou get 25 days.\n\n## Sick\nSelf-certify for 3 days."
    a, b = split_text(text, "p.md"), split_text(text, "p.md")
    assert [c.metadata["section"] for c in a] == ["Policy > Leave", "Policy > Sick"]
    assert [c.metadata["chunk_id"] for c in a] == [c.metadata["chunk_id"] for c in b]
    assert all(c.metadata["source"] == "p.md" for c in a)


def test_long_sections_are_split_by_size():
    chunks = split_text("# T\n\n" + "word " * 1000, "long.md", chunk_size=300, chunk_overlap=30)
    assert len(chunks) > 5 and all(len(c.page_content) <= 300 for c in chunks)


def test_iter_files_skips_readme():
    names = {p.name for p in iter_files([CORPUS])}
    assert "README.md" not in names and "leave-policy.md" in names


def test_load_file_rejects_unknown_types(tmp_path):
    (tmp_path / "x.docx").write_text("hi")
    try:
        load_file(tmp_path / "x.docx")
    except ValueError as e:
        assert "Unsupported" in str(e)
    else:
        raise AssertionError("expected ValueError")


def _doc(text, source="a.md", cid=None):
    return Document(page_content=text, metadata={"source": source, "section": "", "page": None,
                                                  "chunk_id": cid or text[:8].ljust(8, "x")})


def test_store_add_search_replace_delete():
    store = ChunkStore(HashingEmbeddings(), "t", path=":memory:")
    store.add([_doc("annual leave is 25 days", cid="a" * 32), _doc("hotel limit is 180 EUR", cid="b" * 32)])
    assert store.sources() == {"a.md": 2}
    assert store.dense("how much annual leave", 1)[0].page_content.startswith("annual leave")
    assert store.bm25("hotel limit", 1)[0].page_content.startswith("hotel")
    store.add([_doc("replacement text", cid="c" * 32)])  # re-ingesting a source replaces its chunks
    assert store.sources() == {"a.md": 1}
    store.delete_source("a.md")
    assert store.sources() == {} and store.bm25("hotel", 3) == []


def test_tokenize_drops_stopwords():
    assert tokenize("How many days of leave do I get?") == ["days", "leave", "get"]


def test_bm25_scores_rare_terms_higher_and_never_negative():
    from verified_doc_qa.store import BM25

    bm = BM25([["leave", "days"], ["leave", "hotel"], ["leave", "pension"]])
    scores = bm.get_scores(["hotel", "leave"])
    assert scores[1] > scores[0] > 0 and min(bm.idf.values()) > 0
