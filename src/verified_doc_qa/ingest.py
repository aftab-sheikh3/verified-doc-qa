"""Load documents and cut them into chunks that keep their place in the document.

- Markdown/text are split at headings (`#`, `##`, `###`), so a chunk knows its section; long
  sections are split again by size with overlap.
- PDFs are read page by page with pypdf (or with Docling, if installed, which keeps tables
  and handles scanned pages); every chunk keeps its page number for citations.
- Optional *contextual retrieval*: an LLM writes one sentence situating each chunk in its
  document, which is prepended before indexing (one LLM call per chunk).
- Chunk ids are content hashes, so re-ingesting the same file replaces rather than duplicates.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

SUPPORTED = {".md", ".txt", ".pdf"}
HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
SKIP_NAMES = {"readme.md"}

CONTEXT_PROMPT = """<document>
{document}
</document>
Here is a chunk from the document:
<chunk>
{chunk}
</chunk>
Write one short sentence that situates this chunk within the overall document, to improve search retrieval of
the chunk. Answer only with the sentence."""


def chunk_id(source: str, text: str) -> str:
    return hashlib.sha256(f"{source}\n{text}".encode()).hexdigest()[:32]


def _section(meta: dict) -> str:
    return " > ".join(meta[k] for k in ("h1", "h2", "h3") if meta.get(k))


def _pdf_pages(path: Path) -> list[tuple[int, str]]:
    try:
        from docling.document_converter import DocumentConverter  # optional extra

        return [(0, DocumentConverter().convert(str(path)).document.export_to_markdown())]
    except ImportError:
        from pypdf import PdfReader

        return [(i, page.extract_text() or "") for i, page in enumerate(PdfReader(str(path)).pages)]


def split_text(text: str, source: str, chunk_size: int = 800, chunk_overlap: int = 100,
               page: int | None = None) -> list[Document]:
    sections = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False).split_text(text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in splitter.split_documents(sections):
        if not doc.page_content.strip():
            continue
        meta = {"source": source, "section": _section(doc.metadata), "page": page}
        meta["chunk_id"] = chunk_id(source, doc.page_content)
        chunks.append(Document(page_content=doc.page_content, metadata=meta))
    return chunks


def load_file(path: str | Path, chunk_size: int = 800, chunk_overlap: int = 100, name: str | None = None) -> list[Document]:
    path = Path(path)
    source = name or path.name
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported file type {path.suffix!r}; use {', '.join(sorted(SUPPORTED))}")
    if path.suffix.lower() == ".pdf":
        chunks = []
        for page, text in _pdf_pages(path):
            chunks += split_text(text, source, chunk_size, chunk_overlap, page=page)
        return chunks
    return split_text(path.read_text(encoding="utf-8", errors="ignore"), source, chunk_size, chunk_overlap)


def iter_files(paths: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED and f.name.lower() not in SKIP_NAMES)
        elif p.exists():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return files


def contextualize(chunks: list[Document], document_text: str, llm, max_doc_chars: int = 12000) -> list[Document]:
    doc = document_text[:max_doc_chars]
    for chunk in chunks:
        context = llm.invoke(CONTEXT_PROMPT.format(document=doc, chunk=chunk.page_content)).content.strip()
        chunk.metadata["context"] = context
        chunk.page_content = f"{context}\n{chunk.page_content}"
    return chunks


def index_paths(store, paths: list[str | Path], chunk_size: int = 800, chunk_overlap: int = 100,
                contextual_llm=None) -> dict[str, int]:
    """Chunk every file under `paths` and add it to `store`. Returns {source: number of chunks}."""
    counts = {}
    for path in iter_files(paths):
        chunks = load_file(path, chunk_size, chunk_overlap)
        if contextual_llm is not None and chunks:
            full_text = "\n\n".join(c.page_content for c in chunks)
            chunks = contextualize(chunks, full_text, contextual_llm)
        counts[path.name] = store.add(chunks)
    return counts
