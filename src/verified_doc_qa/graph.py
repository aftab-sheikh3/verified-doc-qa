"""The verified-answer workflow as a LangGraph graph.

    retrieve -> grade ──NO_MATCH──────────────────────────────> refuse
                  └─ CAN_ANSWER / PARTIAL -> generate -> verify ─ passed ──────────> finish (answered)
                                                ^          └─ failed, attempts left ─┐
                                                └──────────── feedback ──────────────┘
                                                           └─ failed, no attempts ──> finish (unverified)

- grade: a relevance gate, so off-topic questions are refused instead of answered from thin air.
- generate: structured output (answerable, answer, citations). Citations that point outside the
  retrieved set are dropped. If the model says the sources don't answer it, the reply is a refusal.
- verify: a second, deterministic pass fact-checks the answer against the cited sources. On failure
  the generator retries with the verifier's feedback; when attempts run out, the answer is returned
  flagged *unverified*, never silently.
"""
from __future__ import annotations

import re
import time
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from .config import Settings, get_settings
from .llm import get_chat_model, get_embeddings
from .retrieve import Retriever
from .schemas import AskResponse, CitedAnswer, Relevance, Source, Step, Verification
from .store import ChunkStore

REFUSAL = "I can't answer that from the indexed documents."

GRADE_PROMPT = """Classify whether the passages can answer the question.
If the passages mention the question's topic at all, even incompletely, choose PARTIAL rather than NO_MATCH.

Question: {question}

Passages:
{passages}"""

ANSWER_PROMPT = """Answer the question using only the numbered sources.
- Cite the sources you use inline, like [1] or [2][3], right after the sentence they support.
- Be concise and factual; quote numbers exactly as the sources state them.
- If the sources do not contain the answer, set answerable to false.
{partial}{feedback}
Question: {question}

Sources:
{sources}"""

VERIFY_PROMPT = """You are a strict fact-checker. Check the answer against the sources only (no outside knowledge).
List every claim the sources do not support and every claim they contradict.

Question: {question}

Answer: {answer}

Sources:
{sources}"""


def number_sources(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        where = d.metadata.get("source", "?")
        if d.metadata.get("section"):
            where += f" - {d.metadata['section']}"
        if isinstance(d.metadata.get("page"), int):
            where += f", page {d.metadata['page'] + 1}"
        blocks.append(f"[{i}] ({where})\n{d.page_content}")
    return "\n\n".join(blocks)


def cited_numbers(answer: CitedAnswer, n_sources: int) -> list[int]:
    """Citations from the structured field and from [n] markers in the text, restricted to real sources."""
    found = set(answer.citations) | {int(m) for m in re.findall(r"\[(\d+)\]", answer.answer)}
    return sorted(n for n in found if 1 <= n <= n_sources)


def to_source(n: int, doc: Document) -> Source:
    m = doc.metadata
    return Source(n=n, source=m.get("source", "?"), section=m.get("section") or "", page=m.get("page"),
                  chunk_id=m.get("chunk_id", ""), text=doc.page_content,
                  score=m.get("rerank_score", m.get("score")))


class QAState(TypedDict, total=False):
    question: str
    docs: list[Document]
    relevance: str
    draft: CitedAnswer
    cited: list[int]
    verification: Verification
    attempts: int
    status: str
    steps: list[dict]


def build_store(settings: Settings) -> ChunkStore:
    return ChunkStore(get_embeddings(settings), settings.collection, url=settings.qdrant_url, path=settings.index_dir)


class DocQA:
    def __init__(self, settings: Settings | None = None, llm: BaseChatModel | None = None,
                 store: ChunkStore | None = None, retriever: Retriever | None = None):
        self.settings = s = settings or get_settings()
        self.llm = llm or get_chat_model(s)
        self.store = store or build_store(s)
        self.retriever = retriever or Retriever(self.store, s.retrieval, s.reranker, s.reranker_model,
                                                s.candidates_k, s.top_n, s.hybrid_weights)
        self.graph = self._build()

    # ------------------------------------------------------------------ nodes
    def _retrieve(self, state: QAState) -> dict:
        docs = self.retriever.invoke(state["question"])
        return {"docs": docs, "attempts": 0, "detail": f"{len(docs)} chunks"}

    def _grade(self, state: QAState) -> dict:
        if not state["docs"]:
            return {"relevance": "NO_MATCH", "detail": "nothing retrieved"}
        prompt = GRADE_PROMPT.format(question=state["question"], passages=number_sources(state["docs"][:3]))
        try:
            label = self.llm.with_structured_output(Relevance).invoke(prompt).label
        except Exception as e:  # don't refuse just because a small model produced invalid JSON
            return {"relevance": "PARTIAL", "detail": f"grader failed ({type(e).__name__}); assuming PARTIAL"}
        return {"relevance": label, "detail": label}

    def _generate(self, state: QAState) -> dict:
        previous = state.get("verification")
        feedback = ""
        if previous is not None and not previous.passed:
            problems = previous.unsupported_claims + previous.contradictions
            feedback = ("\nA previous answer failed fact-checking. Fix these problems: "
                        f"{'; '.join(problems) or previous.feedback or 'unsupported claims'}\n")
        partial = ("\nThe sources may answer only part of the question: answer that part and say what is missing.\n"
                   if state.get("relevance") == "PARTIAL" else "")
        prompt = ANSWER_PROMPT.format(question=state["question"], sources=number_sources(state["docs"]),
                                      partial=partial, feedback=feedback)
        draft = self.llm.with_structured_output(CitedAnswer).invoke(prompt)
        cited = cited_numbers(draft, len(state["docs"]))
        return {"draft": draft, "cited": cited, "attempts": state.get("attempts", 0) + 1,
                "detail": f"attempt {state.get('attempts', 0) + 1}, cites {cited}"}

    def _verify(self, state: QAState) -> dict:
        docs = state["docs"]
        cited = [docs[n - 1] for n in state["cited"]] or docs  # check against what it cites (or everything)
        prompt = VERIFY_PROMPT.format(question=state["question"], answer=state["draft"].answer,
                                      sources=number_sources(cited))
        try:
            v = self.llm.with_structured_output(Verification).invoke(prompt)
        except Exception as e:
            v = Verification(supported=False, relevant=False, feedback=f"verifier failed: {type(e).__name__}")
        if not state["cited"]:
            v = v.model_copy(update={"supported": False, "feedback": (v.feedback + " The answer cites no sources.").strip()})
        return {"verification": v, "detail": "passed" if v.passed else f"failed: {v.unsupported_claims + v.contradictions}"}

    def _refuse(self, state: QAState) -> dict:
        return {"status": "refused", "detail": state.get("relevance", "")}

    def _finish(self, state: QAState) -> dict:
        return {"status": "answered" if state["verification"].passed else "unverified"}

    # ------------------------------------------------------------------ edges
    @staticmethod
    def _after_grade(state: QAState) -> str:
        return "refuse" if state["relevance"] == "NO_MATCH" else "generate"

    @staticmethod
    def _after_generate(state: QAState) -> str:
        return "refuse" if not state["draft"].answerable else "verify"

    def _after_verify(self, state: QAState) -> str:
        if state["verification"].passed or state["attempts"] >= self.settings.max_attempts:
            return "finish"
        return "generate"

    def _timed(self, name: str, fn):
        def node(state: QAState) -> dict:
            start = time.perf_counter()
            update = fn(state)
            step = {"node": name, "seconds": round(time.perf_counter() - start, 3), "detail": str(update.pop("detail", ""))}
            return {**update, "steps": [*state.get("steps", []), step]}

        return node

    def _build(self):
        g = StateGraph(QAState)
        for name, fn in [("retrieve", self._retrieve), ("grade", self._grade), ("generate", self._generate),
                         ("verify", self._verify), ("refuse", self._refuse), ("finish", self._finish)]:
            g.add_node(name, self._timed(name, fn))
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", self._after_grade, ["refuse", "generate"])
        g.add_conditional_edges("generate", self._after_generate, ["refuse", "verify"])
        g.add_conditional_edges("verify", self._after_verify, ["generate", "finish"])
        g.add_edge("refuse", END)
        g.add_edge("finish", END)
        return g.compile()

    # ------------------------------------------------------------------ public
    def ask(self, question: str) -> AskResponse:
        start = time.perf_counter()
        state: dict[str, Any] = self.graph.invoke({"question": question}, config={"recursion_limit": 25})
        docs = state.get("docs", [])
        status = state.get("status", "refused")
        draft: CitedAnswer | None = state.get("draft")
        answer = draft.answer if draft is not None and status != "refused" else REFUSAL
        cited = state.get("cited", []) if status != "refused" else []
        return AskResponse(
            question=question, status=status, answer=answer, relevance=state.get("relevance", "NO_MATCH"),
            citations=[to_source(n, docs[n - 1]) for n in cited],
            retrieved=[to_source(i, d) for i, d in enumerate(docs, 1)],
            verification=state.get("verification"), attempts=state.get("attempts", 0),
            steps=[Step(**s) for s in state.get("steps", [])], seconds=round(time.perf_counter() - start, 2))
