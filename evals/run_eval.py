"""Evaluation: a retrieval ablation and end-to-end answer quality on the bundled corpus.

    python -m evals.run_eval                    # or: docqa eval  /  make eval
    python -m evals.run_eval --limit 5 --no-judge

The corpus is indexed into a fresh in-memory collection with the configured embeddings, so the
eval never touches your own index.

1. Retrieval ablation (answerable cases): bm25 | dense | hybrid (RRF) | hybrid + cross-encoder rerank.
   A hit is the first retrieved chunk that contains the case's evidence sentence.
   Metrics: hit@1, recall@5, MRR@10.
2. End to end (the full graph with the configured retrieval):
   - refusal accuracy: unanswerable questions are refused, answerable ones are not
   - key-fact accuracy: every key fact (e.g. "25", "31 March") appears in the answer
   - citation precision: the share of cited chunks that come from the gold document
   - verified rate: the share of answers that passed the verifier
   - LLM judge (1-5): faithfulness to the retrieved context, correctness against the gold answer
   - attempts and latency
Writes evals/results.json (every case) and evals/results.md (tables for the README).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from verified_doc_qa.config import ROOT, Settings, get_settings
from verified_doc_qa.graph import DocQA
from verified_doc_qa.ingest import index_paths
from verified_doc_qa.llm import get_chat_model, get_embeddings
from verified_doc_qa.retrieve import Retriever
from verified_doc_qa.store import ChunkStore

HERE = Path(__file__).parent
CORPUS = ROOT / "corpus"


def load_cases(limit: int | None = None) -> list[dict]:
    cases = [json.loads(line) for line in (HERE / "qa_cases.jsonl").read_text().splitlines() if line.strip()]
    if limit is None:
        return cases
    answerable = [c for c in cases if c["answerable"]][:limit]
    return answerable + [c for c in cases if not c["answerable"]][: max(1, limit // 3)]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def first_hit(evidence: str, texts: list[str]) -> int | None:
    """1-based rank of the first text containing the evidence, or None."""
    ev = norm(evidence)
    return next((i for i, t in enumerate(texts, 1) if ev in norm(t)), None)


def retrieval_metrics(ranks: list[int | None]) -> dict:
    n = len(ranks) or 1
    return {"hit@1": sum(r == 1 for r in ranks) / n,
            "recall@5": sum(r is not None and r <= 5 for r in ranks) / n,
            "mrr@10": sum(1 / r for r in ranks if r is not None and r <= 10) / n}


def has_key_facts(answer: str, facts: list[str]) -> bool:
    a = norm(answer)
    return all(norm(f) in a for f in facts)


class Judgement(BaseModel):
    faithfulness: int = Field(ge=1, le=5, description="1-5: every claim in the answer is supported by the context")
    correctness: int = Field(ge=1, le=5, description="1-5: the answer agrees with the reference answer")
    reasoning: str = Field("", description="One sentence")


JUDGE_PROMPT = """Grade the answer of a document QA system.
Question: {question}
Reference answer: {gold}
Retrieved context:
{context}

Answer to grade: {answer}"""


def judge(llm: BaseChatModel, case: dict, answer: str, context: str) -> Judgement | None:
    try:
        return llm.with_structured_output(Judgement).invoke(JUDGE_PROMPT.format(
            question=case["question"], gold=case["gold_answer"], context=context[:6000], answer=answer))
    except Exception:
        return None


def build_eval_store(settings: Settings) -> ChunkStore:
    store = ChunkStore(get_embeddings(settings), "eval", path=":memory:")
    index_paths(store, [CORPUS], settings.chunk_size, settings.chunk_overlap)
    return store


def run_ablation(store: ChunkStore, cases: list[dict], settings: Settings) -> list[dict]:
    configs = [("bm25", "bm25", "none"), ("dense", "dense", "none"), ("hybrid (RRF)", "hybrid", "none"),
               ("hybrid + rerank", "hybrid", "cross-encoder")]
    rows = []
    for label, mode, reranker in configs:
        r = Retriever(store, mode, reranker, settings.reranker_model, candidates_k=20, top_n=10,
                      weights=settings.hybrid_weights)
        if reranker != "none" and r.reranker == "none":
            rows.append({"config": label, "skipped": "sentence-transformers not installed"})
            continue
        start = time.perf_counter()
        try:
            ranks = [first_hit(c["evidence"], [d.page_content for d in r.invoke(c["question"])]) for c in cases]
        except Exception as e:  # e.g. reranker weights cannot be downloaded
            rows.append({"config": label, "skipped": f"{type(e).__name__}: {e}"[:120]})
            continue
        ms = (time.perf_counter() - start) * 1000 / max(1, len(cases))
        rows.append({"config": label, **retrieval_metrics(ranks), "ms_per_query": round(ms, 1)})
    return rows


def run_end_to_end(qa: DocQA, cases: list[dict], judge_llm: BaseChatModel | None) -> list[dict]:
    results = []
    for case in cases:
        start = time.perf_counter()
        try:
            resp = qa.ask(case["question"])
        except Exception as e:
            results.append({"id": case["id"], "question": case["question"], "answerable": case["answerable"],
                            "error": f"{type(e).__name__}: {e}", "seconds": round(time.perf_counter() - start, 2)})
            continue
        row = {"id": case["id"], "question": case["question"], "answerable": case["answerable"],
               "status": resp.status, "answer": resp.answer, "attempts": resp.attempts,
               "cited": [c.source for c in resp.citations], "seconds": resp.seconds, "error": None,
               "refusal_correct": (resp.status == "refused") == (not case["answerable"])}
        if case["answerable"] and resp.status != "refused":
            row["key_facts"] = has_key_facts(resp.answer, case["key_facts"])
            row["citation_precision"] = (sum(s == case["gold_source"] for s in row["cited"]) / len(row["cited"])
                                         if row["cited"] else 0.0)
            row["verified"] = resp.status == "answered"
            if judge_llm is not None:
                j = judge(judge_llm, case, resp.answer, "\n\n".join(s.text for s in resp.retrieved))
                if j is not None:
                    row["faithfulness"], row["correctness"] = j.faithfulness, j.correctness
        elif case["answerable"]:
            row["key_facts"] = False
        results.append(row)
    return results


def _mean(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.mean(values), 3) if values else None


def summarise_e2e(results: list[dict]) -> dict:
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]
    secs = sorted(r["seconds"] for r in results)
    return {
        "cases": len(results), "errors": sum(bool(r.get("error")) for r in results),
        "refusal_accuracy": _mean(results, "refusal_correct"),
        "refused_unanswerable": _mean(unanswerable, "refusal_correct"),
        "answered_answerable": _mean(answerable, "refusal_correct"),
        "key_fact_accuracy": _mean(answerable, "key_facts"),
        "citation_precision": _mean(answerable, "citation_precision"),
        "verified_rate": _mean(answerable, "verified"),
        "faithfulness_1to5": _mean(answerable, "faithfulness"),
        "correctness_1to5": _mean(answerable, "correctness"),
        "mean_attempts": _mean(results, "attempts"),
        "median_seconds": secs[len(secs) // 2] if secs else None,
        "p95_seconds": secs[max(0, int(len(secs) * 0.95) - 1)] if secs else None,
    }


def _pct(x) -> str:
    return "-" if x is None else f"{x:.0%}"


def to_markdown(ablation: list[dict], e2e: dict, settings: Settings, reranker: str) -> str:
    lines = [f"Chat `{settings.provider}:{settings.model_name}` · embeddings `{settings.embedding_provider}:"
             f"{settings.embedding_model}` · run {time.strftime('%Y-%m-%d')}", "",
             "**Retrieval ablation**", "", "| Retrieval | hit@1 | recall@5 | MRR@10 | ms/query |", "|---|---|---|---|---|"]
    for row in ablation:
        if "skipped" in row:
            lines.append(f"| {row['config']} | skipped: {row['skipped']} | | | |")
        else:
            lines.append(f"| {row['config']} | {_pct(row['hit@1'])} | {_pct(row['recall@5'])} | "
                         f"{row['mrr@10']:.3f} | {row['ms_per_query']} |")
    fmt = lambda v: "-" if v is None else v  # noqa: E731
    lines += ["", f"**End to end** ({e2e['cases']} questions, retrieval `{settings.retrieval}`, "
                  f"reranker `{reranker}`)", "", "| Metric | Value |", "|---|---|",
              f"| Refusal accuracy (all) | {_pct(e2e['refusal_accuracy'])} |",
              f"| - unanswerable correctly refused | {_pct(e2e['refused_unanswerable'])} |",
              f"| - answerable not refused | {_pct(e2e['answered_answerable'])} |",
              f"| Key-fact accuracy | {_pct(e2e['key_fact_accuracy'])} |",
              f"| Citation precision | {_pct(e2e['citation_precision'])} |",
              f"| Verified rate | {_pct(e2e['verified_rate'])} |",
              f"| Judge faithfulness (1-5) | {fmt(e2e['faithfulness_1to5'])} |",
              f"| Judge correctness (1-5) | {fmt(e2e['correctness_1to5'])} |",
              f"| Mean attempts | {fmt(e2e['mean_attempts'])} |",
              f"| Median / p95 latency | {fmt(e2e['median_seconds'])}s / {fmt(e2e['p95_seconds'])}s |",
              f"| Errors | {e2e['errors']} |"]
    return "\n".join(lines) + "\n"


def run(limit: int | None = None, use_judge: bool = True, settings: Settings | None = None,
        llm: BaseChatModel | None = None) -> dict:
    settings = settings or get_settings()
    cases = load_cases(limit)
    store = build_eval_store(settings)
    ablation = run_ablation(store, [c for c in cases if c["answerable"]], settings)
    llm = llm or get_chat_model(settings)
    qa = DocQA(settings, llm=llm, store=store)
    results = run_end_to_end(qa, cases, llm if use_judge else None)
    e2e = summarise_e2e(results)
    report = {"provider": settings.provider, "model": settings.model_name, "embeddings": settings.embedding_model,
              "ablation": ablation, "end_to_end": e2e, "results": results}
    (HERE / "results.json").write_text(json.dumps(report, indent=2, default=str))
    markdown = to_markdown(ablation, e2e, settings, qa.retriever.reranker)
    (HERE / "results.md").write_text(markdown)
    for r in results:
        print(f"{'OK  ' if r.get('refusal_correct') and r.get('key_facts', True) else 'MISS'} "
              f"{r['seconds']:6.2f}s  {r.get('status', 'error'):10s} {r['question']}")
    print("\n" + markdown)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Verified Doc QA evaluation")
    parser.add_argument("--limit", type=int, help="first N answerable cases (+ a few unanswerable)")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM-judge scores")
    args = parser.parse_args()
    run(args.limit, use_judge=not args.no_judge)


if __name__ == "__main__":
    main()
