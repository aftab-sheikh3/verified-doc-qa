from evals.run_eval import first_hit, has_key_facts, load_cases, retrieval_metrics, run_end_to_end, summarise_e2e
from verified_doc_qa.graph import DocQA
from verified_doc_qa.llm import ScriptedChatModel


def test_cases_are_valid():
    cases = load_cases()
    assert len({c["id"] for c in cases}) == len(cases)
    assert sum(not c["answerable"] for c in cases) >= 5
    assert all(c["evidence"] and c["key_facts"] and c["gold_source"] for c in cases if c["answerable"])


def test_metric_helpers():
    assert first_hit("25 days  of leave", ["nothing", "You get 25 days\nof leave."]) == 2
    assert first_hit("x", ["a", "b"]) is None
    m = retrieval_metrics([1, 2, None, 6])
    assert m["hit@1"] == 0.25 and m["recall@5"] == 0.5 and abs(m["mrr@10"] - (1 + 0.5 + 1 / 6) / 4) < 1e-9
    assert has_key_facts("Up to 5 days, expiring 31  March.", ["5", "31 March"])
    assert not has_key_facts("Up to 5 days.", ["5", "31 March"])


def test_end_to_end_scoring(settings, retriever):
    llm = ScriptedChatModel(structured={
        "Relevance": [{"label": "CAN_ANSWER"}, {"label": "NO_MATCH"}],
        "CitedAnswer": {"answerable": True, "answer": "25 days of paid annual leave [1].", "citations": [1]},
        "Verification": {"supported": True, "relevant": True}})
    qa = DocQA(settings, llm=llm, store=retriever.store, retriever=retriever)
    cases = [c for c in load_cases() if c["id"] in ("qa-01", "un-01")]
    results = run_end_to_end(qa, cases, judge_llm=None)
    summary = summarise_e2e(results)
    assert summary["refusal_accuracy"] == 1.0 and summary["key_fact_accuracy"] == 1.0
    assert summary["citation_precision"] == 1.0 and summary["verified_rate"] == 1.0
