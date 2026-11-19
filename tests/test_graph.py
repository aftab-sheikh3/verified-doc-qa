from verified_doc_qa.graph import REFUSAL, DocQA, cited_numbers
from verified_doc_qa.llm import ScriptedChatModel
from verified_doc_qa.schemas import CitedAnswer

Q = "How many days of annual leave do full-time employees get?"
GOOD = {"answerable": True, "answer": "Full-time employees get 25 days of paid annual leave [1].", "citations": [1]}
OK = {"supported": True, "relevant": True}
BAD = {"supported": False, "relevant": True, "unsupported_claims": ["30 days"], "feedback": "the sources say 25"}


def qa(settings, retriever, **structured):
    llm = ScriptedChatModel(structured=structured)
    return DocQA(settings, llm=llm, store=retriever.store, retriever=retriever), llm


def test_answered_and_cited(settings, retriever):
    app, llm = qa(settings, retriever, Relevance={"label": "CAN_ANSWER"}, CitedAnswer=GOOD, Verification=OK)
    resp = app.ask(Q)
    assert resp.status == "answered" and resp.attempts == 1 and resp.verification.passed
    assert resp.citations[0].source == "leave-policy.md" and "25 days" in resp.citations[0].text
    assert [s.node for s in resp.steps] == ["retrieve", "grade", "generate", "verify", "finish"]
    assert llm.calls == ["Relevance", "CitedAnswer", "Verification"]


def test_no_match_is_refused_without_generating(settings, retriever):
    app, llm = qa(settings, retriever, Relevance={"label": "NO_MATCH"})
    resp = app.ask("What is the capital of France?")
    assert resp.status == "refused" and resp.answer == REFUSAL and resp.citations == []
    assert llm.calls == ["Relevance"]


def test_generator_says_unanswerable(settings, retriever):
    app, _ = qa(settings, retriever, Relevance={"label": "PARTIAL"},
                CitedAnswer={"answerable": False, "answer": "Not in the sources."})
    assert app.ask("What is the dress code?").status == "refused"


def test_failed_verification_retries_with_feedback(settings, retriever):
    wrong = {**GOOD, "answer": "Full-time employees get 30 days [1]."}
    app, llm = qa(settings, retriever, Relevance={"label": "CAN_ANSWER"}, CitedAnswer=[wrong, GOOD],
                  Verification=[BAD, OK])
    resp = app.ask(Q)
    assert resp.status == "answered" and resp.attempts == 2 and "25" in resp.answer
    assert llm.calls == ["Relevance", "CitedAnswer", "Verification", "CitedAnswer", "Verification"]


def test_out_of_attempts_is_flagged_unverified(settings, retriever):
    app, _ = qa(settings, retriever, Relevance={"label": "CAN_ANSWER"}, CitedAnswer=GOOD, Verification=BAD)
    resp = app.ask(Q)
    assert resp.status == "unverified" and resp.attempts == settings.max_attempts
    assert resp.verification.unsupported_claims == ["30 days"]


def test_answer_without_citations_never_passes(settings, retriever):
    app, _ = qa(settings, retriever, Relevance={"label": "CAN_ANSWER"},
                CitedAnswer={"answerable": True, "answer": "25 days.", "citations": []}, Verification=OK)
    resp = app.ask(Q)
    assert resp.status == "unverified" and "cites no sources" in resp.verification.feedback


def test_grader_failure_assumes_partial(settings, retriever):
    class Boom(ScriptedChatModel):
        def with_structured_output(self, schema, **kw):
            if schema.__name__ == "Relevance":
                raise ValueError("invalid json")
            return super().with_structured_output(schema, **kw)

    llm = Boom(structured={"CitedAnswer": GOOD, "Verification": OK})
    resp = DocQA(settings, llm=llm, store=retriever.store, retriever=retriever).ask(Q)
    assert resp.relevance == "PARTIAL" and resp.status == "answered"


def test_cited_numbers_drop_invented_sources():
    ans = CitedAnswer(answerable=True, answer="A [1]. B [7]. C [2][3].", citations=[1, 9])
    assert cited_numbers(ans, 3) == [1, 2, 3]
