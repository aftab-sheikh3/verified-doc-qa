"""Structured outputs of the agents, and the API's request/response models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field


class Relevance(BaseModel):
    """Can the retrieved passages answer the question?"""

    label: Literal["CAN_ANSWER", "PARTIAL", "NO_MATCH"] = Field(
        description="CAN_ANSWER: the passages contain enough explicit information for a full answer. "
                    "PARTIAL: they discuss the topic but miss some details. "
                    "NO_MATCH: they do not discuss the question's topic at all.")
    reason: str = Field("", description="One short sentence")


class CitedAnswer(BaseModel):
    """An answer grounded in numbered sources."""

    answerable: bool = Field(description="False if the sources do not contain the answer")
    answer: str = Field(description="The answer, citing sources inline like [1] or [2][3]")
    citations: list[int] = Field(default_factory=list, description="Numbers of the sources the answer relies on")


class Verification(BaseModel):
    """Fact-check of an answer against its sources."""

    supported: bool = Field(description="Every factual claim in the answer is supported by the sources")
    unsupported_claims: list[str] = Field(default_factory=list, description="Claims the sources do not support")
    contradictions: list[str] = Field(default_factory=list, description="Claims the sources contradict")
    relevant: bool = Field(description="The answer addresses the question")
    feedback: str = Field("", description="If not supported: what to fix, in one or two sentences")

    @computed_field
    @property
    def passed(self) -> bool:
        return self.supported and self.relevant and not self.unsupported_claims and not self.contradictions


class Source(BaseModel):
    n: int
    source: str
    section: str = ""
    page: int | None = None
    chunk_id: str
    text: str
    score: float | None = None


class Step(BaseModel):
    node: str
    seconds: float
    detail: str = ""


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000, examples=["How many days of annual leave do I get?"])


class AskResponse(BaseModel):
    question: str
    status: Literal["answered", "unverified", "refused"]
    answer: str
    relevance: str
    citations: list[Source] = []   # the sources the answer cites
    retrieved: list[Source] = []   # everything the answer was allowed to use
    verification: Verification | None = None
    attempts: int = 0
    steps: list[Step] = []
    seconds: float = 0.0
