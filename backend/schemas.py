"""Request and response models for the FinDocGPT API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)


class Source(BaseModel):
    label: str
    chunk_id: str
    kind: str
    text: str
    ticker: str | None = None
    fiscal_year: int | None = None
    item: str | None = None
    subsection: str | None = None
    source_url: str | None = None
    rerank_score: float | None = None
    bm25_rank: int | None = None
    dense_rank: int | None = None
    fused_rank: int | None = None


class Verdict(BaseModel):
    sentence: str
    supported: bool
    citations: list[int]
    reason: str


class ChatResponse(BaseModel):
    question: str
    route: str
    plan: dict[str, Any]
    answer: str
    draft: str
    abstained: bool
    calculation: dict[str, Any] | None = None
    verdicts: list[Verdict]
    removed_claims: list[str]
    sources: list[Source]
    trace: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    vector_store: str
    generator_model: str
    agent_model: str
    embedding_model: str
