"""Request and response models for the FinDocGPT API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=10)


class SourcePayload(BaseModel):
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    rrf_score: float | None = None
    bm25_rank: int | None = None
    dense_rank: int | None = None
    reranker_score: float | None = None
    fused_rank: int | None = None
    reranked_rank: int | None = None


class ClaimVerdictPayload(BaseModel):
    claim: str
    supported: bool
    source_ids: list[str]
    rationale: str


class ComputationPayload(BaseModel):
    answer: str
    metric: str
    values_by_year: dict[int, float]
    calculation: str
    table_id: str
    table_title: str
    ticker: str


class ChatResponse(BaseModel):
    query: str
    route: str
    route_reason: str
    draft_answer: str
    answer: str
    unsupported_claims: list[str]
    sources: list[SourcePayload]
    verdicts: list[ClaimVerdictPayload]
    computation: ComputationPayload | None = None


class HealthResponse(BaseModel):
    status: str
    qdrant_url: str
    generation_model: str | None = None
    judge_model: str | None = None
